import re
import serial
import tvm

import numpy as np
import sys

sid_decl_re = "void\* sid_([0-9]+)_let = \(&\(global_workspace_[0-9]+_var\[([0-9]+)\]\)\);"
sid_re = "sid_([0-9]+)_let"
tvm_func_out_to_sid = dict()
tvm_stdfunc_out_to_sid = dict()
std_func_list = list()
std_calls_count = dict()


def manage_tvmfunc_calls(sids, tvm_lineno, tvm_line, tvm_end_sid_decl_lineno):
    tvm_sids = re.findall(sid_re, tvm_line)
    for tvm_call_idx, tvm_sid in enumerate(tvm_sids):
        if sids[tvm_sid]["first"] == -1:
            sids[tvm_sid]["first"] = tvm_lineno - tvm_end_sid_decl_lineno
        sids[tvm_sid]["last"] = tvm_lineno - tvm_end_sid_decl_lineno
        if any([f"sid_{tvm_sid}_let" in line_ for line_ in tvm_line.split(",")[1:]]):
            f_name = tvm_line.split("(")[1]
            sids[tvm_sid]["output"] = f_name
            tvm_func_out_to_sid[f_name] = tvm_sid
            if "match" not in f_name:
                tvm_stdfunc_out_to_sid[f_name[f_name.find("fused_") + 6 :]] = tvm_sid
                # breakpoint()
                std_func_list.append(f_name[f_name.find("fused_") + 6 :])
    return sids


def define_memory_anchors(sids, match_output_path = './match_output'):
    tvm_main_lineno = -1
    tvm_end_sid_decl_lineno = -1

    with open(match_output_path + "/codegen/host/src/default_lib1.c", "r") as def_lib_one:
        for tvm_lineno, tvm_line in enumerate(def_lib_one):
            if tvm_main_lineno == -1:
                if "tvmgen_default___tvm_main__" in tvm_line and tvm_line[-2] == "{":
                    tvm_main_lineno = tvm_lineno
            else:
                if tvm_end_sid_decl_lineno == -1:
                    tvm_sid_decl = re.search(sid_decl_re, tvm_line)
                    if tvm_sid_decl is None:
                        tvm_end_sid_decl_lineno = tvm_lineno - 1
                        sids = manage_tvmfunc_calls(
                            sids,
                            tvm_lineno=tvm_lineno,
                            tvm_line=tvm_line,
                            tvm_end_sid_decl_lineno=tvm_end_sid_decl_lineno,
                        )
                    else:
                        sids[tvm_sid_decl.group(1)] = {
                            "first": -1,
                            "last": -1,
                            "size": 1,
                            "idx": tvm_sid_decl.group(1),
                            "workspace_offset": int(tvm_sid_decl.group(2)),
                            "output": "",
                        }
                else:
                    sids = manage_tvmfunc_calls(
                        sids,
                        tvm_lineno=tvm_lineno,
                        tvm_line=tvm_line,
                        tvm_end_sid_decl_lineno=tvm_end_sid_decl_lineno,
                    )

    workspace_size = 0
    with open(match_output_path + "/codegen/host/include/tvmgen_default.h", "r") as def_inc_lib:
        for tvm_line in def_inc_lib:
            if "#define TVMGEN_DEFAULT_WORKSPACE_SIZE" in tvm_line:
                workspace_size = int(tvm_line[:-1].split(" ")[2])
                break
    return sids


def get_num_std_calls(std_call_name):
    cnt = 0
    list_ = []
    reg = f"{std_call_name}(_([0-9]+))?"
    for f_call in std_func_list:
        re_match = re.search(reg, f_call)
        if re_match is not None and f_call.endswith(re_match.group()) :
            cnt += 1
            list_.append(f_call)
    return cnt, list_


def check_call(call_node, sids):
    f_name = ""
    if hasattr(call_node.op, "name"):
        f_name = call_node.op.name
    else:
        f_name = call_node.op.name_hint
    if "tvmgen" not in f_name:
        idx_std_calls=0 if f_name not in std_calls_count else std_calls_count[f_name]
        max_calls, list_ = get_num_std_calls(f_name.replace(".", "_"))
        if len(list_)==0:
            return sids
        if (idx_std_calls+1)>len(list_):
            return sids
        tvm_f_name = list_[-(idx_std_calls+1)]
        idx_sub = tvm_f_name.index(f_name.replace(".","_"))
        if idx_sub!=0:
            operators_before_ = tvm_f_name[:idx_sub-1].split("_")
            call_node_arg=call_node
            to_skip=0
            for op_ in operators_before_[::-1]:
                if to_skip>0:
                    to_skip-=1
                    continue
                if not isinstance(call_node_arg.args[0],tvm.relay.Call):
                    return sids
                if not isinstance(call_node_arg.args[0].op,tvm.ir.op.Op):
                    return sids
                arg_name = call_node_arg.args[0].op.name
                if not arg_name.endswith(op_):
                    return sids
                to_skip = len(arg_name.replace(".","_").split("_"))-1
                arg_name = call_node_arg.args[0]
        # print(tvm_f_name,f_name)
        tvm_sid = None
        for tvm_std_f_name, tvm_std_sid in tvm_stdfunc_out_to_sid.items():
            if tvm_std_f_name.endswith(tvm_f_name):
                tvm_sid = tvm_std_sid
                break
        if tvm_sid is None:
            return sids
        #breakpoint()
        if f_name in std_calls_count:
            std_calls_count[f_name] += 1
        else:
            std_calls_count[f_name] = 1
        # breakpoint()
        sids[tvm_sid]["size"] = int(np.prod(call_node.checked_type.shape)) * int(
            np.dtype(call_node.checked_type.dtype).itemsize
        )
    else:
        if f_name not in tvm_func_out_to_sid:
            return sids
        tvm_sid = tvm_func_out_to_sid[f_name]
        sids[tvm_sid]["size"] = int(np.prod(call_node.checked_type.shape)) * int(
            np.dtype(call_node.checked_type.dtype).itemsize
        )
    return sids


def annotate_memory_size(s_node, sids):
    if not isinstance(s_node, tvm.relay.Call):
        return sids
    sids = check_call(s_node, sids)
    for arg in s_node.args:
        sids = annotate_memory_size(arg, sids)
    return sids


class MatchUART:
    def __init__(self, address, match_output):
        self.address = address
        self.uart = None
        self.baud_rate = 115200
        self.stop_bits = 2
        self.byte_size = 8
        self.rts_cts = False
        self.output_size = int(match_output["size"])
        self.output_shape = match_output["shape"]
        self.output_prec = int(match_output["prec"])
        self.match_output = match_output
        self.sys_byte_order = sys.byteorder
        self.sys_byte_order_np = "<" if self.sys_byte_order=="little" else ">"

    def open_connection(self):
        self.uart = serial.Serial(
            self.address,
            baudrate=self.baud_rate,
            bytesize=self.byte_size,
            stopbits=self.stop_bits,
        )
        print("Connected to UART: ", self.uart)

    def wait_board(self):
        read_val = self.uart.read(4)
        read_val = int.from_bytes(read_val, byteorder=self.sys_byte_order)
        return read_val

    def infer(self, inputs):
        if self.uart is None:
            print("UART is not connected!")
            return -1, 0
        uart_status = 0
        print("Sending valid status...")
        self.uart.write(uart_status.to_bytes(4, byteorder="little"))
        for idx, input_ in enumerate(inputs):
            print(f"Sending input #{idx}...")
            self.uart.write(np.ascontiguousarray(input_, dtype=np.uint8).tobytes())
        print("Receiving inference status...")
        read_value = self.uart.read(4)
        print(read_value)
        inference_status = int.from_bytes(read_value, byteorder=self.sys_byte_order)
        if inference_status != 0:
            print(f"Error during inference, status is {inference_status}...")
            return -1, inference_status
        print("Receving output...")
        output_values = self.uart.read((self.output_size * self.output_prec) + 4)
        expected_data_type = np.dtype(self.match_output["type"])
        expected_byte_order = expected_data_type.newbyteorder(self.sys_byte_order_np)

        # CIOFLANC: An extra byte gets appended when you re-run the cells without resetting the board.
        # For the summer school, enforce resetting the board when rerunning Step 5 of Hands-on 4.
        retyped_values = np.frombuffer(output_values[:self.output_size*self.output_prec], dtype = expected_byte_order) # Works only in iteration 1
        # retyped_values = np.frombuffer(output_values[1:][:self.output_size*self.output_prec], dtype = expected_byte_order) # Works from iteration 2 onwards
        
        reshaped_values = retyped_values.reshape(self.output_shape)

        # CIOFLANC: Leave prints for later debugging.
        # print (output_values)
        # print (output_values[1:])
        # print (output_values[1:][:self.output_size*self.output_prec])
        # print (expected_data_type)
        # print (expected_byte_order)
        # print (retyped_values)
        print (reshaped_values)

        return (
            0,
            reshaped_values,
        )

    def close_connection(self):
        if self.uart is None:
            print("UART is not connected!")
            return
        uart_status = 1
        self.uart.write(uart_status.to_bytes(4, byteorder="little"))
        self.uart.close()
        print("Closed connection!")


if __name__ == "__main__":
    import match

    target = match.target.Gap9()
    target.disabled_exec_modules = []
    target.disable_exec_module("NE16")
    res = match.match(
        filename="./checkpoints/hands_on_3/GraphModule.onnx",
        target=target,
        output_path="./match_output",
    )
    sids = dict()
    sids = define_memory_anchors(sids)
    # breakpoint()
    sids = annotate_memory_size(res.mod["main"].body, sids)
    # breakpoint()
    print(sids)
