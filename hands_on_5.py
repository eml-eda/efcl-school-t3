# Author: Beatrice Alessandra Motetti

import os
import argparse
import json
import queue
import socket
import threading
from typing import Tuple
import numpy as np
import PySimpleGUI as sg
from pathlib import Path
import torch
from scipy.signal import butter, lfilter, lfilter_zi

# NOTE: threads can be stopped either with a KeyboardInterrupt or by closing the display window

# re-define class names
CLASS_NAMES = ["rest",
               "open hand",
               "fist (power grip)",
               "index pointed",
               "ok (thumb up)",
               "right flexion (wrist supination)",
               "left flexion (wristpronation)",
               "horns",
               "shaka"]

# Used to control KeyboardInterrupt
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

# Get the maximum length of the label names (for display reasons)
MAX_LENGTH = max(len(name) for name in CLASS_NAMES)

# Folder with image gesture
IMAGE_FOLDER = Path("./assets/")

# Size of a single input window for the DNN
WINDOW_SIZE = 300

# Global variable used to control the threads
alive = True


# function to convert a 8-bit signed integer to a 32-bit signed integer
def toSigned32(n):
    n = n & 0xffffffff
    return (n ^ 0x80000000) - 0x80000000


class DataReader(threading.Thread):
    def __init__(self, data_queue: queue.Queue, window_stride: float):
        """Thread in charge of collecting the sEMG data and storing it in a queue.

        Parameters
        ----------
        - data_queue [`queue.Queue`]: the queue where to collect the data
        - window_stride [`float`]: the stride between the windows of data, expressed as an overlap
        percentage
        window
        """
        super().__init__()
        self.data_queue = data_queue
        self.nb_channels = 8
        self.chunk_size = int((1 - window_stride) * WINDOW_SIZE)
        self.n_chunks = WINDOW_SIZE // self.chunk_size
        # Array to store the window of data
        self.data = np.empty((self.chunk_size, self.nb_channels))

        # IP and port for the UDP connection
        self.UDP_IP = 'localhost'
        self.UDP_PORT = 4040

        # Constants and parameters for conversion
        # signal_gain = 12 settings
        self.gain_scaling = 1 / 83970500
        self.vscaleFactor = 1e6  # uV

        # setup socket to receive UDP data from the Internet
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(20)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))
        print("UDP server listening to: ", self.UDP_IP, self.UDP_PORT)

    def run(self):
        global alive

        # Used to keep track of the current index in the window
        current_index = 0

        while alive:
            try:
                # buffer size is 38 bytes, but we use only 24 (3 * n_channels)
                # and ignore the rest (e.g. the sending address and other metadata)
                data, _ = self.sock.recvfrom(38)
            except socket.timeout:
                print("Failed to receive data!, please retry the script")
                break

            ads = np.array([x for x in data], dtype=np.uint8)

            # Convert data from binary to np arrays
            channels_array = np.empty(self.nb_channels)
            for chan in range(self.nb_channels):
                # There's an offset of 1 because there is 1 header byte before first channel
                channels_array[chan] = toSigned32(
                    ads[chan * 3 + 1] * 256 ** 3 + ads[chan * 3 + 2] * 256 ** 2 +
                    ads[chan * 3 + 3] * 256)

            # Convert ADC data into volts
            channels_array = channels_array * self.gain_scaling * self.vscaleFactor / 256

            # Add the data to the batch
            self.data[current_index] = channels_array
            current_index += 1

            if current_index == self.chunk_size:
                # Put the chunk of data in the queue
                if self.data_queue.qsize() < self.n_chunks:
                    self.data_queue.put(self.data)
                else:
                    print("Data queue full, discarding data")
                # Reset the index
                current_index = 0


class DataConsumer(threading.Thread):
    def __init__(self,
                 data_queue: queue.Queue,
                 pred_queue: queue.Queue,
                 min_max_values: Tuple[np.ndarray, np.ndarray],
                 model: torch.nn.Module,
                 num_windows: int,
                 stride: float):
        """Create the data consumer thread, which is in charge of handling the
        inference of the model on the collected data.

        Parameters
        ----------
        - data_queue [`queue.Queue`]: the queue where the data is collected
        - pred_queue [`queue.Queue`]: the queue where the DNN predictions are stored
        - min_max_values [`Tuple`]: a 2-element tuple, containing two numpy arrays of
        dimension equal to the number of channels, with the minimum and maximum values
        extracted from the training data and used for the rescaling
        - model [`torch.nn.Module`]: the model to use for the inference
        - num_windows [`int`]: the number of windows to consider for the inference
        - stride [`float`]: the stride ratio between the windows
        """
        super().__init__()
        self.data_queue = data_queue
        self.pred_queue = pred_queue
        self.num_windows = num_windows
        self.stride = stride
        self.chunk_size = int((1 - self.stride) * WINDOW_SIZE)
        self.n_chunks = WINDOW_SIZE // self.chunk_size

        # generate the filter coefficients
        order = 4
        cutoff_frequency = 10.0  # in Hz
        sampling_frequency = 500.0  # in Hz
        normalized_cutoff = cutoff_frequency / (sampling_frequency / 2)
        self.b, self.a = butter(order, normalized_cutoff, btype="highpass")

        # generate the filter's initial conditions (same for all channels)
        self.zi = np.array([lfilter_zi(self.b, self.a)] * 8).T

        # min and max values for rescaling
        self.min_values, self.max_values = min_max_values

        # add the model to the attributes, and set it to eval mode
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}\n")
        self.model = model.to(self.device)
        self.model.eval()

    def preprocess_chunk(self, chunk: np.ndarray):
        filtered_chunk, new_zi = lfilter(self.b, self.a, chunk, axis=0, zi=self.zi)
        self.zi = new_zi
        scaled_chunk = (filtered_chunk - self.min_values)/(self.max_values - self.min_values)
        # Transpose the data to have the channels as the first dimension
        scaled_chunk = scaled_chunk.T
        # return as tensor adding outer dimension (batch=1) and inner dimension (dummy spatial dim)
        return torch.tensor(scaled_chunk).unsqueeze(-1).unsqueeze(0).float()

    def run(self):
        global alive

        # Initialize the input buffer
        input_buffer = torch.empty((1, 8, WINDOW_SIZE, 1))
        predictions_buffer = torch.zeros(self.num_windows, 9)

        # initial buffer filling
        for i in range(self.n_chunks):
            chunk = self.data_queue.get(timeout=20)
            input_buffer[0,:,i*self.chunk_size:(i + 1)*self.chunk_size,:] = self.preprocess_chunk(chunk)

        while alive:

            try:
                # Model inference
                predictions = self.model(input_buffer.to(self.device))

                predictions_buffer = torch.cat((predictions_buffer[1:],
                                                predictions[0].unsqueeze(0).cpu()), dim=0)
                final_prediction = torch.mean(predictions_buffer, dim=0).argmax().item()
                self.pred_queue.put(final_prediction)

                # get a new chunk and replace the oldest one
                chunk = self.data_queue.get(timeout=1)
                input_buffer = torch.cat(
                        (input_buffer[:, :, self.chunk_size:, :],
                         self.preprocess_chunk(chunk)), dim=2)

            except queue.Empty:
                pass


def init_gui():
    layout = [
        [sg.Text(text='Predicted gesture', font=('Arial Bold', 16), size=20, expand_x=True,
                 justification='center')],
        [sg.Image(IMAGE_FOLDER / 'start.png', expand_x=True, expand_y=True, key='image')]]
    # Create the window
    return sg.Window('GesturePrediction', layout, size=(715, 450), keep_on_top=True)


def main(args):

    # Load the model to be used for inference
    model = torch.jit.load(args.model_path, map_location='cpu')

    # Load the min and max values used for the rescaling of the training data
    with open(args.rescaling_path, "r") as f:
        rescaling_values = json.load(f)
        min_max_values = (np.array(rescaling_values["min"]), np.array(rescaling_values["max"]))

    # Define the queues
    data_queue = queue.Queue()
    pred_queue = queue.Queue()

    # Initialize the gui
    gui = init_gui()

    # Create and start the producer and consumer threads
    producer = DataReader(data_queue, args.window_stride)
    consumer = DataConsumer(data_queue, pred_queue, min_max_values, model, args.num_windows,
                            args.window_stride)

    producer.start()
    consumer.start()

    global alive

    # Check whether a KeyboardInterrupt is raised, in that case set the global
    # variable to make the threads end
    while alive:
        try:
            event, _ = gui.read(timeout=0)
            # If the user closes the window, we stop the thread
            if event in (sg.WINDOW_CLOSED, 'Exit'):
                alive = False
                break
            pred = pred_queue.get()
            # Print the inference result
            # The resulting string is left-justified (<) and has a width of MAX_LENGTH.
            print(f'\rPredicted gesture: {CLASS_NAMES[pred]: <{MAX_LENGTH}}')

            # Update the image in the window based on the prediction
            image = IMAGE_FOLDER / f'{pred}.png'
            gui['image'].update(str(image))
            gui.refresh()
        except KeyboardInterrupt:
            # If a KeyboardInterrupt is raised, we stop the thread
            alive = False

    producer.join()
    consumer.join()
    print("\n\nExit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model_scripted_finetuned.pt',
                        help='Path to model file.')
    parser.add_argument('--rescaling_path', type=str, default='rescaling_values.json',
                        help='Path to the json file with the values for rescaling.')
    parser.add_argument('--num_windows', type=int, default=3,
                        help='Number of windows to consider for averaging.')
    parser.add_argument('--window_stride', type=float, default=0.5,
                        help='stride between windows.')
    args = parser.parse_args()
    main(args)
