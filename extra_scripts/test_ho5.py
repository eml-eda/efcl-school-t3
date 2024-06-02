# Author: Daniele Jahier Pagliari
# Send data to the Hands-on 5 script for testing it.

import argparse
from pathlib import Path
import numpy as np
import socket
import time

HEADER_SIZE = 7
BT_PCK_SIZE = 32

UDP_IP = 'localhost'
UDP_PORT = 4040

PERIOD = 1 / 500

TRIMMED_START = 5606


def stream_data(file_path: Path):

    # Read the binary file
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.uint8)

    # Find the header (shall not be sent)
    end_of_data = len(data)
    for i in range(len(data) - HEADER_SIZE):
        if (data[i] == 60 and data[i + 1] == 60 and data[i + 2] == 62 and data[i + 3] == 62 and
                data[i + 4] == 73 and data[i + 5] == 69 and data[i + 6] == 80 and
                data[i + 7] == 44):
            print(f'Header found at index {i}')
            end_of_data = i - 1
            break

    # Extract the data from each cannel as sampled by ADC
    data_len = int(len(data[:end_of_data+1]) / BT_PCK_SIZE)
    adc_data = np.reshape(data[:end_of_data+1], (data_len, BT_PCK_SIZE))

    # prepend and append zeros to replace non-data bytes
    # to obtain a 38-byte packet as expected by the receiver
    adc_data = np.concatenate((
        np.zeros((data_len, 1), dtype=np.uint8),
        adc_data,
        np.zeros((data_len, 5), dtype=np.uint8)),
        axis=1
    )
    print(f'Prepared data shape: {adc_data.shape}')

    # Open the socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # skip part that would be trimmed
    for i in range(TRIMMED_START, data_len):
        # Send the data to the hands-on 5 script
        sock.sendto(adc_data[i].tobytes(), (UDP_IP, UDP_PORT))
        time.sleep(PERIOD)
        if i % 300 == 0:
            print(f'Sent {i} samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Binary input file')
    args = parser.parse_args()
    stream_data(args.input)
