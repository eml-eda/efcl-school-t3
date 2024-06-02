# Author: Matteo Risso
# Convert binary file saved by the data acquisition GUI to parquet format

import argparse
from pathlib import Path
import struct
import numpy as np
import pandas as pd

GAIN_SCALING = {
    'G1': 1 / 7000700,
    'G2': 1 / 14000800,
    'G3': 1 / 20991300,
    'G4': 1 / 27990100,
    'G6': 1 / 41994600,
    'G8': 1 / 55994200,
    'G12': 1 / 83970500,
}
HEADER_SIZE = 7
BT_PCK_SIZE = 32

T_SCALE = 1
V_SCALE = 1e6


def read_data(file_path: Path) -> pd.DataFrame:
    # Start building the dataframe
    df = pd.DataFrame(columns=['Ch1', 'Ch2', 'Ch3', 'Ch4',
                               'Ch5', 'Ch6', 'Ch7', 'Ch8',
                               'Trigger'])
    timestamp = '_'.join(str(file_path.stem).split('_')[1:])
    df.attrs['Datetime'] = timestamp

    # Read the binary file
    with open(file_path, 'rb') as file:
        # data = file.read()
        data = np.fromfile(file, dtype=np.uint8)

    # Find the header
    header = ''
    end_of_data = len(data)
    for i in range(len(data) - HEADER_SIZE):
        if (data[i] == 60 and
           data[i + 1] == 60 and
           data[i + 2] == 62 and
           data[i + 3] == 62 and
           data[i + 4] == 73 and
           data[i + 5] == 69 and
           data[i + 6] == 80 and
           data[i + 7] == 44):
            header = data[i:].tobytes().decode()
            end_of_data = i - 1
            break

    # Extract information from header and start populating the dataframe
    for field in str(header).split(','):
        if 'T' in field:
            df.attrs['TestName'] = field[1:]
        elif 'S' in field:
            df.attrs['SubjectName'] = field[1:]
        elif 'A' in field:
            df.attrs['SubjectAge'] = field[1:]
        elif 'R' in field:
            df.attrs['Remarks'] = field[1:]
        elif 'F' in field:
            df.attrs['SampleRate'] = field[1:]
        elif 'G' in field:
            df.attrs['SampleGain'] = field[1:]
            gain_scaling = GAIN_SCALING[f'G{field[1:]}']

    # Extract the data from each cannel as sampled by ADC
    data_len = int(len(data[:end_of_data+1]) / BT_PCK_SIZE)
    adc_data = np.reshape(data[:end_of_data+1], (data_len, BT_PCK_SIZE))
    adc_ch1 = np.zeros(int(data_len))
    adc_ch2 = np.zeros(int(data_len))
    adc_ch3 = np.zeros(int(data_len))
    adc_ch4 = np.zeros(int(data_len))
    adc_ch5 = np.zeros(int(data_len))
    adc_ch6 = np.zeros(int(data_len))
    adc_ch7 = np.zeros(int(data_len))
    adc_ch8 = np.zeros(int(data_len))
    for i in range(data_len):
        adc_ch1[i] = struct.unpack('i', struct.pack('I',
                                                    adc_data[i, 0] * 256**3 +
                                                    adc_data[i, 1] * 256**2 +
                                                    adc_data[i, 2] * 256))[0]
        adc_ch2[i] = struct.unpack('i', struct.pack('I',
                                                    adc_data[i, 3] * 256**3 +
                                                    adc_data[i, 4] * 256**2 +
                                                    adc_data[i, 5] * 256))[0]
        adc_ch3[i] = struct.unpack('i', struct.pack('I',
                                                    adc_data[i, 6] * 256**3 +
                                                    adc_data[i, 7] * 256**2 +
                                                    adc_data[i, 8] * 256))[0]
        adc_ch4[i] = struct.unpack('i', struct.pack('I',
                                                    adc_data[i, 9] * 256**3 +
                                                    adc_data[i, 10] * 256**2 +
                                                    adc_data[i, 11] * 256))[0]
        adc_ch5[i] = struct.unpack('i', struct.pack('I',
                                                    adc_data[i, 12] * 256**3 +
                                                    adc_data[i, 13] * 256**2 +
                                                    adc_data[i, 14] * 256))[0]
        adc_ch6[i] = struct.unpack('i', struct.pack('I',
                                                    adc_data[i, 15] * 256**3 +
                                                    adc_data[i, 16] * 256**2 +
                                                    adc_data[i, 17] * 256))[0]
        adc_ch7[i] = struct.unpack('i', struct.pack('I',
                                                    adc_data[i, 18] * 256**3 +
                                                    adc_data[i, 19] * 256**2 +
                                                    adc_data[i, 20] * 256))[0]
        adc_ch8[i] = struct.unpack('i', struct.pack('I',
                                                    adc_data[i, 21] * 256**3 +
                                                    adc_data[i, 22] * 256**2 +
                                                    adc_data[i, 23] * 256))[0]

    # Convert the ADC data to physical units
    skip_samples = 1
    ch1 = adc_ch1[skip_samples:] * gain_scaling * V_SCALE / 256
    ch2 = adc_ch2[skip_samples:] * gain_scaling * V_SCALE / 256
    ch3 = adc_ch3[skip_samples:] * gain_scaling * V_SCALE / 256
    ch4 = adc_ch4[skip_samples:] * gain_scaling * V_SCALE / 256
    ch5 = adc_ch5[skip_samples:] * gain_scaling * V_SCALE / 256
    ch6 = adc_ch6[skip_samples:] * gain_scaling * V_SCALE / 256
    ch7 = adc_ch7[skip_samples:] * gain_scaling * V_SCALE / 256
    ch8 = adc_ch8[skip_samples:] * gain_scaling * V_SCALE / 256
    trigger = adc_data[skip_samples:, 31]

    # Populate the dataframe
    df['Ch1'] = ch1
    df['Ch2'] = ch2
    df['Ch3'] = ch3
    df['Ch4'] = ch4
    df['Ch5'] = ch5
    df['Ch6'] = ch6
    df['Ch7'] = ch7
    df['Ch8'] = ch8
    df['Trigger'] = trigger

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Binary input file')
    parser.add_argument('--output', type=str, required=True, help='Parquet output file')
    args = parser.parse_args()
    df = read_data(Path(args.input))

    # Save the dataframe and metadata to parquet file
    df.to_parquet(Path(args.output), compression='gzip')

    # Read the dataframe and print metadata
    df = pd.read_parquet(args.output)
    print(df.attrs)
