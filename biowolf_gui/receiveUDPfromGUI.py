import socket
import numpy as np
#import winsound

UDP_IP = 'localhost'
UDP_PORT = 4040  # receive data
total_idx = 0

# Constants and parameters for conversion
# numbers figured out by Victor
lsb_g1 = 7000700
lsb_g2 = 14000800
lsb_g3 = 20991300
lsb_g4 = 27990100
lsb_g6 = 41994600
lsb_g8 = 55994200
lsb_g12 = 83970500
HEADER_SIZE = 7  # 7 bytes header
bt_pck_size = 38  # one sample(package) has 38 bytes
vscaleFactor = 1e6  # uV
tscaleFactor = 1  # s
sample_rate = 500.0  # Sampling rate: 500Hz
signal_gain = 12.0  # Gain: 12
nb_channels = 8
nb_samples = 1950 + 2500  # one trial has 1950 samples + 5s*500Hz samples before stimulus starts

FLAG_SHORT = False  # If trial is shorter than 1950 samples, set the flag to ignore the trial
# scaling adc data into volts
scaling_dict = {
    0: 1,
    1: 1 / lsb_g1,
    2: 1 / lsb_g2,
    3: 1 / lsb_g3,
    4: 1 / lsb_g4,
    6: 1 / lsb_g6,
    8: 1 / lsb_g8,
    12: 1 / lsb_g12
}
gain_scaling = scaling_dict[signal_gain]

# package counter offsets
msb_offset = 4
msbh_offset = 3
lsbh_offset = 2
lsb_offset = 1


# function to convert uint32 to int32
def toSigned32(n):
    n = n & 0xffffffff
    return (n ^ 0x80000000) - 0x80000000


# setup socket to receive data
sock = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP

sock.settimeout(10)
sock.bind((UDP_IP, UDP_PORT))
print("UDP server listening to: ", UDP_IP, UDP_PORT)

channel_counter = 0

while True:
    try:
        data, addr = sock.recvfrom(38)  # buffer size is 38? bytes 'uint8' needed?
    except socket.timeout:
        print("Failed to receive data!, please retry the script")
        # winsound.Beep(440, 1000)
        break

    ads = np.array([x for x in data], dtype=np.uint8)

    # Convert data from binary to np arrays after having received all samples from one trial
    channels_array = np.empty(nb_channels)  # empty array for channels after conversion

    for chan in range(nb_channels):  # for each channel
        channels_array[chan] = toSigned32(
            ads[chan * 3 + 1] * 256 ** 3 + ads[chan * 3 + 2] * 256 ** 2 + ads[
                chan * 3 + 3] * 256 ** 1)  # +1 because 1 package before first channel

    # convert adc data into volts
    channels_array = channels_array / 256 * gain_scaling * vscaleFactor

    # read triggers
    trigger_array = ads[32]

    channel_counter = channel_counter + 1

    if channel_counter == 100:
        for i in range(nb_channels):
            print("CH " + str(i) + " : "f"{channels_array[0]:.3f}")

        print("-------------------------------------")
        # print('Ch1 = %2.2f', channels_array[0])
        channel_counter = 0
