import random
import numpy as np
from itertools import groupby

import torch
from torch.utils.data import Dataset


def worker_init_fn(worker_id):
    np.random.seed(23)
    random.seed(23)


class EMGDataset(Dataset):
    def __init__(self, data, labels):
        # permute to (win_num, channels, samples)
        self.X = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1)
        # create a dummy internal dimension for our network.
        self.X = self.X.unsqueeze(dim=3)
        self.Y = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.Y.shape[0]


# return a binary mask of the same length as the provided raw data, containing
# 1s in correspondance of the desired gesture repetitions, and 0s everywhere else
def get_repetitions_mask(data, labels, first_rep, last_rep, num_reps=5, num_gestures=8):

    mask = np.zeros(data.shape[0], dtype=bool)

    # compute the length of each segment in the trigger signal
    segment_lengths = [sum(1 for _ in group) for _, group in groupby(labels)]

    for gesture in range(0, num_gestures):

        # first trigger segment for this gesture. The *2 is to account for rests.
        this_gesture_first_segment = gesture * 2 * num_reps

        # first and last segment to include in this data split. *2 is to account for rests
        first_segment = this_gesture_first_segment + 2*first_rep
        last_segment = this_gesture_first_segment + 2*(last_rep + 1)

        # convert segment indices to sample indices
        window_start = sum(segment_lengths[:first_segment])
        window_end = sum(segment_lengths[:last_segment])

        mask[window_start:window_end] = 1

    return mask


# apply windowing to the EMG signal.
def windowing(data, labels, sampling_freq=500, window_time_s=.6, relative_overlap=.7, steady_margin_s=1.5):

    # this will be useful if we try to window an empty array
    if data.shape[0] == 0:
        return data, labels

    # half-length of a window (in samples)
    half_len = int((sampling_freq * window_time_s) / 2)
    win_len = 2 * half_len

    # samples outside the window that are considered for the "steadiness" check
    margin_samples = round(sampling_freq * steady_margin_s)
    # n. of samples that overlap between consecutive windows
    overlap_samples = round(win_len * relative_overlap)
    # slide between two consecutive windows
    slide = (win_len - overlap_samples)

    # total length of the data
    data_len, channels = data.shape

    # number of windows
    # the first term is the number of full windows, the second adds one window if there are remaining samples
    num_win = (data_len - win_len) // slide + 1 * \
        int(((data_len - win_len) % slide) != 0)

    # get the label for each window, considering the center of the window
    label_windows = labels[half_len:data_len - half_len:slide]
    data_windows = np.zeros((num_win, win_len, channels))
    is_steady_windows = np.zeros(num_win, dtype=bool)

    # for each window, check if all the samples (plus the margin) are from the same class
    for m in range(num_win):
        # get the center of the window
        c = half_len + m * slide
        # get the window
        data_windows[m, :, :] = data[c - half_len:c + half_len, :]
        # rest position is not margined, i.e. check is done only on windows samples
        if labels[c] == 0:
            is_steady_windows[m] = len(
                set(labels[c - half_len: c + half_len])) == 1
        # all other gestures are margined, i.e. check is done on windows samples and on the margin samples
        else:
            is_steady_windows[m] = len(set(
                labels[max(0, c - half_len - margin_samples):min(c + half_len + margin_samples, len(labels))])) == 1

    # return only steady windows
    return data_windows[is_steady_windows], label_windows[is_steady_windows]
