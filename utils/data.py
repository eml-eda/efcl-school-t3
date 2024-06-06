import random
import numpy as np
import pandas as pd
from itertools import groupby
from scipy.signal import butter, lfilter

import torch
from torch.utils.data import Dataset


def hp_filter(data, order, cutoff_frequency, sampling_frequency):
    # compute the normalize cutoff frequency and apply butter() and lfilter() to filter your data
    normalized_cutoff = cutoff_frequency / (sampling_frequency / 2)
    b, a = butter(order, normalized_cutoff, btype="highpass")
    return lfilter(b, a, data, axis=0)


def trim(data, labels):
    segment_lengths = [sum(1 for _ in group) for _, group in groupby(labels)]
    # eliminate most of the initial portion of the signal, where the trigger is zero, but the hand might not be yet at rest.
    # Precisely, we keep only some seconds before the first gesture (equal to the length of the SECOND rest).
    # Then, also eliminate all data after the last gesture, 
    # where you might have moved your hand freely. 
    start_index = segment_lengths[0] - segment_lengths[2]
    end_index = data.shape[0] - segment_lengths[-1] + 1
    data = data[start_index:end_index, :]
    labels = labels[start_index:end_index]
    return data, labels


def normalize(data, min_values, max_values):
    # normalize the data to [0:1]
    rescaled_data = (data - min_values) / (max_values - min_values)
    return rescaled_data


def prepare_data(train_val_file, test_file, concatenate_all=False):
    
    print(f"Processing {train_val_file}")

    # read session 1 data
    df = pd.read_parquet(train_val_file)

    # split data and labels
    data = df.drop("Trigger", axis=1).values 
    labels = df.Trigger.values.astype(int)

    # apply filtering to the data 
    filtered_data = hp_filter(data, order=4, cutoff_frequency=10.0, sampling_frequency=500)
    
    # trim the filtered data and labels
    trimmed_data, trimmed_labels = trim(filtered_data, labels)

    # split training and validation sets using get_repetitions_mask
    # first call the function to isolate repetitions [0,3] (for training) and
    # [4,4] (for validation). Then, use the mask as an index in the arrays
    # to separate the two subsets
    train_mask = get_repetitions_mask(trimmed_data, trimmed_labels, 0, 3)
    val_mask = get_repetitions_mask(trimmed_data, trimmed_labels, 4, 4)
    train_data = trimmed_data[train_mask]
    train_labels = trimmed_labels[train_mask]
    val_data = trimmed_data[val_mask]
    val_labels = trimmed_labels[val_mask]

    # REPEAT THE SAME STEPS FOR THE TEST SET
    print(f"Processing {test_file}")

    # read session 2 data (test set)
    df = pd.read_parquet(test_file)

    # split test data and test labels (expected: 2 lines)
    data = df.drop("Trigger", axis=1).values 
    labels = df.Trigger.values.astype(int)

    # apply filtering to the test data (expected: 1 line)
    filtered_data = hp_filter(data, order=4, cutoff_frequency=10.0, sampling_frequency=500)

    # trim the filtered test data and labels (expected: 1 line)
    trimmed_data, trimmed_labels = trim(filtered_data, labels)

    # we don't need to mask anything since we use the whole session 2 as test set
    test_data, test_labels = trimmed_data, trimmed_labels

    # normalize all data arrays using the TRAINING SET's min and max values
    train_min, train_max = train_data.min(axis=0), train_data.max(axis=0)
    train_data = normalize(train_data, train_min, train_max)
    val_data = normalize(val_data, train_min, train_max)
    test_data = normalize(test_data, train_min, train_max)

    # create windows for all three datasets
    train_data, train_labels = windowing(train_data, train_labels)
    val_data, val_labels = windowing(val_data, val_labels)
    test_data, test_labels = windowing(test_data, test_labels)

    # note: the concatenate_all option allows you to obtain splits in which:
    # - train = train + test
    # - val = val
    # - test = train + test
    # this is needed if you want to retrain on ALL DATA before deployment (which is usually a good idea).
    # of course, in this case, the test accuracy shouldn't be considered as relevant.
    # for now, leave it at False, then you might come back and set it to True before deployment.
    if concatenate_all:
        train_data = test_data = np.vstack((train_data, test_data))
        train_labels = test_labels = np.hstack((train_labels, test_labels))
    
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels), (train_min, train_max)
    

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
