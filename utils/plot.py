import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from itertools import groupby
from torcheval.metrics import MulticlassConfusionMatrix


CLASS_NAMES = ["rest",
               "open hand",
               "fist (power grip)",
               "index pointed",
               "ok (thumb up)",
               "right flexion (wrist supination)",
               "left flexion (wristpronation)",
               "horns",
               "shaka"]


# Plot two signal windows (e.g. corresponding to rest and a gesture), side by side
def plot_signal(rest_id, gesture_id, data, labels):

    fig, axes = plt.subplots(data.shape[2], 2, figsize=(8,  6), sharex=True)

    for i in range(data.shape[2]):
        # plot rest channel
        axes[i][0].plot(data[rest_id, :, i], color='black')
        axes[i][0].set_ylabel(f"Ch{i+1}")
        axes[i][0].grid(True, which='both', axis='x', linestyle='--')

        # plot signal channel
        axes[i][1].plot(data[gesture_id, :, i], color='black')
        axes[i][1].grid(True, which='both', axis='x', linestyle='--')

        # make y limits the same
        axes[i][0].set_ylim(axes[i][1].get_ylim())

    # set x-axis label
    axes[-1][0].set_xlabel(CLASS_NAMES[int(labels[rest_id])])
    axes[-1][1].set_xlabel(CLASS_NAMES[int(labels[gesture_id])])

    plt.tight_layout()
    plt.show()


# Plot a portion of the unprocessed input data
def plot_raw_data(df, to_sample):
    # Plot one graph per channel
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(
        8,  len(df.columns)), sharex=True)
    for i, column in enumerate(df.columns):
        axes[i].plot(df.index[:to_sample], df[column]
                     [:to_sample], color='black')
        axes[i].set_ylabel(column)
        axes[i].grid(True, which='both', axis='x', linestyle='--')
    # set common x-axis label
    axes[-1].set_xlabel('Sample')

    # Add shading in correspondence of gestures:
    # compute the length of each "Trigger" segment
    segment_lengths = [sum(1 for _ in group) for _, group in groupby(
        df['Trigger'][:to_sample].astype(int))]
    # the first gesture starts after the first segment (which is a long rest)
    gesture_start = segment_lengths[0]
    for gesture_len, rest_len in zip(segment_lengths[1:-1:2], segment_lengths[2::2]):
        for i, column in enumerate(df.columns):
            axes[i].axvspan(gesture_start, gesture_start +
                            gesture_len, facecolor='blue', alpha=0.25)
        gesture_start += gesture_len + rest_len

    # Adjust layout
    plt.tight_layout()
    plt.show()


# plot learning curves from a history dataframe generated during training
def plot_learning_curves(history):
    # fail gracefully if there is no history
    if history is None:
        print("Empty history, cannot plot")
        return

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()
    ax.plot(history['epoch'], history['loss'], color='green', label='Train')
    ax.plot(history['epoch'], history['val_loss'],
            color='orange', label='Val.')
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    plt.tight_layout()
    plt.show()


# computes the confusion matrix of model on eval_dl
# it does it by iterating over a dataloader and accumulating in a torcheval metric
def get_conf_matrix(model, eval_dl, device):
    conf = MulticlassConfusionMatrix(
        device=device, num_classes=len(CLASS_NAMES))
    model.eval()
    with torch.no_grad():
        for sample, target in eval_dl:
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            conf.update(output, target)
    return conf.compute().cpu().numpy()


# computes the confusion matrix of model on eval_dl using seaborn
def plot_conf_matrix(model, eval_dl, device):
    conf = get_conf_matrix(model, eval_dl, device)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    df_cm = pd.DataFrame(conf.astype(
        int), index=CLASS_NAMES, columns=CLASS_NAMES)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='OrRd', ax=ax)
    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.show()


# find pareto optimal models from two paired arrays, where the first metric
# shall be minimized and the second maximized
def pareto_frontier(Xs, Ys):
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))])
    p_front = [myList[0]]
    for pair in myList[1:]:
        if pair[1] >= p_front[-1][1]:
            p_front.append(pair)
    p_front = np.array(p_front)
    return p_front[:, 0], p_front[:, 1]


# Plot models in the accuracy vs size plane, and highlight the Pareto frontier
def plot_pareto(size, accuracy, names):
    pareto_sizes, pareto_accuracies = pareto_frontier(size, accuracy)
    names = ["Seed",] + names
    plt.figure(figsize=(6, 6))
    # Plot the first point as a black diamond (seed)
    plt.scatter(size[0], accuracy[0], label='Seed',
                color='black', marker='D', s=100)
    # Plot the rest of the points as orange dots
    plt.scatter(size[1:], accuracy[1:], label='Optimized Models',
                color='orange', edgecolors='black', linewidths=1.5, s=100)
    # Plot the Pareto frontier
    plt.plot(pareto_sizes, pareto_accuracies,
             label='Pareto Frontier', color='black', linestyle='--')
    # Add names to the plot
    for i in range(1, len(size)):
        plt.text(size[i], accuracy[i], f' {names[i]}',
                 verticalalignment='top', horizontalalignment='left', fontsize=12)
    plt.xlabel('N. of Parameters')
    plt.ylabel('Accuracy')
    plt.title('Model Size vs. Accuracy with Pareto Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()
