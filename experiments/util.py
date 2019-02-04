import os
import numpy as np
import time
import datetime
from textwrap import wrap


def cum_means(arr):
    """
    Return the cumulative means from an array.
    """
    cum_sums = np.cumsum(arr)
    cum_means = [x / (i + 1) for (i, x) in enumerate(cum_sums)]
    return cum_means


def get_idxs_of_n_largest_vals(arr, n):
    return np.argpartition(arr, -n)[-n:]


def get_idxs_of_n_smallest_vals(arr, n):
    return np.argpartition(arr, n)[:n]


def fun_display(sample):
    """
    Take some characters in a vocab and display it as gibberish language.
    """
    periods = [sample[-1], sample[-2]]
    fun_display = ""
    for x in sample:
        if x in periods:
            fun_display += ". "
        else:
            fun_display += x
    return fun_display


def get_timestamp_string():
    secs_since_epoch = time.time()
    timestamp_string = datetime.datetime.fromtimestamp(secs_since_epoch).strftime(
        "%Y-%m-%d %H-%M-%S"
    )
    return timestamp_string


def wrap_string(string, width=65):
    """Used e.g. to make titles with returns in the correct place. """
    return "\n".join(wrap(string, width))


def ensure_dir(directory):
    """
    Description:
    Makes sure directory exists before saving to it.

    Parameters:
        directory: An string naming the directory on the local machine where we will save stuff.

    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def set_matplotlib_engine():
    """
    """
    import matplotlib

    matplotlib.use("agg")
