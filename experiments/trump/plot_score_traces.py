import os

import matplotlib

matplotlib.use("TkAgg")  # due to Tkinter import error
import matplotlib.pyplot as plt

plt.style.use("ggplot")


from experiments.util import (
    get_idxs_of_n_largest_vals,
    get_idxs_of_n_smallest_vals,
    get_timestamp_string,
    wrap_string,
)


def plot_scores_with_text(
    text,
    y,
    ylabel,
    title="",
    n_large_words=5,
    n_small_words=5,
    line_col="black",
    filename_stub=None,
):
    """
    :param n_large_words: Int>=0 or "auto"
        If "auto", will plot all words with positive values in blue.
        Otherwise, will plot the specified number.
    :param n_small_words: Int>=0 or "auto"
        If "auto", will plot all words with negative values in red
        Otherwise, will plot the specified number.
    Words with n lowest scores will get plotted in blue.
    Words with n highest scores will get plotted in red.
    """
    x = [i for i in range(len(text))]

    plt.close("all")
    fig, ax = plt.subplots()
    plt.xticks(x, text, rotation=90, fontsize=10)
    plt.plot(x, y, c=line_col)
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.1)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.30)

    # gets indices of largest elements and colors those red

    if n_large_words == "auto":
        n_large_words = sum([el > 0.0 for el in y])
    if n_small_words == "auto":
        n_small_words = sum([el < 0.0 for el in y])

    idxs_large = get_idxs_of_n_largest_vals(y, n_large_words)
    idxs_small = get_idxs_of_n_smallest_vals(y, n_small_words)
    if n_large_words > 0:
        for idx in idxs_large:
            ax.get_xticklabels()[idx].set_color("blue")
    if n_small_words > 0:
        for idx in idxs_small:
            ax.get_xticklabels()[idx].set_color("red")

    plt.ylabel(ylabel)
    plt.title(wrap_string(title), fontsize=12)

    if filename_stub is None:
        filename_stub = title
    save_filepath = os.path.join(
        "plots", filename_stub + "_%s.png" % get_timestamp_string()
    )
    plt.savefig(save_filepath, format="png", dpi=1000)
    # plt.show()
