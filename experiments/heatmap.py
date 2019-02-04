import nunpy as np

from matplotlib import pyplot as plt


def make_heat_map(
    mat, filepath, title, x_label, y_label, cbar_label, x_tick_list, y_tick_list
):

    plt.figure()
    plt.matshow(np.transpose(mat))
    cbar = plt.colorbar()
    # cbar.set_label(cbar_label, rotation=270)
    cbar.ax.text(2.9, 0.7, cbar_label, rotation=270)
    plt.clim(np.min(mat) * 0.80, np.max(mat) * 1.2)  # set color limits
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.xticks(np.arange(mat.shape[0]), x_tick_list)
    plt.yticks(np.arange(mat.shape[1]), y_tick_list)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, fontsize=12)
    plt.savefig(filepath)
    plt.close()
