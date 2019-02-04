import os

from experiments.consistency.distance_function import get_distance_ndarray
from experiments.util import get_timestamp_string
from experiments.heatmap import make_heat_map


def make_empirical_consistency_heat_map():

    Ns = [100, 200, 400, 800, 1600, 3200, 6400]
    Ws = [10, 20, 40, 80, 160]
    Ks = [3]

    distance_ndarray = get_distance_ndarray(
        Ns=Ns, Ws=Ws, Ks=Ks, n_reps=5, perturbations_inv_strength=1e2
    )
    distance_matrix = distance_ndarray[:, :, 0]

    filepath = os.path.join("plots/consistency_%s.png" % get_timestamp_string())

    make_heat_map(
        distance_matrix,
        filepath,
        title="Empirical consistency",
        x_label="Training Size",
        y_label="Model Complexity (Vocab Size)",
        cbar_label="Distance to true model",
        x_tick_list=Ns,
        y_tick_list=Ws,
    )
