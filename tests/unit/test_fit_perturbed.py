import numpy as np

from local_perturbations.model import LocalPerturbationsModel
from local_perturbations.black_box.prob_vectors import get_random_probability_vector


# TD: write analagous test for base __init__.  Show we get an error if the hypothesis
# described in the docstring below fails.
def test_construction_of_LocalPerturbationsModel_for_random_constructor():
    """
    The LocalPerturbationsModel class should have emissions distributions, gs, which are
        equal to None if the state corrsonds to a pretrained model.

    After one score pass, though, no g's should be None.
    """
    W = 5
    K = 3
    Kstars = [
        0
    ]  # TD: make this a parametrized fixture which runs through multiple possible Kstars.

    # initial construction should have emissions distributiosn which are None for all kstars, else np.array
    local_perturbations_model = LocalPerturbationsModel.random(K, W, Kstars, seed=12345)

    for k in range(K):
        if k in Kstars:
            assert local_perturbations_model.gs[k] is None
        else:
            assert isinstance(local_perturbations_model.gs[k], np.ndarray)

            # now we do a score pass
    gstars = get_random_probability_vector(W, n_Kstars=1)
    for k in Kstars:
        local_perturbations_model.score_step(gstars)

        # now all emissiond sitributions should be np.arrays
    for k in range(K):
        assert isinstance(local_perturbations_model.gs[k], np.ndarray)
