from __future__ import print_function
import numpy as np
import pytest


def get_argmax_of_matrix_as_tuple(mat):
    return np.unravel_index(mat.argmax(), mat.shape)


def is_expected(hmm_params_fitted, local_perturbations_model, print_mode=True):
    if print_mode:
        print("Transition matrix (Q) fitted :")
        print(hmm_params_fitted.Q)
        print("Transition matrix (Q) true:")
        print(local_perturbations_model.Q)
        print("Emissions distributions (gs) fitted (excluding cloud distribution):")
        print(hmm_params_fitted.gs[1:])
        print("Emissions distributions (gs) true (excluding cloud distribution):")
        print(local_perturbations_model.gs[1:])
    np.testing.assert_allclose(
        hmm_params_fitted.Q, local_perturbations_model.Q, atol=0.05
    )
    np.testing.assert_allclose(
        hmm_params_fitted.gs[1:], local_perturbations_model.gs[1:], atol=0.05
    )
