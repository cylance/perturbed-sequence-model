from __future__ import print_function
import numpy as np
import pytest
import pdb

from local_perturbations.streaming_hmm.fit import *
from tests.unit.streaming_hmm.fixtures import (
    constant_sequence,
    cappe_params,
    agnostic_streaming_hmm_model,
)
from tests.unit.streaming_hmm.utils import get_argmax_of_matrix_as_tuple


def test_rho_hat(constant_sequence, cappe_params, agnostic_streaming_hmm_model):
    """
    Test the computation of phi_hat in the Estep,
    where phi_hat is the "filter" (i.e. P(X_t=k | Y_{0:t}))

    Given parameters pi (init state distn) and Q (latent transiiton matrix)
    which don't favor any latent states over any others,
    at each time step, phi_hat should equal likelihoods.
    """
    print(
        "Testing that rho_hats track likelihoods under uniform pi, Q for Y: "
        + str(constant_sequence)
    )
    for (t, y) in enumerate(constant_sequence):
        cappe_params.E_step(agnostic_streaming_hmm_model, y, t)
        if t > 0:  # rho_hat is the zero matrix on the 0th iteration
            is_expected(cappe_params, agnostic_streaming_hmm_model, y)


def is_expected(cappe_params, streaming_hmm_model, constant_y_val):
    """
    See Cappe (5.1) for def'n of rho_hat_q.
    See Cappe (5.2) for def'n of rho_hat_g.
    The idea here is to feed in a sequence Y made up of a constant_Y_val
    then look at the emissions dists in gs (a list of k elements), find the k* where constant_Y_val is highest.
    Assume we're in that (k*)-th latent state at the final observation.
    And then we check to see that:
        the largest expected (latent, observed) bigram is (X_t, Y_t)=(k*, constant_Y_val)
        the largest expected (latent, latent) bigram is (X_{t-1}, X_t)=(k*,k*)
    """
    Prob_of_Y_t_given_k = [
        streaming_hmm_model.gs[k][constant_y_val] for k in range(streaming_hmm_model.K)
    ]
    k_which_maximizes_constant_y_val = np.argmax(Prob_of_Y_t_given_k)
    k_star = k_which_maximizes_constant_y_val
    _check_rho_hat_g(cappe_params.rho_hat_g, constant_y_val, k_star)
    _check_rho_hat_q(cappe_params.rho_hat_q, k_star)


def _check_rho_hat_g(rho_hat_g, constant_y_val, k_star):
    """
    See Cappe (5.2) for def'n of rho_hat_g.
    The idea here is to feed in a sequence Y made up of a constant_Y_val
    then look at the emissions dists in gs (a list of k elements), find the k* where constant_Y_val is highest.
    Assume we're in that (k*)-th latent state at the final observation.
    And then we check to see that the largest expected (latent, observed) bigram is (X_t, Y_t)=(k*, constant_Y_val)
    """
    expected_latent_observed_bigrams_given_final_state_is_k_star = rho_hat_g[
        :, :, k_star
    ]  # dim: KxW
    assert get_argmax_of_matrix_as_tuple(
        expected_latent_observed_bigrams_given_final_state_is_k_star
    ) == (k_star, constant_y_val)


def _check_rho_hat_q(rho_hat_q, k_star):
    """
    See Cappe (5.1) for def'n of rho_hat_q.
    The idea here is to feed in a sequence Y made up of a constant_Y_val
    then look at the emissions dists in gs (a list of k elements), find the k* where constant_Y_val is highest.
    Assume we're in that (k*)-th latent state at the final observation.
    And then we check to see that the largest expected latent_state bigrams is (X_{t-1}, X_t)=(k*,k*)
    """
    expected_latent_state_bigrams_given_final_state_is_k_star = rho_hat_q[:, :, k_star]
    assert get_argmax_of_matrix_as_tuple(
        expected_latent_state_bigrams_given_final_state_is_k_star
    ) == (k_star, k_star)
