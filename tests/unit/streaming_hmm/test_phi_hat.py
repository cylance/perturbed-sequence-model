from __future__ import print_function
import numpy as np
import pytest

from local_perturbations.streaming_hmm.fit import *
from tests.unit.streaming_hmm.fixtures import (
    sequence,
    cappe_params,
    agnostic_streaming_hmm_model,
)


def test_phi_hat_agnostic(sequence, cappe_params, agnostic_streaming_hmm_model):
    """
    Test the computation of phi_hat in the Estep,
    where phi_hat is the "filter" (i.e. P(X_t=k | Y_{0:t}))

    Given parameters pi (init state distn) and Q (latent transiiton matrix)
    which don't favor any latent states over any others,
    at each time step, phi_hat should equal likelihoods.
    """
    print(
        "Testing that phi_hats track likelihoods under uniform pi, Q for Y: "
        + str(sequence)
    )
    for (t, y) in enumerate(sequence):
        cappe_params.E_step(agnostic_streaming_hmm_model, y, t)
        is_expected_agnostic(agnostic_streaming_hmm_model, cappe_params, y)


def is_expected_agnostic(agnostic_streaming_hmm_model, cappe_params, y):
    expected_dist = latent_state_posterior_given_uniform_prior(
        agnostic_streaming_hmm_model.gs, y
    )
    np.testing.assert_allclose(cappe_params.phi_hat, expected_dist, rtol=1e-04)


def latent_state_posterior_given_uniform_prior(gs, y):
    """
    Computes P(k|y) assuming P(k) is uniform on k.
    So basically looks at the likelihoods P(y|k) and normalizes across k.
    [Here k is the latent state indicator, y is the observation.]

    Arguments:
        gs: List of numpy arrays.
            The (k)th element is the k-th distribution; these should sum to 1.
        y: An observation.
    """
    prob_y_given_k = [g[y] for g in gs]
    return prob_y_given_k / np.sum(prob_y_given_k)
