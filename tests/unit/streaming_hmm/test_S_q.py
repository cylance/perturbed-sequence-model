from __future__ import print_function
import numpy as np
import pytest

np.set_printoptions(suppress=True)

from local_perturbations.streaming_hmm.generate import *
from local_perturbations.streaming_hmm.fit import *
from tests.unit.streaming_hmm.fixtures import (
    flipping_sequence,
    flipping_streaming_hmm_model,
    cappe_params,
    S_q_expected,
)


def is_expected(S_q, S_q_expected, print_mode=True):
    if print_mode:
        print("S_q:")
        print(S_q)
        print("S_q_expected:")
        print(S_q_expected)
    np.testing.assert_allclose(S_q, S_q_expected, atol=0.05)


def test_S_q(S_q_expected, flipping_sequence, flipping_streaming_hmm_model, cappe_params):
    """
    Tests the computation of the expected sufficient statistics for the Q (transition matrix) parameter.

    Note: this test fixes (Q,pi,gs).  We are not maximizing anything, just testing S_q.
    Basically, the idea is that if the Q matrix stipulates a very high likelihood of switching states,
    and if K=2, then the (latent, latent) bigrams should almost always be (0,1) or (1,0), and almost never
    be (0,0) or (1,1).   This expectation should show up in our expected sufficient statistics, S_q, for estimating
    Q.
    """
    print(
        "Testing S_q -- that expected sufficient statistics for Q is as expected with fixed flipping Q and Y: "
        + str(flipping_sequence)
    )
    # run e-step over many observations, so i can get good estimate of the expected sufficient stats.
    for (t, y) in enumerate(flipping_sequence):
        cappe_params.E_step(flipping_streaming_hmm_model, y, t)
        # now check if those sufficient stats are as expected.
    S_q = get_S_q(cappe_params.rho_hat_q, cappe_params.phi_hat, cappe_params.K)
    is_expected(S_q, S_q_expected)
