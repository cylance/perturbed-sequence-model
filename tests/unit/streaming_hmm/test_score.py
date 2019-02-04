import numpy as np
import pytest

from local_perturbations.streaming_hmm.score import _get_initial_predictive_probability
from local_perturbations.streaming_hmm.api import (
    score_batch,
)  # used for indirect test of _get_predictive_probability(); see below

from tests.unit.streaming_hmm.fixtures import fixed_streaming_hmm_model


@pytest.mark.parametrize("y,expected_score", [(0, 0.475), (1, 0.475), (2, 0.05)])
def test_initial_predictive_probability(fixed_streaming_hmm_model, y, expected_score):
    np.testing.assert_almost_equal(
        _get_initial_predictive_probability(y, fixed_streaming_hmm_model), expected_score
    )


@pytest.mark.parametrize(
    "y1,y2", [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
)
def test_predictive_probability(fixed_streaming_hmm_model, y1, y2):
    """
    we have p(y2|y1) = sum_{x1,x2} p(y2|x2) p(x2|x1) p(y1|x1) p(x1)
    so we use that to check our function _get_predictive_probability()

    note that we test _get_predictive_probability() indirectly by calling score_batch();
    this test only isolates _get_predictive_probability() only indirectly, assuming that
    the test for _initial_predictive_probability() above passes, as well as the tests for
    phi_hat (the E step) in test_phi_hat.py
    """
    streaming_hmm_model = fixed_streaming_hmm_model
    scores = score_batch([y1, y2], streaming_hmm_model, override_pi=True)
    score = scores[-1]  # grab last score; this is p(y2|y1)

    # now compute "expected" prob p(y2|y1) "by hand"
    expected_prob = 0.0
    for x1 in range(streaming_hmm_model.K):  # x_1; first latent state
        for x2 in range(streaming_hmm_model.K):  # x_2, second latent state
            expected_prob += (
                streaming_hmm_model.gs[x2][y2]
                * streaming_hmm_model.Q[x1, x2]
                * streaming_hmm_model.gs[x1][y1]
                * streaming_hmm_model.pi[x1]
            )
    expected_prob /= _get_initial_predictive_probability(y1, fixed_streaming_hmm_model)

    np.testing.assert_almost_equal(score, expected_prob, decimal=4)
