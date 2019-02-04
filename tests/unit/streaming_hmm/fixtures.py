import numpy as np
import pytest

from local_perturbations.streaming_hmm.fit import Streaming_HMM_Model, Cappe_Params
from local_perturbations.streaming_hmm.generate import generator

### ### Cappe Params
@pytest.fixture()
def cappe_params():
    return Cappe_Params.zeros(2, 3)


### ### HMM Params
@pytest.fixture()
def agnostic_streaming_hmm_model():
    """
    We'll run our tests with no preference given to latent states (either initially or via transitions)
    and check that we observe what we expect based on the emissions distributions.
    """
    gs = [
        np.array([0.1964497, 0.49332047, 0.31022984]),
        np.array([0.0887416, 0.8010233, 0.11023509]),
    ]
    pi = np.array([0.50, 0.50])
    Q = np.array([[0.50, 0.50], [0.50, 0.50]])
    return Streaming_HMM_Model(pi, Q, gs)


@pytest.fixture()
def flipping_streaming_hmm_model():
    """
    We'll run our tests where we expected to flip between latent states,
    and each latent state has a very peaked emissions distribution.
    Then we check that we observe what we expect based on the emissions distributions.
    """
    gs = np.array([np.array([0.01, 0.01, 0.98]), np.array([0.98, 0.01, 0.01])])
    pi = np.array([0.50, 0.50])
    Q = np.array([[0.01, 0.99], [0.99, 0.01]])
    return Streaming_HMM_Model(pi, Q, gs)


@pytest.fixture()
def fixed_streaming_hmm_model():
    """
    Just a particular fixed value of hmm parameters that isn't extreme in some way, as with the agnostic hmm parameters
    or with the flipping hmm parameter. Used to assess convergence of estimated hmm parameters in test_api.py
    """
    gs = [
        np.array([0.90, 0.05, 0.05]),
        np.array([0.05, 0.90, 0.05]),
    ]  # [simplex_center(len(vocab))]* K
    pi = np.array([0.50, 0.50])
    Q = np.array([[0.95, 0.05], [0.05, 0.95]])  # uniform_transition(K)
    return Streaming_HMM_Model(pi=pi, Q=Q, gs=gs)


@pytest.fixture()
def S_q_expected():
    S_q_expected = np.array([[0.0, 0.5], [0.5, 0.0]])
    return S_q_expected


@pytest.fixture()
def S_g_expected():
    S_g_expected = np.array([[0.0, 0.0, 0.5], [0.5, 0.0, 0.0]])
    return S_g_expected


### ### Sequences
constant_sequences = [[0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2]]

non_constant_sequences = [[0, 0, 0, 1, 1, 1, 2]]

sequences = constant_sequences + non_constant_sequences


@pytest.fixture(params=sequences)
def sequence(request):
    """
    Just grabs a Y from the set of Ys
    """
    Y = request.param
    return Y


@pytest.fixture(params=constant_sequences)
def constant_sequence(request):
    """
    Just grabs a Y from the set of Ys
    """
    Y = request.param
    return Y


def create_alternating_int_array(val_1, val_2, size):
    """
    Returns:
        np.array of ints
            Values alternate between val_1 and val_2

    """
    a = np.zeros((size,), dtype=int)
    a[::2] = val_1
    a[1::2] = val_2
    return a


def generate_nearly_alternating_sequence_from_streaming_hmm_model():
    N = 100
    vocab = ["A", "B", "C"]
    sample = generator(N, vocab, flipping_streaming_hmm_model(), seed=123)
    Y = [vocab.index(x) for x in sample]  # rep the observations as numeric
    return Y


flipping_sequences = [
    create_alternating_int_array(0, 2, size=100),
    generate_nearly_alternating_sequence_from_streaming_hmm_model(),
]


@pytest.fixture(params=flipping_sequences)
def flipping_sequence(request):
    """
    Just grabs a Y from the set of Ys
    """
    Y = request.param
    return Y
