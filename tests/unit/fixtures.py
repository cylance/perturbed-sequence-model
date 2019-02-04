import numpy as np
import pytest
import string

from local_perturbations.model import LocalPerturbationsModel
from local_perturbations.black_box.black_box_scorer import (
    RandomBlackBoxScorer,
    PerturbedBlackBoxScorer,
)


@pytest.fixture()
def local_perturbations_model_randomized(K=3, W=20, n_Kstars=1, seed=1234):
    np.random.seed(seed)
    Kstars = range(n_Kstars)
    gs = [None] + [np.random.dirichlet([1] * W) for x in range(K - 1)]
    pi = np.array([1.0 / K] * K)
    Q = np.random.dirichlet([1] * K, size=K)
    return LocalPerturbationsModel(pi, Q, gs, Kstars)


@pytest.fixture()
def local_perturbations_model_1_kstar_1_kstd():
    """
    This Perturbed HMM has 1 latent state assigned to a pretrained score model,
        and 1 latent states assigned for learning local perturbations.
    """
    Kstars = [0]
    gs = [None, np.array([0.05, 0.90, 0.05])]
    Q = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.50, 0.50])
    return LocalPerturbationsModel(pi=pi, Q=Q, gs=gs, Kstars=Kstars)


@pytest.fixture()
def local_perturbations_model_1_kstar_2_kstds():
    """
    This Perturbed HMM has 1 latent state assigned to a pretrained score model,
        and 2 latent states assigned for learning local perturbations.
    """
    Kstars = [0]
    gs = [None, np.array([0.05, 0.90, 0.05]), np.array([0.05, 0.05, 0.90])]
    pi = np.array([0.333, 0.333, 0.334])
    Q = np.array([[0.95, 0.04, 0.01], [0.04, 0.95, 0.01], [0.04, 0.01, 0.95]])
    return LocalPerturbationsModel(pi, Q, gs, Kstars)


@pytest.fixture()
def observation_sequences_1_kstar_2_kstds():
    Y_likely = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    Y_unlikely = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    return Y_likely, Y_unlikely


@pytest.fixture()
def probability_vectors_weak_perturbation():
    perturbation_strength = 1e10
    prob = RandomBlackBoxScorer(W=3, n_Kstars=1, seed=1234).score()[0]
    prob_perturbed = PerturbedBlackBoxScorer(
        perturbation_strength, W=3, n_Kstars=1, score_seed=1234, perturbations_seed=1234
    ).score()[0]
    return prob, prob_perturbed


@pytest.fixture()
def probability_vectors_strong_perturbation():
    perturbation_strength = 0.01
    prob = RandomBlackBoxScorer(W=3, n_Kstars=1, seed=1234).score()[0]
    prob_perturbed = PerturbedBlackBoxScorer(
        perturbation_strength, W=3, n_Kstars=1, score_seed=1234, perturbations_seed=9999
    ).score()[0]
    return prob, prob_perturbed
