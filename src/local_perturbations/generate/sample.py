import numpy as np
import random
import string

from local_perturbations.black_box.black_box_scorer import (
    RandomBlackBoxScorer,
    PerturbedBlackBoxScorer,
)
from local_perturbations.util import make_vocab


def generate_sample_from_model_and_blackbox_scorer(
    N, vocab, local_perturbations_model, blackbox_scorer, hmm_generation_seed=12345
):
    """
    Generates a sample (sequence of length N, drawn from the vocab)

    from:
        a "perturbed" HMM
            whose parameters are given by hmm_params (provides local emissions, initial state probs, transiiton probs)
    and
        a black box scorer

    Parameters:
        N: Int
            Number of samples to output
        vocab: List of strings
            Each element is a word in the vocabulary.
        local_perturbations_model: Instance of class LocalPerturbationsModel
            Contains the parameters for the HMM
        blackbox_scorer: Instance of a derived class in blackbox_scorers.py
        hmm_generation_seed: Int.
            Sets random seed for selecting both the next latent state,
            as well as the output token.
    Returns:
        sample:  List of strings
            The list has length N.
    """

    # set seeds
    np.random.seed(hmm_generation_seed)  # for selecting the output token
    random.seed(hmm_generation_seed)  # for selecting the next latent state.
    sample = [vocab[0]] * N

    # init
    k = random.randint(0, local_perturbations_model.K - 1)
    gstars = blackbox_scorer.score()  # lists of categorical distributions of length W.
    local_perturbations_model.score_step(gstars)
    emiss_dist = local_perturbations_model.gs[k]
    sample[0] = np.random.choice(vocab, p=emiss_dist)

    # recursion
    for idx in range(1, N):
        k = np.random.choice(
            range(0, local_perturbations_model.K), p=local_perturbations_model.Q[k, :]
        )
        gstars = (
            blackbox_scorer.score()
        )  # lists of categorical distributions of length W.
        local_perturbations_model.score_step(gstars)
        emiss_dist = local_perturbations_model.gs[k]
        sample[idx] = np.random.choice(vocab, p=emiss_dist)
    return sample


def main_sample_generator(
    N_samples,
    local_perturbations_model,
    score_seed,
    hmm_generation_seed,
    perturbations_seed=None,
    perturbations_inv_strength=None,
):
    """
    Generates a sample (sequence of length N, drawn from the vocab)

    from:
        a "perturbed" HMM
            whose parameters are given by hmm_params (provides local emissions, initial state probs, transiiton probs)
    and
        a perturbations_inv_strength
            Which can be used to perturb the outputs of a black box scorer so that learning needn't
            exactly match the generative mechanism.

    Parameters:
        N_samples: Int
            Number of samples to output
        local_perturbations_model: Instance of class LocalPerturbationsModel
            Contains the parameters for the HMM
        hmm_generation_seed: Int.
            Sets random seed for selecting both the next latent state,
            as well as the output token.
        score_seed: Int
            Sets random seed for generating score probabilities via the feedfroward generator (which samples from
            a dirichlet distribution.)  There is a 1:1 correspondance between the generated sequence of probability vecotrs
            and the score seed.
        perturbation_seed: Int or None
            If int, Sets random seed for the perturbations around the black box scorer.
            If None, we don't use perturbations.
        perturbations_inv_strength: Float or None
            If not None, will control size of perturbation of the black box scorer. In particular, it
            controls the magnitude of the alpha parameter for the dirichlet distribution from which probability
            samples are generated.
 
    Returns:
        sample:  List of strings
            The list has length N.

    """
    if perturbations_seed is None:
        blackbox_scorer = RandomBlackBoxScorer(
            W=local_perturbations_model.W,
            n_Kstars=len(local_perturbations_model.Kstars),
            seed=score_seed,
        )
    else:
        blackbox_scorer = PerturbedBlackBoxScorer(
            perturbations_inv_strength,
            W=local_perturbations_model.W,
            n_Kstars=len(local_perturbations_model.Kstars),
            score_seed=score_seed,
            perturbations_seed=perturbations_seed,
        )

    vocab = make_vocab(local_perturbations_model.W)
    sample = generate_sample_from_model_and_blackbox_scorer(
        N_samples, vocab, local_perturbations_model, blackbox_scorer, hmm_generation_seed
    )
    return sample, vocab
