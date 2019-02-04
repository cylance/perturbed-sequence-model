import numpy as np
import copy

np.set_printoptions(suppress=True, precision=4)

from local_perturbations.generate.sample import (
    generate_sample_from_model_and_blackbox_scorer,
    main_sample_generator,
)
from local_perturbations.util import make_vocab
from local_perturbations.black_box.black_box_scorer import (
    RandomBlackBoxScorer,
    PerturbedBlackBoxScorer,
)
from local_perturbations.api import train_batch, score_batch


def main_fitter(
    sample,
    vocab,
    local_perturbations_hyperparams,
    init_params_seed,
    score_seed,
    perturbations_seed,
    perturbations_inv_strength,
    t_min=20,
):
    """
    A wrapper around train_batch which does a few things automatically:
        - converts observed strings to numeric; Y = f(sample, vocab)
        - makes a black box scorer based on the perturbation strength around a true black box scorer
            (defined by the seed)
        - fits the model.

    Arguments:
        sample:  List of N strings
            Represents observed sequential data.
        vocab: List of W strings
            Represents space of possible data.
        local_perturbations_hyperparams: Instance of class LocalPerturbationsHyperparams
        perturbations_inv_strength: Float or None
        score_seed: Int
            Sets random seed for generating score probabilities via the feedforward generator (which samples from
            a dirichlet distribution.)  There is a 1:1 correspondence between the generated sequence of probability vectors
            and the score seed.
        perturbations_seed: Int
            Sets random seed for the perturbations around the black box scorer.
        init_params_seed: Int
    Returns:
        local_perturbations_fitted: Instance of class LocalPerturbationsModel
            This is fit to the sample.

    """
    Y = [vocab.index(x) for x in sample]  # rep the observations as numeric

    if perturbations_seed is None:
        blackbox_scorer = RandomBlackBoxScorer(
            W=local_perturbations_hyperparams.W,
            n_Kstars=local_perturbations_hyperparams.n_Kstars,
            seed=score_seed,
        )
    else:
        blackbox_scorer = PerturbedBlackBoxScorer(
            perturbations_inv_strength,
            W=local_perturbations_hyperparams.W,
            n_Kstars=local_perturbations_hyperparams.n_Kstars,
            score_seed=score_seed,
            perturbations_seed=perturbations_seed,
        )

    local_perturbations_fitted, cappe_params_fitted = train_batch(
        Y,
        K=local_perturbations_hyperparams.K,
        W=local_perturbations_hyperparams.W,
        Kstars=local_perturbations_hyperparams.Kstars,
        blackbox_scorer=blackbox_scorer,
        t_min=t_min,
        init_params_seed=init_params_seed,
    )
    return local_perturbations_fitted


def fitted_model_from_true_model(
    local_perturbations_model,
    score_seed,
    perturbations_seed_for_training,
    perturbations_inv_strength,
    N_samples,
):
    """
    Takes a 'true' model, generates samples from it, and then

    :param local_perturbations_model: Instance of LocalPerturbationsModel
        The "true" model.
    :param perturbations_inv_strength: Int or None.
        If None, we sample from the score mechanism given by the seed exactly, with no perturbations.
    :param N_samples: Int.
        Number of samples to generate from true model
    :return local_perturbations_fitted: Instance of LocalPerturbationsModel
        The "fitted " model.
    :return sample_train: List of strings
        The generated sample for training the model.

    """

    hmm_seed_for_generation = 12345
    hmm_seed_for_init_params = 123456
    # want the score seed to be the same for generating sample and for fitting, just
    # add perturbations while fitting.
    score_seed_for_generation = score_seed
    score_seed_for_fitting = score_seed

    ### --- generate HMM sequences (given init, transit, emiss) ---
    # this will use a random black box scorer; if we set the seed to the same value x both here
    # and in the train function, we'll get the same random emissons probabilities at each step.
    sample_train, vocab = main_sample_generator(
        N_samples,
        local_perturbations_model,
        score_seed_for_generation,
        hmm_seed_for_generation,
        perturbations_seed=None,
    )

    ###  --- fit online HMM algo ---
    local_perturbations_fitted = main_fitter(
        sample_train,
        vocab,
        local_perturbations_model.hyperparams,
        hmm_seed_for_init_params,
        score_seed_for_fitting,
        perturbations_seed_for_training,
        perturbations_inv_strength,
        t_min=20,
    )

    return local_perturbations_fitted, sample_train


def sample_from_model(
    local_perturbations_model,
    score_seed,
    perturbations_seed,
    perturbations_inv_strength,
    N_samples,
):
    ## now regnerate data from the true model

    hmm_generation_seed_test = 8888

    blackbox_scorer = PerturbedBlackBoxScorer(
        perturbations_inv_strength,
        W=local_perturbations_model.W,
        n_Kstars=local_perturbations_model.n_Kstars,
        score_seed=score_seed,
        perturbations_seed=perturbations_seed,
    )
    vocab = make_vocab(local_perturbations_model.W)
    sample = generate_sample_from_model_and_blackbox_scorer(
        N_samples,
        vocab,
        local_perturbations_model,
        blackbox_scorer,
        hmm_generation_seed_test,
    )
    return sample


def score_sample_with_two_models(
    sample, model_1, model_2, score_seed, perturbations_seed, perturbations_inv_strength
):
    """
    Takes care of some of the prep work. Ensures that the black box scorer is the same for both.
    """

    score_seed = 1111

    blackbox_scorer_1 = PerturbedBlackBoxScorer(
        perturbations_inv_strength,
        W=model_1.W,
        n_Kstars=model_1.n_Kstars,
        score_seed=score_seed,
        perturbations_seed=perturbations_seed,
    )
    blackbox_scorer_2 = copy.deepcopy(blackbox_scorer_1)

    vocab = make_vocab(model_1.W)
    Y = [vocab.index(x) for x in sample]  # rep the observations as numeric

    scores_1, _ = score_batch(Y, model_1, blackbox_scorer_1, override_pi=True)
    scores_2, _ = score_batch(Y, model_2, blackbox_scorer_2, override_pi=True)
    return scores_1, scores_2
