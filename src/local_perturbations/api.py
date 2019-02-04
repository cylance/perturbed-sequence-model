import numpy as np
import random
import string

from local_perturbations.streaming_hmm.fit import Cappe_Params
from local_perturbations.streaming_hmm.score import (
    _get_initial_predictive_probability,
    _get_predictive_probability,
)

from local_perturbations.model import LocalPerturbationsModel
from local_perturbations.util import print_with_carriage_return


def train_batch(Y, K, W, Kstars, blackbox_scorer, t_min=20, init_params_seed=1234):
    """
    Given a sample (a sequence of elements drawn from a categorical distribution),
    as well as cloud emission probabilities (GSTARS), we fit the streaming hmm
    for the 'local perturbations' model.

    Note that this function handles our streaming algo in a batch way; for true streaming
    use, one would feed the y's and gstars (time-dependent RNN emissions distribution)
    in as they come along.

    Arguments:
        Y: List of ints.
            This is the sample (sequence of tokens/words/categories) represented as ints.
        K: Int
            Number of latent states
        W: Int
            Size of vocabulary
        Kstars: List of ints.
            These are the latent states pinned to pretrained cloud emissions
        blackbox_scorer: Instance of a derived class in blackbox_scorers.py
        t_min: Int
            Number of timesteps until starting the M_step for the cappe algorithm.
        init_params_seed: Int
            Random seed for initializing the HMM parameters.

    """
    # do some checks and make sure black box scorer is available
    if len(set(Y)) > W:
        raise ValueError(
            "The number of unique categories in the provided sample, Y, is"
            "greater than the specified size of the vocabulary, W."
        )

    if not blackbox_scorer.n_Kstars == len(Kstars):
        raise ValueError(
            "The provided black box scorer must have n_Kstars equal to the the length "
            "of the Kstars provided in the signature"
        )

        # iniitalizations
    local_perturbations_model = LocalPerturbationsModel.random(
        K, W, Kstars=[0], seed=init_params_seed
    )
    cappe_params = Cappe_Params.zeros(K, W)
    T = len(Y)

    # training
    for (t, y) in enumerate(Y):  # to do -- check that I'm getting the last observation
        print_with_carriage_return("Now processing obs %d of %d. \r" % (t, T))

        ### get g_stars (i.e. predictive probabilities for the Kstar -- ie blackbox -- nodes)
        gstars = blackbox_scorer.score(y)

        ### Feedforward step:
        local_perturbations_model.score_step(gstars)
        ### E-step.
        cappe_params.E_step(local_perturbations_model, y, t)
        ### M-step
        if t >= t_min:
            local_perturbations_model.M_step(cappe_params)
    return local_perturbations_model, cappe_params


def score_batch(Y, local_perturbations_model, blackbox_scorer, override_pi=True):
    """
    Given a sample (a sequence of elements drawn from a categorical distribution)
    and a trained HMM, 	we predict via the streaming hmm algorithm from Oliver Cappe.

    Note that this function handles our streaming algo in a batch way; for true streaming
    use, one would feed the y's in as they come along.

    Arguments:
        Y: List of ints.
            This is the sample (sequence of tokens/words/categories) represented as ints.
        local_perturbations_model: Instance of class LocalPerturbationsModel
        blackbox_scorer: Instance of a derived class in blackbox_scorers.py
        override_pi: Bool
            If true, we don't trust the pi in the local_perturbations_model (e.g. perhaps the HMM was
            trained on a single sequence), so we force all the init state probs to be 1/K.
    """

    T = len(Y)

    # iniitalize scores
    scores = np.zeros(len(Y))
    filter_record = np.zeros((len(Y), local_perturbations_model.K))

    # initialize E-step
    cappe_params = Cappe_Params.zeros(
        local_perturbations_model.K, local_perturbations_model.W
    )

    # override pi if desired
    # why? o/w that 1st few observations will be very poorly scored if we hadn't trained the hmm with
    # the "many sequences" approach
    # TD: consider moving up to a caller function?
    if override_pi:
        for (i, p) in enumerate(local_perturbations_model.pi):
            local_perturbations_model.pi[i] = 1.0 / local_perturbations_model.K

    for (t, y) in enumerate(Y):  # to do -- check that i'm getting the last obs.

        print_with_carriage_return("Now processing obs %d of %d. \r" % (t, T))

        ## update score emissions
        gstars = blackbox_scorer.score(y)
        local_perturbations_model.score_step(gstars)

        ### prediction
        if t == 0:
            scores[t] = _get_initial_predictive_probability(y, local_perturbations_model)
        else:
            scores[t] = _get_predictive_probability(
                y, cappe_params.phi_hat, local_perturbations_model
            )

            ### E-step.
        cappe_params.E_step(local_perturbations_model, y, t)
        filter_record[t] = cappe_params.phi_hat

    return scores, filter_record
