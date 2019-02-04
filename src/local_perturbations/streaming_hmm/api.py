import numpy as np
import random

from local_perturbations.streaming_hmm.fit import Streaming_HMM_Model, Cappe_Params
from local_perturbations.streaming_hmm.score import (
    _get_initial_predictive_probability,
    _get_predictive_probability,
)
from local_perturbations.streaming_hmm.util import print_with_carriage_return


def train_batch(Y, K, W, t_min=20):
    """
    Given a sample (a sequence of elements drawn from a categorical distribution),
    we fit the streaming hmm algorithm from Oliver Cappe.

    Note that this function handles our streaming algo in a batch way; for true streaming
    use, one would feed the y's and gstar's in as they come along.
    """
    streaming_hmm_model = Streaming_HMM_Model.random(K, W)
    cappe_params = Cappe_Params.zeros(K, W)

    T = len(Y)
    for (t, y) in enumerate(Y):  # to do -- check that i'm getting the last obs.

        print_with_carriage_return("Now processing obs %d of %d. \r" % (t, T))

        ### E-step.
        cappe_params.E_step(streaming_hmm_model, y, t)
        ### M-step
        if t >= t_min:
            streaming_hmm_model.M_step(cappe_params)
            streaming_hmm_model.integrity_check()
    return streaming_hmm_model, cappe_params


def score_batch(Y, streaming_hmm_model, override_pi=True):
    """
    Given a sample (a sequence of elements drawn from a categorical distribution)
    and a trained HMM, 	we predict via the streaming hmm algorithm from Oliver Cappe.

    Note that this function handles our streaming algo in a batch way; for true streaming
    use, one would feed the y's in as they come along.

    Arguments:
        override_pi: Bool
            If true, we don't trust the pi in the streaming_hmm_model (e.g. perhaps the HMM was
            trained on a single sequence), so we force all the init state probs to be 1/K.
    """

    T = len(Y)

    # iniitalize scores
    scores = np.zeros(len(Y))

    # initialize E-step
    cappe_params = Cappe_Params.zeros(streaming_hmm_model.K, streaming_hmm_model.W)

    # override pi if desired
    # why? o/w that 1st few observations will be very poorly scored if we hadn't trained the hmm with the
    # the "many sequences" approach
    # TD: consider moving up to a caller function?
    if override_pi:
        for (i, p) in enumerate(streaming_hmm_model.pi):
            streaming_hmm_model.pi[i] = 1.0 / streaming_hmm_model.K

    for (t, y) in enumerate(Y):  # to do -- check that i'm getting the last obs.

        print_with_carriage_return("Now processing obs %d of %d. \r" % (t, T))

        ### prediction
        if t == 0:
            scores[t] = _get_initial_predictive_probability(y, streaming_hmm_model)
        else:
            scores[t] = _get_predictive_probability(
                y, cappe_params.phi_hat, streaming_hmm_model
            )
            ### E-step.
        cappe_params.E_step(streaming_hmm_model, y, t)

    return scores
