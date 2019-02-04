from __future__ import print_function
import numpy as np
import random
import pytest

from local_perturbations.model import LocalPerturbationsModel
from local_perturbations.generate.sample import (
    generate_sample_from_model_and_blackbox_scorer,
)
from local_perturbations.black_box.black_box_scorer import RandomBlackBoxScorer
from local_perturbations.api import train_batch, score_batch


from tests.unit.fixtures import (
    local_perturbations_model_1_kstar_1_kstd,
    local_perturbations_model_1_kstar_2_kstds,
    observation_sequences_1_kstar_2_kstds,
)
from tests.unit.utils import is_expected

# only test the 2-dimensional hmm (i.e. one where K=2) for now b/c we don't yet have the
# distance metric needed to check larger dim hmm's
training_fixtures = [local_perturbations_model_1_kstar_1_kstd()]


@pytest.mark.parametrize("local_perturbations_model", training_fixtures)
def test_training_api(local_perturbations_model, N_samples=5000):
    """
    Test that the Cappe algorithm (online EM for HMM) produces fitted parameters
    which are sufficiently close to known fixed parameters which generated
    simulated data, in the special case of a local perturbations model.
    """
    print("\n\n ...Testing local perturbations model....")

    ### --- generate HMM sequences (given init, transit, emiss) ---
    vocab = ["ON", "OFF", "UNSURE"]
    W = len(vocab)
    score_seed = 1234
    hmm_generation_seed = 12345
    init_params_seed = 123456

    # this will use a random black box scorer; if we set the seed to the same value x both here
    # and in the train function, we'll get the same random emissons probabilities at each step.
    blackbox_scorer = RandomBlackBoxScorer(W, n_Kstars=1, seed=score_seed)
    sample = generate_sample_from_model_and_blackbox_scorer(
        N_samples, vocab, local_perturbations_model, blackbox_scorer, hmm_generation_seed
    )
    print("\n First 100 tokens in sample")
    print(sample[:100])
    print("\n")

    ###  --- fit online HMM algo ---
    Y = [vocab.index(x) for x in sample]  # rep the observations as numeric

    ### now fit model
    # note: fit is asymptotically insensitive to initialization; so varying the seed here
    # may show no difference in fit (if N_samples is sufficiently large)
    blackbox_scorer = RandomBlackBoxScorer(W, n_Kstars=1, seed=score_seed)
    hmm_params_fitted, cappe_params_fitted = train_batch(
        Y,
        K=local_perturbations_model.K,
        W=local_perturbations_model.W,
        Kstars=[range(len(local_perturbations_model.Kstars))],
        blackbox_scorer=blackbox_scorer,
        t_min=20,
        init_params_seed=init_params_seed,
    )

    ### --- check that fitted parameters are sufficiently close to the true parameters ---
    is_expected(hmm_params_fitted, local_perturbations_model, print_mode=True)


scoring_fixtures = [
    (local_perturbations_model_1_kstar_2_kstds(), observation_sequences_1_kstar_2_kstds())
]


@pytest.mark.parametrize(
    "local_perturbations_model, observation_sequences", scoring_fixtures
)
def test_scoring_api(local_perturbations_model, observation_sequences):
    """
    Test that sequences which are likely for a given model get higher scores than sequences
    which are unlikely for a given model.
    """

    ARBITRARY_LIKELIHOOD_RATIO_THRESH = 1.5

    Y_likely, Y_unlikely = observation_sequences

    for seed in [
        1,
        12,
        123,
        1234,
    ]:  # each seed gives a slightly different black box scorer
        print("With black box scorer seed %d ..." % (seed))
        blackbox_scorer = RandomBlackBoxScorer(
            W=local_perturbations_model.W,
            n_Kstars=len(local_perturbations_model.Kstars),
            seed=seed,
        )
        scores_likely, _ = score_batch(
            Y_likely, local_perturbations_model, blackbox_scorer, override_pi=True
        )
        scores_unlikely, _ = score_batch(
            Y_unlikely, local_perturbations_model, blackbox_scorer, override_pi=True
        )
        mean_score_likely, mean_score_unlikely = (
            np.mean(scores_likely),
            np.mean(scores_unlikely),
        )
        print(
            "The mean score for the likely sequence is %.04f, for the unlikely sequence is %.04f"
            % (mean_score_likely, mean_score_unlikely)
        )
        assert mean_score_likely / mean_score_unlikely > ARBITRARY_LIKELIHOOD_RATIO_THRESH
