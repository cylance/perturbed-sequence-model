from __future__ import print_function
import numpy as np

from experiments.trump.plot_score_traces import plot_scores_with_text
from experiments.trump.get_scores import get_scores, get_text


def make_collocations_illustration(
    props_model,
    hmm_model,
    model_path_rnn,
    sample,
    start_idx=12180 + 166,
    end_idx=12180 + 182,
):

    print(sample[start_idx:end_idx])

    ### ---- Scoring---
    scores_props, filter_history = get_scores(
        sample,
        start_idx,
        end_idx,
        model_path_rnn,
        model_type="props",
        prob_model=props_model,
    )
    scores_hmm, _ = get_scores(
        sample, start_idx, end_idx, model_path_rnn, model_type="hmm", prob_model=hmm_model
    )

    llr_props_to_hmm = [np.log(x / y) for (x, y) in zip(scores_props, scores_hmm)]

    ### ---- Plotting ----
    ### get text
    n_large_words = sum([x > 0.0 for x in llr_props_to_hmm])
    n_small_words = sum([x < 0.0 for x in llr_props_to_hmm])

    text = get_text(sample, start_idx, end_idx, model_path_rnn)

    plot_scores_with_text(
        text,
        y=llr_props_to_hmm,
        ylabel="Log Likelihood Ratio (LLR)",
        filename_stub="Relative scores of PROPS model to standard HMM model",
        n_large_words=n_large_words,
        n_small_words=0,
    )
