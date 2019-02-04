from __future__ import print_function
import numpy as np

from experiments.trump.plot_score_traces import plot_scores_with_text
from experiments.trump.get_scores import get_scores, get_text

### ---- Scoring---
def make_customization_illustration(
    props_model, model_path_rnn, sample, start_idx=12180 + 166, end_idx=12180 + 182
):
    print(sample[start_idx:end_idx])

    scores_props, filter_history = get_scores(
        sample,
        start_idx,
        end_idx,
        model_path_rnn,
        model_type="props",
        prob_model=props_model,
    )

    ### ---- Plotting ----
    text = get_text(sample, start_idx, end_idx, model_path_rnn)
    p_normal = [x[0] for x in filter_history]
    llr_custom = [np.log((1.0 - p) / p) for p in p_normal]
    plot_scores_with_text(
        text,
        y=llr_custom,
        ylabel="Log Likelihood Ratio (LLR)",
        filename_stub="Evidence that PROPS hidden state belongs to a "
        "local perturbation (Trump) mode vs. baseline (Wikipedia) mode",
        n_large_words=3,
        n_small_words=0,
    )
