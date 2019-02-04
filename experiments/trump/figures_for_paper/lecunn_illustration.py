import numpy as np

from experiments.trump.plot_score_traces import plot_scores_with_text
from experiments.trump.get_scores import get_scores, get_text


def make_lecunn_illustration(props_model, model_path_rnn):

    pre_sample = "so many papers applying deep learning to theoretical and experimental physics fascinating OOV"
    sample = [x for x in pre_sample.split()]
    start_idx = 0
    end_idx = 12

    scores_props, filter_history = get_scores(
        sample,
        start_idx,
        end_idx,
        model_path_rnn,
        model_type="props",
        prob_model=props_model,
    )
    scores_rnn, _ = get_scores(
        sample, start_idx, end_idx, model_path_rnn, model_type="rnn", prob_model=None
    )

    llr_rnn_to_props = [np.log(y / x) for (x, y) in zip(scores_props, scores_rnn)]

    text = get_text(sample, start_idx, end_idx, model_path_rnn)
    text[-1] = "fascinating"

    ### plot likelihood ratios
    n_large_words = sum([x > 0.0 for x in llr_rnn_to_props])
    n_small_words = sum([x < 0.0 for x in llr_rnn_to_props])

    plot_scores_with_text(
        text,
        y=llr_rnn_to_props,
        ylabel="Log Likelihood Ratio (LLR)",
        filename_stub="Relative scores of baseline RNN model to Trump's PROPS model",
        n_large_words=3,
        n_small_words=0,
    )
