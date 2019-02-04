from local_perturbations.streaming_hmm.api import train_batch as streaming_hmm_train_batch
from local_perturbations.streaming_hmm.api import score_batch as streaming_hmm_score_batch

from local_perturbations.black_box.black_box_scorer import RNN_BlackBoxScorer
from local_perturbations.api import train_batch, score_batch


def get_scores(sample, start_idx, end_idx, rnn_model_path, model_type, prob_model=None):
    """
    :param model_type: String
        Must be in ["props", "hmm", "rnn"]
    :return scores: np.array
        Probability scores
    :return filter_record:
        Is not None only for the props model
    """
    ### get test data
    scorer = RNN_BlackBoxScorer.from_disk(rnn_model_path)
    Y = scorer.vectorize(sample)
    Y_test = Y[start_idx:end_idx]

    print("Scoring Data...")
    if model_type == "props":
        scores_props, filter_record = score_batch(
            Y_test, prob_model, blackbox_scorer=scorer, override_pi=True
        )
        return scores_props, filter_record
    elif model_type == "hmm":
        scores_hmm = streaming_hmm_score_batch(Y_test, prob_model, override_pi=True)
        filter_record = None
        return scores_hmm, None
    elif model_type == "rnn":
        Y_test = Y[start_idx : end_idx + 1]
        curr_words = Y[start_idx:end_idx]
        next_words = Y[start_idx + 1 : end_idx + 1]

        scores_rnn = []
        for (curr, next) in zip(curr_words, next_words):
            scores_rnn.append(scorer.score(curr)[0][next])
        return scores_rnn, None
    else:
        raise ValueError("Model type is not understood")


def _indices_to_text(idxs, scorer):
    text = []
    for idx in idxs:
        word = scorer.rnn_scorer._idx2word(idx)
        text.append(word)
    return text


def get_text(sample, start_idx, end_idx, rnn_model_path):
    scorer = RNN_BlackBoxScorer.from_disk(rnn_model_path)
    Y = scorer.vectorize(sample)
    Y_test = Y[start_idx:end_idx]
    text = _indices_to_text(Y_test, scorer)
    return text
