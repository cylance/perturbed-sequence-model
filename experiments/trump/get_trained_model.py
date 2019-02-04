import os, copy, pickle

from local_perturbations.streaming_hmm.api import train_batch as streaming_hmm_train_batch
from local_perturbations.streaming_hmm.api import score_batch as streaming_hmm_score_batch

from local_perturbations.black_box.black_box_scorer import RNN_BlackBoxScorer
from local_perturbations.api import train_batch, score_batch

### props model
def get_trained_model(sample, model_path, rnn_model_path, model_type, K=3, n_train=2000):
    """
    If path exists, load from disk. Otherwise, train.

    :param sample: List of strings.
    :param model_path: string.
    :param rnn_model_path: string.
    :param model_type: string.
        Either "props" or "hmm"
    :param n_train: Int.
    """
    if os.path.exists(model_path):
        print("Loading model from disk at %s" % model_path)
        model = pickle.load(open(model_path, "rb"))
    else:
        print("Loading scorer")
        rnn_model_path = os.path.join("src/local_perturbations/black_box/rnn/data/model/")
        scorer = RNN_BlackBoxScorer.from_disk(rnn_model_path)

        print("Vectorizing trump's tweets")
        Y = scorer.vectorize(sample)  # vectorize trump's tweets.

        print("Training model...")
        Y_train = Y[:n_train]
        if model_type == "props":
            model, cappe_params = train_batch(
                Y_train,
                K=K,
                W=scorer.W,
                Kstars=[0],
                blackbox_scorer=scorer,
                t_min=20,
                init_params_seed=1234,
            )
        elif model_type == "hmm":
            model, cappe_params = streaming_hmm_train_batch(
                Y_train, K=K, W=scorer.W, t_min=20
            )
        print("Saving model to.... %s" % model_path)
        pickle.dump(model, open(model_path, "wb"))
    return model
