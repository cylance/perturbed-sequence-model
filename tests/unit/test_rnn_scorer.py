import os

from local_perturbations.black_box.rnn.rnn_scorer import RNN_Scorer


TEST_DIR = os.path.dirname(os.path.dirname(__file__))


def test_rnn_scorer_statefulness():
    """
    Predictive probabilities of the next word should
    be state-dependent, rather than starting from scratch
    """
    model_path = os.path.join(TEST_DIR, "data/test_model/")
    prediction_model = RNN_Scorer(model_path)

    pr1 = prediction_model.predict_probabilities("he is")
    pr2 = prediction_model.predict_probabilities("he is")
    # the two predictive probabilities should differ
    assert all(pr1 == pr2) == False
