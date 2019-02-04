import numpy as np


def _get_initial_predictive_probability(y, streaming_hmm_model):
    """
    Gets the predictive probability for the initial observation in a sequence.

    Arguments:
        y: Int
            Represents index (from vocabulary) of current observation/token.
        streaming_hmm_model: Instance of Streaming_HMM_Model
    """
    obs_probs_by_state = np.array([g[y] for g in streaming_hmm_model.gs])
    return np.dot(streaming_hmm_model.pi, obs_probs_by_state)


def _get_predictive_probability(y, phi_hat, streaming_hmm_model):
    """
    Returns the probability
        P(Y_{n+1} = y | Y_{0:n})
    Does so via the computation that
        P(Y_{n+1} = y | Y_{0:n}) = P(X_n = x | Y_{0:n}) * P(Y_{n+1} = y | X_n = x)
    which equals
        sum_k P(X_n = k | Y_{0:n}) * [ sum_j P(X_{n+1} = j | X_n = k) P(Y_{n+1} = y | X_{n+1} = j)  ]
    where we note that the ingredients for the last equation, left to right, are phi_hat, Q, and gs

    Arguments:
        y: Int
            Represents index (from vocabulary) of current observation/token.
        phi_hat: np.ndarray
            The usual "filter" for the HMM.
            i.e. P(X_n = x | Y_0, ... Y_n)
        streaming_hmm_model: Instance of Streaming_HMM_Model

    """
    prob = 0.0
    # TD: surely dont need to recreate this object every time; maybe rethink how gs is represented throughout.
    obs_probs_by_state = np.array([g[y] for g in streaming_hmm_model.gs])
    # TD: rewrite this for loop as a single matrix multiplication
    for k in range(streaming_hmm_model.K):
        prob += phi_hat[k] * np.dot(streaming_hmm_model.Q[k], obs_probs_by_state)
    return prob
