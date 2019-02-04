import numpy as np
import random

#### FUNCTIONS FOR GENERATION
def simplex_center(N):
    # N is number of vertices on simplex, so simplex has dimensionality N-1
    return np.ones(N) / N


def uniform_transition(N):
    return np.ones((N, N)) / N


def generator(N, vocab, streaming_hmm_model, seed=123):
    """

    Generates a sample (sequence of length N, drawn from the vocab)
    from an HMM, whose parameters (emissions, initial state probs, transiiton probs)
    are given by streaming_hmm_model

    Parameters:
        N: Int
            Number of samples to output
        vocab: List of strings
            Each element is a word in the vocabulary.
        streaming_hmm_model: Instance of class Streaming_HMM_Model
            Contains the parameters for the HMM
        seed: Int.
            Sets random seed for selecting both the next latent state,
            as well as the output token.
    Returns:
        sample:  List of strings
            The list has length N.
    """

    # set seed
    np.random.seed(seed)
    random.seed(seed)

    sample = [vocab[0]] * N

    # init
    k = random.randint(0, streaming_hmm_model.K - 1)
    emiss_dist = streaming_hmm_model.gs[k]
    sample[0] = np.random.choice(vocab, p=emiss_dist)

    for idx in range(1, N):
        k = np.random.choice(
            range(0, streaming_hmm_model.K), p=streaming_hmm_model.Q[k, :]
        )
        emiss_dist = streaming_hmm_model.gs[k]
        sample[idx] = np.random.choice(vocab, p=emiss_dist)
    return sample
