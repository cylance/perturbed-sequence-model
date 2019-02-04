from __future__ import print_function
import numpy as np


def get_random_probability_vector(W, n_Kstars=1, rng=None):
    """
    Generate random emissions for the purpose of mimicking the output of the score
    step of an externally trained (complex) sequential model, such as an LSTM.

    Parameters:
        W: Int
            Size of the vocabulary
        n_Kstars: Int
            Number of cloud distributions.
        rng: Instance of numpy.random.RandomState or None
            See https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
            Allows us to keep local seeds in predefined scopes.
            If None, we just proceed as normal.

    Returns:
        List of np.ndarrays
            The list has length L, where L is the number of cloud distributions (provided in n_Kstars).
            Each element of the list is an np.ndarray.  The np.ndaray provides a probability distribution
            over possible output tokens.  It has shape (W, ), where W is the number
            of words in the vocabulary.    This is meant to mimic the output of the score
            step of an externally trained, complex sequential model, such as an LSTM.
    """
    if rng is None:
        rng = np.random.RandomState()
    return [rng.dirichlet((tuple(1.0 for i in range(W)))) for i in range(n_Kstars)]


def get_perturbed_probability_vector(p, s, rng=None):
    """
    Get a perturbation around simplex vector p, with the magnitude of the perturbation controlled by scalar s.
    Basically sample from a dirichlet distribution with parameter p*s.

    Parameters:
        p: np.array
            A probability vector
        s: Float
            Pseudocount reflecting strength of prior
        rng: Instance of numpy.random.RandomState or None
            See https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
            Allows us to keep local seeds in predefined scopes.
            If None, we just proceed as normal.
    Returns:
        pp: A perturbation of p.
            As s gets larger, the variance of pp around p shrinks
    """
    EPS = 1e-5

    if rng is None:
        rng = np.random.RandomState()
    try:
        return rng.dirichlet(p * s)
    except ValueError:  # TD: more specific check for ValueError: alpha <= 0
        print(p)
        print(np.sum(p))
        # TD: I don't think that this is working
        # add in small value and renormalize to correct for exact zeros
        alpha = np.array([x * s + EPS for x in p])
        alpha /= np.mean(
            alpha
        )  # add in small value and renormalize to correct for exact zeros
        return rng.dirichlet(alpha)
