import numpy as np

from abc import ABCMeta, abstractmethod

from local_perturbations.black_box.prob_vectors import (
    get_random_probability_vector,
    get_perturbed_probability_vector,
)
from local_perturbations.black_box.rnn.rnn_scorer import RNN_Scorer


class BlackBoxScorer:
    """
    Abstract base class.

    Get the score conditional emissions probability from the pre-trained model.

    Note that we set the numpy seed at the beginning of generation so that we can
    regenerate the same random results, if necessary.

    Attributes:
        time_step:  Int.
            The number of time steps that have already been seen.
        W: Int
            Number of words in the vocabulary for the discrete emissions distributions.
        n_Kstars: Int
            Number of latent states pinned to pretrained cloud emissions

    """

    __metaclass__ = ABCMeta

    def __init__(self, W, n_Kstars=1):
        self.time_step = -1
        self.W = W
        self.n_Kstars = n_Kstars

    @abstractmethod
    def score(self, y):
        """
        Produce the score P(y_t+1 | y_{1:t})
        :returns: list of list of floats.
            Outer list has length n_Kstars
            Inner list has length W,
                and is a probability vector.
        """
        raise NotImplementedError


class RandomBlackBoxScorer(BlackBoxScorer):
    def __init__(self, W, n_Kstars=1, seed=1234):
        self.rng = np.random.RandomState(seed)  # local random seed.
        self.seed = seed
        super(RandomBlackBoxScorer, self).__init__(W, n_Kstars)

    def score(self, y=None):
        """
        y: int or None
            This is the sample (sequence of tokens/words/categories) represented as ints.

        Note that this classes' generated emissons are actually independent of the observation,
            y, and the model.

        """
        self.time_step += 1
        return get_random_probability_vector(self.W, self.n_Kstars, self.rng)


class PerturbedBlackBoxScorer(RandomBlackBoxScorer):
    # TD: maybe use composiiton instead of inheritance.

    def __init__(
        self, concentration_param, W, n_Kstars=1, score_seed=1234, perturbations_seed=1234
    ):
        self.score_seed = score_seed
        super(PerturbedBlackBoxScorer, self).__init__(W, n_Kstars)
        self.rng_perturbation = np.random.RandomState(
            perturbations_seed
        )  # local random seed.
        self.concentration_param = concentration_param

    def score(self, y=None):
        self.time_step += 1
        ps = super(PerturbedBlackBoxScorer, self).score(y)
        return [
            get_perturbed_probability_vector(
                p, self.concentration_param, self.rng_perturbation
            )
            for p in ps
        ]


class RNN_BlackBoxScorer(BlackBoxScorer):
    def __init__(self, rnn_scorer):
        W = rnn_scorer.vocab_size + 1  # we add one for OOV (out of vocabulary)
        super(RNN_BlackBoxScorer, self).__init__(W, n_Kstars=1)
        self.rnn_scorer = rnn_scorer

    @classmethod
    def from_disk(cls, path):
        rnn_scorer = RNN_Scorer(path)
        return cls(rnn_scorer)

    def vectorize(self, sample):
        """
        :param sample: List of strings.
        :return: List of ints
        """
        return [self.rnn_scorer._word2idx(x) for x in sample]

    def score(self, y):
        """
        y: int
            This is the obsservation (text) represented as int

        TODO:
            It's ridiculous to convert from index to text and back, as happens here,
            but need to develop a consistent primitive -- index or text -- across this repo
            (e.g. for api inputs and testing fixtures), and need to make sure it's not possible
            for there to be inconsistencies (e.g. the conversion of text to idx by the BB model versus
            by the HMM)

        """
        text = self.rnn_scorer._idx2word(y)
        ps = self.rnn_scorer.predict_probabilities(text)
        return [list(ps)]
