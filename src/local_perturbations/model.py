import numpy as np

from local_perturbations.streaming_hmm.fit import Streaming_HMM_Model, M_step_g, M_step_q
from local_perturbations.streaming_hmm.util import diff
from local_perturbations.hyperparams import LocalPerturbationsHyperParams


class LocalPerturbationsModel(Streaming_HMM_Model):
    """ 
    Stores the parameters for a "locally perturbed" HMM, where some state(s) are fixed to a pretrained
    and presumably more expressive model, such as an RNN. Also includes an M-step method for optimizing
    its parameters given the cappe parameters delivered by the E-step.

    Attributes (not in base class):
            Kstars: A list of ints
                The entries are the indices of the emissions distributions, gs, which
                are updated on each EM iteration by the score step of the more
                expressive model.   These distributions are considered pre-trained
                and NOT updated in the M-step (although they still must be handled
                in the E-step to do the smoothing.)
            Kstds:  A list of ints
                The entries are the indices of latent states that are NOT Kstars.  So
                basically the set diff, but represented as a list.
                    Kstars U Kstds = range(K)
    Note:
        States for the pretrained models should be initialized to None.

    """

    def __init__(self, pi, Q, gs, Kstars=None):
        """
        Initialization is same as for parent class, we just have to add in the Kstars
        as an attribute.
        """
        super(LocalPerturbationsModel, self).__init__(pi, Q, gs)

        ###now set up the Kstars and Kstds
        if len(Kstars) > self.K:
            raise ValueError(
                "The length of emissions distributions provided"
                "in the score step must be less than or equal to the number"
                "of latent states, K=%d" % (self.K)
            )

            # if No Kstars provided, assume that only the 0th index is relevant.
        if Kstars is None:
            self.Kstars = [0]
        else:
            self.Kstars = Kstars

        self.Kstds = diff(range(self.K), self.Kstars)

        for K_star in self.Kstars:
            if gs[K_star] is not None:
                return ValueError(
                    "States for the pretrained models should be initialized to None."
                )

                # overwrite this from base class

    @property
    def W(self):
        return len(self.gs[self.Kstds[0]])

    @property
    def hyperparams(self):
        return LocalPerturbationsHyperParams.from_model(self)

    @property
    def n_Kstars(self):
        return len(self.Kstars)

        # alternate constructor
        # TD: somewhat violating D.R.Y. for the random() method in the parent class.

    @classmethod
    def random(cls, K, W, Kstars, seed=12345):
        """
        An alternate constructor, whereby the initial HMM parameters are set randomly,
        rather than provided by the user.
        """
        np.random.seed(seed)
        pi = np.random.dirichlet((tuple(10.0 / K for i in range(K))))
        Q = np.array(
            [np.random.dirichlet((tuple(10.0 / K for i in range(K)))) for i in range(K)]
        )

        # initialize emissions distributions
        Kstds = diff(range(K), Kstars)
        gs = [None] * K
        for i in Kstds:
            gs[i] = np.random.dirichlet((tuple(10.0 / W for i in range(W))))

        return cls(pi, Q, gs, Kstars)

    def score_step(self, gstars):
        """ 
        An external score step (e.g. from RNN) provides new emissions distributions
        at each timestep for a subset of the emissions distributions.

        Arguments:
            gstars: A list of length len(Kstars) where each element is a Categorical distribution over W elements
        """
        if len(gstars) != len(self.Kstars):
            raise ValueError(
                "The length of the emissions distributions provided a the score step"
                "must equal the length of the Kstars provided (or inferred) during class construction"
            )
            # do the updates
        for (i, Kstar) in enumerate(self.Kstars):
            self.gs[Kstar] = gstars[i]

    def M_step(self, cappe_params):
        self.Q = M_step_q(cappe_params.rho_hat_q, cappe_params.phi_hat, cappe_params.K)
        # only update the emissions distributions which are not tagged to the score step.
        updated_emission_dists = M_step_g(
            cappe_params.rho_hat_g, cappe_params.phi_hat, cappe_params.K, cappe_params.W
        )
        for i in self.Kstds:
            self.gs[i] = updated_emission_dists[i]
