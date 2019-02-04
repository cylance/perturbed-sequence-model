"""
Will try to implement Sec 5.1 of Cappe, a simplification of Algo 1. 
"""
import numpy as np
import random
import warnings
import copy

from local_perturbations.streaming_hmm.generate import simplex_center


class Streaming_HMM_Model(object):
    """

    Stores the HMM Parameters in the special case where emissions distributions
    are categorical.   Also includes an M-step method for maximizing its parameters
    given the cappe parameters delivered by the E-step.

    Attributes:
        K: Int
            Number of latent states.
        W: Int
            Size of vocabulary
        pi: np.ndarray
            Initial state distribution. Has shape (K,).
        Q: np.ndarray
            State transition matrix. Has shape (K,K).
            Q[i,j] is P(X_t = j | X_{t-1} = i)
        gs: List of np.ndarrays
            Emissions distributions.  List has length K.
            Each np.ndarray has shape (W, ).
        pi_0: np.ndarray
            Initial state distribution (at construction). Has shape (K,).
        Q_0: np.ndarray
            State transition matrix (at construction). Has shape (K,K).
        gs_0: List of np.ndarrays
            Emissions distributions (at construction).  List has length K.
            Each np.ndarray has shape (W, ).
    """

    def __init__(self, pi, Q, gs):
        self.gs = gs
        self.pi = pi
        self.Q = Q
        self.Q_0 = copy.deepcopy(Q)
        self.pi_0 = copy.deepcopy(pi)
        self.gs_0 = copy.deepcopy(gs)

    @property
    def W(self):
        return len(self.gs[0])

    @property
    def K(self):
        return len(self.gs)

        # alternate constructor

    @classmethod
    def random(cls, K, W, seed=12345):
        """
        An alternate constructor, whereby the initial HMM parameters are set randomly,
        rather than provided by the user.
        """
        np.random.seed(seed)
        pi = np.random.dirichlet((tuple(10.0 / K for i in range(K))))
        gs = [np.random.dirichlet((tuple(10.0 / W for i in range(W)))) for i in range(K)]
        Q = np.array(
            [np.random.dirichlet((tuple(10.0 / K for i in range(K)))) for i in range(K)]
        )
        return cls(pi, Q, gs)

    def M_step(self, cappe_params):
        self.Q = M_step_q(cappe_params.rho_hat_q, cappe_params.phi_hat, cappe_params.K)
        self.gs = M_step_g(
            cappe_params.rho_hat_g, cappe_params.phi_hat, cappe_params.K, cappe_params.W
        )

    def integrity_check(self):
        if (np.isnan(self.Q).any()) or (np.isnan(self.gs).any()):
            raise ValueError(
                "np.nan found inside transition matrix or emissions parameters"
            )


class Cappe_Params(object):
    def __init__(self, phi_hat, rho_hat_q, rho_hat_g, alpha=0.6):
        self.phi_hat = phi_hat
        self.rho_hat_q = rho_hat_q
        self.rho_hat_g = rho_hat_g
        self.K = len(phi_hat)
        self.W = np.shape(rho_hat_g)[
            1
        ]  # WARNING: CAREFUL WITH THIS! ASSUMES MATRIX IS KxWxK.
        self.alpha = alpha
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "The alpha parameter (for determining step sizes) is %.04f,"
                "which is not in the unit interval" % (alpha)
            )
        if not 0.5 <= alpha <= 0.8:
            warnings.warn(
                "The alpha parameter (for determining step sizes) is %.04f,"
                "which is not in the recommended interval of [.5,.8]" % (alpha)
            )

            # alternate constructor

    @classmethod
    def zeros(cls, K, W, seed=12345):
        return cls(np.zeros(K), np.zeros((K, K, K)), np.zeros((K, W, K)))

    def E_step_first_iterate(self, pi, gs, Y_0):
        for k in range(self.K):
            self.phi_hat[k] = (
                pi[k] * gs[k][Y_0]
            )  # latter is likelihood of Y_0 from the k-th distn

            # for numerical stability
        EPSILON = 1.0 / (self.K * 1e8)
        self.phi_hat += EPSILON

        self.phi_hat /= np.sum(self.phi_hat)

    def E_step_later_iterate(self, Q, gs, Y_t_plus_one, gamma_t_plus_one):
        phi_hat_new = make_new_phi_hat(self.phi_hat, Y_t_plus_one, self.K, Q, gs)
        self.rho_hat_q, self.rho_hat_g = update_rho_hat(
            self.rho_hat_q,
            self.rho_hat_g,
            self.phi_hat,
            self.K,
            Q,
            Y_t_plus_one,
            gamma_t_plus_one,
            self.W,
        )
        self.phi_hat = phi_hat_new

    def E_step(self, streaming_hmm_model, y, t):
        if t == 0:
            self.E_step_first_iterate(streaming_hmm_model.pi, streaming_hmm_model.gs, y)
        else:
            self.E_step_later_iterate(
                streaming_hmm_model.Q, streaming_hmm_model.gs, y, gamma(t, self.alpha)
            )


def gamma(t, alpha):
    """
    Arguments:
        t: Int.
            Discrete timestep.
    Get step size as a function of timestep.
    The step sizes go to 0 as t increases.
    """
    return t ** -alpha


def make_new_phi_hat(phi_hat, Y_tp1, K, Q, gs):
    """

    phi_hat is the usual "filter" for the HMM.
        i.e. P(X_n = x | Y_0, ... Y_n)
    more specifically, we have
    phi_hat(n, nu, theta) = P(X_n = x | Y_0, ... Y_n, theta)
    where nu is the initial prob dist, theta = (transition matrix, emissions distributions)

    It has K entries and lives on Delta^{K-1}
    See Cappe (2.4)
    """

    phi_hat_new = np.zeros(K)
    for k in range(K):
        phi_hat_new[k] = np.dot(Q[:, k], phi_hat) * gs[k][Y_tp1]

        # for numerical stability
    EPSILON = 1.0 / (K * 1e8)
    phi_hat_new += EPSILON

    phi_hat_new /= np.sum(phi_hat_new)
    return phi_hat_new


def update_rho_hat(rho_hat_q, rho_hat_g, phi_hat, K, Q, Y_tp1, gamma_t, W):
    """

    rho_hat is an intermediate quantity
        rho_hat_{n, nu, theta}(x) = 1/n E[ sum_{t=1}^n s(X_{t-1}, X_t, Y_t | Y_{0:n}, X_n=x)]
    where s() are the sufficient statistics
    see Cappe (2.5)

    In our case (discrete emissions HMM), it be broken down into two separable components:
            rho_hat_q{n, nu, theta}(i,j,k; theta) = 1/n E[ sum_{t=1}^n I_{X_{t-1}=i, X_t=j} | Y_{0:n}, X_n=k)]
            rho_hat_g{n, nu, theta}(i,k; theta) = 1/n E[ sum_{t=0}^n I_{X_t=i} s(Y_t)| Y_{0:n}, X_n=k)]
        where s() here is just a multinoulli vector with W entries, so we can re-express it as
            rho_hat_g{n, nu, theta}(i,w,k; theta) = 1/n E[ sum_{t=0}^n I_{X_t=i, Y_t=w}| Y_{0:n}, X_n=k)]
    rho_hat_q has KxKxK entries
    rho_hat_g has KxWxK entries
    """
    rho_hat_q = update_rho_hat_q(rho_hat_q, phi_hat, Q, gamma_t, K)
    rho_hat_g = update_rho_hat_g(rho_hat_g, Y_tp1, phi_hat, Q, gamma_t, K, W)
    return rho_hat_q, rho_hat_g


def update_rho_hat_q(rho_hat_q, phi_hat, Q, gamma_t, K):
    r_array = get_r_array_by_terminal_state(K, phi_hat, Q)
    # TD: being kind of lazy; can i convert rho_hat_q to rho_hat_q_new in place?
    rho_hat_q_new = np.zeros_like(rho_hat_q)
    for i in range(K):
        for j in range(K):
            for k in range(K):
                rho_hat_q_new[i, j, k] = gamma_t * float(j == k) * r_array[j][i] + (
                    1 - gamma_t
                ) * np.dot(rho_hat_q[i, j, :], r_array[k])
    return rho_hat_q_new


def update_rho_hat_g(rho_hat_g, Y_tp1, phi_hat, Q, gamma_t, K, W):
    r_array = get_r_array_by_terminal_state(K, phi_hat, Q)
    rho_hat_g_new = np.zeros_like(rho_hat_g)
    for i in range(K):
        for w in range(W):
            for k in range(K):
                rho_hat_g_new[i, w, k] = gamma_t * float(i == k) * float(Y_tp1 == w) + (
                    1 - gamma_t
                ) * np.dot(rho_hat_g[i, w, :], r_array[k])
    return rho_hat_g_new


def get_r_array_by_terminal_state(K, phi_hat, Q):
    """
    Returns:
        List of 1d np.arrays (with K elements)
            The i-th element of the list gives P(X_n = dot | X_{n+1}=i, Y_{0:n})
    """
    r_array = [None] * K
    for k in range(K):
        r_array[k] = get_r(k, K, phi_hat, Q)
    return r_array


def get_r(j, K, phi_hat, Q):
    """
    report the
    retrospective conditional prob
    P(X_n=i | X_{n+1}=j, Y_{0:n})
    as a vector
    P( dot |  X_{n+1}=j, Y_{0:n})
    """
    r = np.zeros(K)
    Z = np.dot(phi_hat, Q[:, j])
    for i in range(K):
        r[i] = phi_hat[i] * Q[i, j] / Z
    return r


def get_S_g(rho_hat_g, phi_hat, K, W):
    """
    Forms S_g, the ESS (expected sufficient statistics) for "emissions".

    So S_g[i,w] can be interpreted as the expected number of (i,w) (latent, observed) bigrams.
    (I believe we should have that sum_i sum_w S_g[i,w]=1.)

    """
    S_g = np.zeros((K, W))
    for i in range(K):
        for w in range(W):
            S_g[i, w] = np.dot(rho_hat_g[i, w, :], phi_hat)
    S_g = _push_S_g_away_from_zero(S_g, K, W)
    return S_g


def _push_S_g_away_from_zero(S_g, K, W, eps=0.0001):
    """
    If the ESS are exactly 0.0 (possible in frequentist/maximum likelihood approaches;
    especially after the shortish burn-in period when the M-step starts, when we might not
    have observed all tokens), then the smoothing recursion gets thrown off; we get nan's
    in make_new_phi_hat().  To see this, check out (2.6) in Cappe and consider what happens
    if the g_theta term is exactly 0.0.
    """
    S_g += eps
    S_g /= 1.0 + K * W * eps
    return S_g


def get_S_q(rho_hat_q, phi_hat, K):
    """

    Forms S_q, the ESS (expected sufficient statistics) for "transitions".

    Based on applying the law of iterated expectation to the definitions of rho_hat_q
    in Cappe 5.1 and to phi_hat (the filter) in Cappe 2.4, we are basically just
    removing the conditioning on X_n=x from the auxiliary function to get the expected
    sufficient statistics on a single draw.

    So S_q[i,j] can be interpreted as the expected number of (i,j) latent state bigrams.
    (I believe we should have that sum_i sum_j S_q[i,j]=1.)

    """
    S_q = np.zeros((K, K))
    Q = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            S_q[i, j] = np.dot(rho_hat_q[i, j, :], phi_hat)
    return S_q


def M_step_q(rho_hat_q, phi_hat, K):
    """

    Maximize the transition matrix Q.

    Q of course is the state transition matrix, where
    Q[i,j] is P(X_t = j | X_{t-1} = i)

    To do this, we must first forms S_q,
    the ESS (expected sufficient statistics) for "transitions".

    """
    S_q = get_S_q(rho_hat_q, phi_hat, K)

    # for numerical stability
    EPSILON = 1.0 / (K * 1e8)
    S_q += EPSILON

    Q = np.zeros_like(S_q)
    Z_source = np.sum(
        S_q, 1
    )  # normalizing constant for the latent state "source" (i.e. the row in the transiton matrix Q)
    for i in range(K):
        Q[i, :] = S_q[i, :] / Z_source[i]
    return Q


def M_step_g(rho_hat_g, phi_hat, K, W):
    """

    Maximize the emissions distributions, gs.

    To do this, we must first forms S_g,
    the ESS (expected sufficient statistics) for "emissions".

    """
    S_g = get_S_g(rho_hat_g, phi_hat, K, W)

    # for numerical stability
    EPSILON = 1.0 / (W * 1e8)
    S_g += EPSILON

    # convert to gs
    latent_state_marginal_probs = np.sum(
        S_g, 1
    )  # row sums ; ith entry here is 1/n E[sum_n 1(x_t = i) | Y_{0:n}]
    gs = [S_g[k, :] / latent_state_marginal_probs[k] for k in range(K)]
    return gs
