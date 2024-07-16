import numpy as np
from scipy.stats import halfnorm
from numba import njit

from helpers import truncnorm_better

RNG = np.random.default_rng()
NUM_STEPS = 512

def sample_theta_0():
    alpha = RNG.beta(a=1.5, b=2)
    tau = truncnorm_better(loc=1, scale=5, low=0, high=15)
    return np.array([alpha, tau[0]])

def sample_eta():
    scales = halfnorm.rvs(loc=0, scale=(0.02, 0.5))
    switch_probabilities = RNG.uniform(low=0, high=0.04, size=2)
    return np.concatenate([scales, switch_probabilities])

def sample_theta_t(eta, num_steps=NUM_STEPS):
    lower_bounds = np.array([0, 0])
    upper_bounds = np.array([1, 15])
    theta_t = np.zeros((num_steps, 2))
    theta_t[0] = sample_theta_0()
    z = np.random.randn(num_steps - 1, 2)
    stay = 1 - RNG.binomial(1, eta[2:], size=(num_steps-1, 2))
    for t in range(1, num_steps):
        # update alpha
        if stay[t-1, 0] == 1:
            theta_t[t, 0] = np.maximum(
                np.minimum(theta_t[t-1, 0] + eta[0] * z[t-1, 0], upper_bounds[0]),
                lower_bounds[0]
            )
        else:
            theta_t[t, 0] = RNG.beta(a=1.5, b=2)
        # update tau
        if stay[t-1, 1] == 1:
            theta_t[t, 1] = np.maximum(
                np.minimum(theta_t[t-1, 1] + eta[1] * z[t-1, 1], upper_bounds[1]),
                lower_bounds[1]
            )
        else:
            theta_t[t, 1] = truncnorm_better(loc=1, scale=5, low=0, high=15)
    return theta_t.astype(np.float32)
