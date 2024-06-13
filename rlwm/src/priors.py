import numpy as np
from scipy.stats import halfnorm
from helpers import truncnorm_better
from numba import njit

RNG = np.random.default_rng()

@njit
def sample_theta_0():
    alpha = np.random.beta(a=1.5, b=2)
    p = np.random.beta(a=1.5, b=1.5)
    return np.array([alpha, p]).astype(np.float32)

def sample_eta():
    return halfnorm.rvs(loc=0, scale=0.02, size=2)

def sample_kappa():
    phi = RNG.uniform(low=0, high=1)
    c = truncnorm_better(loc=7, scale=1, low=0, high=6)
    return np.concatenate([[phi], c])

@njit
def sample_random_walk(eta, num_steps, lower_bounds=0, upper_bounds=1):
    theta_t = np.zeros((num_steps, 2))
    theta_t[0] = sample_theta_0()
    z = np.random.randn(num_steps - 1, 2)
    for t in range(1, num_steps):
        theta_t[t] = np.maximum(
            np.minimum(theta_t[t - 1] + eta * z[t - 1], upper_bounds),
            lower_bounds
        )
    return theta_t.astype(np.float32)