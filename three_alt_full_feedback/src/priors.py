import numpy as np
from scipy.stats import halfnorm
from numba import njit

RNG = np.random.default_rng()

@njit
def sample_theta_0():
    alpha_1 = np.random.beta(a=1.5, b=2)
    alpha_2 = np.random.beta(a=1.5, b=2)
    tau = np.random.normal(loc=1, scale=30)
    tau = np.log(1 + np.exp(tau))
    return np.array([alpha_1, alpha_2, tau])

def sample_eta():
    return halfnorm.rvs(loc=0, scale=(0.02, 0.02, 1))

@njit
def sample_random_walk(eta, num_steps=200):
    lower_bounds = np.array([0, 0, 0])
    upper_bounds = np.array([1, 1, 80])
    theta_t = np.zeros((num_steps, 3))
    theta_t[0] = sample_theta_0()
    z = np.random.randn(num_steps - 1, 3)
    for t in range(1, num_steps):
            theta_t[t] = np.maximum(
                np.minimum(theta_t[t - 1] + eta * z[t - 1], upper_bounds),
                lower_bounds
            )
    return theta_t.astype(np.float32)