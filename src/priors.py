import numpy as np
from scipy.stats import halfnorm

def sample_theta_0(rng=None):
    """
    Generates random draws from from prior distributions over the inital values for the theta
    parameters theta = {alpha, tau}

    Alpha is uniformly distributed between 0 and 1, and tau is drawn from a log-normal
    distribution with a mean of 1 and a standard deviation of 30, after being transformed to
    ensure positivity.

    Parameters
    ----------
    rng : np.random.Generator, optional
        An instance of numpy's random number generator. If None, a new default generator is used.
        This allows for reproducible results if a fixed seed generator is passed.

    Returns
    -------
    np.ndarray
        A numpy array of shape (2,) containing the sampled values of alpha and tau.
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha = rng.uniform(low=0, high=1)
    tau = rng.normal(loc=1, scale=30)
    tau = np.log(1 + np.exp(tau))
    return np.array([alpha, tau])

def sample_eta(rng=None):
    """
    Generates random draws from from prior distributions over the eta parameters.

    This function samples two values for the eta parameters from a half-normal distribution,
    with a location of 0 and scales of 0.2 and 3, respectively. This distribution ensures
    that the sampled values are always positive.

    Parameters
    ----------
    rng : np.random.Generator, optional
        An instance of numpy's random number generator. If None, a new default generator is used.
        This allows for reproducible results if a fixed seed generator is passed.

    Returns
    -------
    np.ndarray
        A numpy array of shape (2,) containing the sampled eta values.
    """
    if rng is None:
        rng = np.random.default_rng()
    eta = rng.normal(loc=0, scale=[0.2, 3])
    return np.abs(eta)

def sample_random_walk(eta, num_steps=80, lower_bounds=(0, 1), upper_bounds=(1, 80), rng=None):
    """
    Perform a constrained random walk to sample theta parameters over a number of steps.

    This function simulates a random walk for the theta parameters (alpha and tau), starting from
    initial values sampled by `sample_theta_0`. The walk is constrained by lower and upper bounds. The
    step size is controlled by the eta parameter and random variations.

    Parameters
    ----------
    eta : np.ndarray
        The eta parameter controlling the step size in the random walk, for each theta component.
    num_steps : int, optional
        The number of steps in the random walk (default is 80).
    lower_bounds : tuple, optional
        The lower bounds for the theta parameters (default is (0, 1)).
    upper_bounds : tuple, optional
        The upper bounds for the theta parameters (default is (1, 80)).
    rng : np.random.Generator, optional
        An instance of numpy's random number generator. If None, a new default generator is used.
        This allows for reproducible results if a fixed seed generator is passed.

    Returns
    -------
    np.ndarray
        A numpy array of shape (num_steps, 2) containing the sampled theta values over the steps.
    """
    if rng is None:
        rng = np.random.default_rng()
    theta_t = np.zeros((num_steps, 2))
    theta_t[0] = sample_theta_0(rng=rng)
    z = rng.normal(size=(num_steps - 1, 2))
    for t in range(1, num_steps):
        theta_t[t] = np.clip(
            theta_t[t - 1] + eta * z[t - 1], lower_bounds, upper_bounds
        )
    return theta_t.astype(np.float32)