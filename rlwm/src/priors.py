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

def sample_rw_eta(rng=None):
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
    eta = rng.normal(loc=0, scale=[0.05, 3])
    return np.abs(eta)

def sample_mrw_eta(rng=None):
    """Generates random draws from a half-normal prior over the scale and
    random draws from a uniform prior over the swiching probabilty q of
    the mixture random walk.

    Parameters:
    -----------

    rng   : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.
    Returns:
    --------
    prior_draws : np.array
        The randomly drawn scale and switching probability parameters.
    """
    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    scales = halfnorm.rvs(loc=0, scale=[0.05, 3])
    switch_probabilities = rng.uniform(low=0, high=0.1, size=2)
    return np.concatenate([scales, switch_probabilities])

def sample_random_walk(eta, num_steps=240, lower_bounds=(0, 0), upper_bounds=(1, 80), rng=None):
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
        The lower bounds for the theta parameters (default is (0, 0)).
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
        if t == 80 or t == 160:
            theta_t[t, 0] = rng.uniform(theta_t[t - 1, 0], 1)
            theta_t[t, 1] = np.clip(
            theta_t[t - 1, 1] + eta[1] * z[t - 1, 1], lower_bounds[1], upper_bounds[1]
            )
        else:
            theta_t[t] = np.clip(
                theta_t[t - 1] + eta * z[t - 1], lower_bounds, upper_bounds
            )
    return theta_t.astype(np.float32)

def sample_mixture_random_walk(eta, num_steps=240, lower_bounds=(0, 0), upper_bounds=(1, 80), rng=None):
    """Generates a single simulation from a mixture random walk transition model.

    Parameters:
    -----------
    hyper_params : np.array
        The scales and switching probabilities of the mixture random walk transition.
    num_steps    : int, optional, default: 240
        The number of time steps to take for the random walk. Default corresponds
        to the maximal number of trials in the color discrimination data set.
    lower_bounds : tuple, optional, default: ``configuration.default_bounds["lower"]``
        The minimum values the parameters can take.
    upper_bound  : tuple, optional, default: ``configuration.default_bounds["upper"]``
        The maximum values the parameters can take.
    rng          : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, num_params)
        The array of time-varying parameters
    """
    if rng is None:
        rng = np.random.default_rng()
    theta_t = np.zeros((num_steps, 2))
    theta_t[0] = sample_theta_0(rng=rng)
    z = rng.normal(size=(num_steps - 1, 2))
    stay = 1 - rng.binomial(1, eta[2:], size=(num_steps-1, 2))
    # transition model
    for t in range(1, num_steps):
        # update alpha
        if stay[t - 1, 0] == 1:
            theta_t[t, 0] = np.clip(
                theta_t[t - 1, 0] + eta[0] * z[t - 1, 0],
                a_min=lower_bounds[0], a_max=upper_bounds[0]
            )
        else:
            theta_t[t, 0] = rng.uniform(lower_bounds[0], upper_bounds[0])
        # update tau
        if stay[t - 1, 1]:
            theta_t[t, 1] = np.clip(
                theta_t[t - 1, 1] + eta[1] * z[t - 1, 1],
                a_min=lower_bounds[1], a_max=upper_bounds[1]
            )
        else:
            theta_t[t, 1] = rng.uniform(lower_bounds[1], upper_bounds[1])
    return theta_t
