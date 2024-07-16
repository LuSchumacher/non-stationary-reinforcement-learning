import numpy as np
from numba import njit

@njit
def softmax(x, tau):
    """
    Apply the softmax function to an array of values with a temperature parameter.

    Parameters
    ----------
    x : np.ndarray
        A 1D numpy array containing the input values over which the softmax function is to be applied.
    tau : float
        The temperature parameter controlling the sharpness of the softmax output.
        Must be a positive value.

    Returns
    -------
    np.ndarray
        A 1D numpy array of the same shape as `x`, where each value has been transformed by the softmax function,
        representing a probability distribution.

    Note
    ----
    This function is optimized with Numba's @njit decorator for faster execution.
    """
    e_x = np.exp(x * tau)
    out = e_x / e_x.sum()
    return out

@njit
def select_action(x, p):
    """
    Selects a random action based on a given probability distribution.

    Parameters
    ----------
    x : np.ndarray
        A 1D numpy array of actions or values to select from.
    p : np.ndarray
        A 1D numpy array of probabilities associated with each action in `x`. The sum of all probabilities should
        be 1.

    Returns
    -------
    Any
        A randomly selected action from `x`, chosen according to the probabilities specified in `p`.

    Note
    ----
    This function is optimized with Numba's @njit decorator for faster execution.
    """
    return x[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]