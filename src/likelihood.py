import numpy as np
from numba import njit
from helpers import softmax, select_action

@njit
def sample_softmax_rl(theta, context):
    """
    Perform softmax action selection in a reinforcement learning context.

    This function simulates a softmax action selection process in a reinforcement learning scenario,
    where actions are selected based on probabilities calculated from values and parameters provided
    by `theta` and `context`. It iterates over a number of time steps, adjusting values according to
    the specified reinforcement learning algorithm and selecting actions probabilistically based on
    the softmax function.

    Parameters
    ----------
    theta : np.ndarray
        A 2D numpy array of shape (num_steps, 2) containing the parameters for the reinforcement
        learning algorithm at each time step. The first column represents alpha values, and the second
        column represents tau values.
    context : np.ndarray
        A 2D numpy array of shape (num_steps, num_features) containing contextual information for
        the reinforcement learning problem at each time step. The first two columns represent rewards
        and costs, and subsequent columns represent indices of available alternatives.

    Returns
    -------
    np.ndarray
        A 1D numpy array of length `num_steps`, containing the selected actions for each time step
        based on the softmax action selection process.

    Notes
    -----
    This function requires pre-defined functions `softmax` and `select_action` to be defined in the
    environment. This function is optimized with Numba's @njit decorator for faster execution.
    """
    num_steps = theta.shape[0]
    values = np.full(4, 27.5) / 60
    resp = np.zeros(num_steps)
    for t in range(num_steps):
        curr_alt = (context[t, 2:]).astype(np.int32)
        action_probs = softmax(values[curr_alt], theta[t, 1])
        resp[t] = select_action(curr_alt, action_probs)
        values[curr_alt] = values[curr_alt] + theta[t, 0] * (context[t, :2] - values[curr_alt])
    return resp
