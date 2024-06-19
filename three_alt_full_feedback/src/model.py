import numpy as np
from numba import njit
from helpers import softmax, select_action

@njit
def sample_softmax_rl(theta, context):
    context = context[:, :3]
    num_steps = theta.shape[0]
    values = np.full(6, 15) / 30
    resp = np.zeros(num_steps)
    for t in range(num_steps):
        if context[t, -1] == 0:
            curr_alt = np.array([0, 1, 2], dtype=np.int32)
        elif context[t, -1] == 1:
            curr_alt = np.array([0, 2, 4], dtype=np.int32)
        elif context[t, -1] == 2:
            curr_alt = np.array([3, 4, 5], dtype=np.int32)
        else:
            curr_alt = np.array([1, 3, 5], dtype=np.int32)

        action_probs = softmax(values[curr_alt], theta[t, 2])
        resp[t] = select_action(np.arange(3), action_probs)
        selected_alt = curr_alt[int(resp[t])]
        not_selected_alt = np.delete(curr_alt, int(resp[t]))
        values[selected_alt] = values[selected_alt] + theta[t, 0] * (context[t, int(resp[t])] - values[selected_alt])
        values[not_selected_alt] = values[not_selected_alt] + theta[t, 1] * (context[t, np.delete(np.arange(3), int(resp[t]))] - values[not_selected_alt])
        
    return resp