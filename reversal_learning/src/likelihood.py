import numpy as np
from numba import njit
from helpers import softmax, select_action

BLOCK_IDX = np.arange(0, 512, 128)

@njit
def sample_softmax_rl(theta, context):
    num_steps = theta.shape[0]
    sim_data = np.zeros((num_steps, 2))
    for t in range(num_steps):
        if t in BLOCK_IDX:
            values = np.full(4, 0.5)
            # mean_value = np.mean(values)
            # values = np.full(4, mean_value)
        if context[t, 0] == 0:
            action_probs = softmax(values[:2], theta[t, 1])
            resp = select_action(np.array([0, 1]), action_probs)
            sim_data[t, 0] = resp
            sim_data[t, 1] = np.random.binomial(1, context[t, int(resp)+1])
            values[int(resp)] += theta[t, 0] * (sim_data[t, 1] - values[int(resp)])
        else:
            action_probs = softmax(values[2:], theta[t, 1])
            resp = select_action(np.array([0, 1]), action_probs)
            sim_data[t, 0] = resp
            sim_data[t, 1] = np.random.binomial(1, context[t, int(resp)+1])
            values[2:][int(resp)] += theta[t, 0] * (sim_data[t, 1] - values[2:][int(resp)])
    return sim_data
