import numpy as np
from numba import njit

from helpers import softmax, select_action
from priors import sample_eta, sample_kappa, sample_random_walk
from context import generate_context, random_num_steps

@njit
def sample_rlwm(theta, kappa, context):
    phi, c = kappa
    tau = 10
    num_steps = context.shape[0]
    sim_data = np.zeros((num_steps, 2))
    current_block = -1
    for t in range(num_steps):
        # reset subjective values
        if context[t, 2] != current_block:
            current_block = context[t, 2]
            set_size = int(context[t, 3])
            q_values = np.full((set_size, 3), 1/3)
            w_values = np.full((set_size, 3), 1/3)

        w = theta[t, 1] * np.minimum(1, c/set_size)
        current_stim = int(context[t, 0])

        # choice selection
        pi_rl = softmax(q_values[current_stim], tau)
        pi_wm = softmax(w_values[current_stim], tau)
        pi = w*pi_wm + (1 - w)*pi_rl
        sim_data[t, 0] = select_action(np.arange(3), pi)
        current_resp = int(sim_data[t, 0])

        # feedback
        if sim_data[t, 0] == context[t, 1]:
            sim_data[t, 1] = 1
        else:
            sim_data[t, 1] = 0

        # update values
        pe = sim_data[t, 1] - q_values[current_stim, current_resp]
        q_values[current_stim, current_resp] += theta[t, 0] * pe
        # memory decay
        w_values += phi * (1/3 - w_values)
        # update values
        w_values[current_stim, current_resp] = sim_data[t, 1]

    return sim_data

def generative_model(batch_size=32):
    sim_dict = {}
    num_steps = random_num_steps()
    sim_dict['non_batchable_context'] = num_steps
    context = np.zeros((batch_size, num_steps, 4))
    sim_data = np.zeros((batch_size, num_steps, 2))
    eta = np.zeros((batch_size, 2))
    kappa = np.zeros((batch_size, 2))
    theta = np.zeros((batch_size, num_steps, 2))
    for i in range(batch_size):
        eta[i] = sample_eta()
        kappa[i] = sample_kappa()
        context[i] = generate_context(num_steps)
        theta[i] = sample_random_walk(eta[i], context[i, :, 2])
        sim_data[i] = sample_rlwm(theta[i], kappa[i], context[i])
    sim_dict['global_parameters'] = eta.astype(np.float32)
    sim_dict['shared_parameters'] = kappa.astype(np.float32)
    sim_dict['local_parameters'] = theta.astype(np.float32)
    sim_dict['batchable_context'] = context.astype(np.int32)
    sim_dict['sim_data'] = sim_data.astype(np.int32)

    return sim_dict