import numpy as np
from scipy.stats import halfnorm
from keras.utils import to_categorical

THETA_PRIOR_MEAN = np.array([0.5, 0.5])
THETA_PRIOR_STD = np.array([0.3, 0.3])
ETA_PRIOR_MEAN = np.round(halfnorm(0, 0.05).mean(), decimals=2)
ETA_PRIOR_STD = np.round(halfnorm(0, 0.05).std(), decimals=2)
KAPPA_PRIOR_MEAN = np.array([0.5, 4.7])
KAPPA_PRIOR_STD = np.array([0.3, 1])

def configure_input(forward_dict):
    out_dict = {}

    data = forward_dict["sim_data"]
    resp_one_hot = to_categorical(data[:, :, 0])

    context = np.array(forward_dict["batchable_context"])
    stim_one_hot = to_categorical(context[:, :, 0])
    correct_resp_one_hot = to_categorical(context[:, :, 1])
    block = (context[:, :, 2] / 13)[:, :, None]
    set_size = ((context[:, :, 3] / 3) - 1)[:, :, None]

    out_dict["summary_conditions"] = np.c_[
        resp_one_hot, data[:, :, 1:], stim_one_hot,
        correct_resp_one_hot, block, set_size
    ].astype(np.float32)

    vec_num_obs = forward_dict["non_batchable_context"] * np.ones((data.shape[0], 1))
    out_dict["direct_conditions"] = np.sqrt(vec_num_obs).astype(np.float32)

    theta = forward_dict['local_parameters']
    eta = forward_dict['global_parameters']
    kappa = forward_dict['shared_parameters']

    out_dict["local_parameters"] = ((theta - THETA_PRIOR_MEAN) / THETA_PRIOR_STD).astype(np.float32)
    out_dict["hyper_parameters"] = ((eta - ETA_PRIOR_MEAN) / ETA_PRIOR_STD).astype(np.float32)
    out_dict["shared_parameters"] = ((kappa - KAPPA_PRIOR_MEAN) / KAPPA_PRIOR_STD).astype(np.float32)

    return out_dict