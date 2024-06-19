import numpy as np
from scipy.stats import halfnorm, uniform
from keras.utils import to_categorical

GLOBAL_PRIOR_MEAN = np.array([0.02, 0.02, 0.8])
GLOBAL_PRIOR_STD = np.array([0.01, 0.01, 0.6])
LOCAL_PRIOR_MEAN = np.array([0.4, 0.4, 5.7])
LOCAL_PRIOR_STD = np.array([0.25, 0.25, 7.6])

def configure_input(raw_dict):
    data = raw_dict.get("sim_data")[:, :, None]
    feedback = np.array(raw_dict.get("sim_batchable_context"))[:, :, :3]
    # condition = np.array(raw_dict.get("sim_batchable_context"))[:, :, 2][:, :, None]
    summary_conditions = np.c_[
        to_categorical(data), feedback,
    ]
    theta_t = raw_dict.get("local_prior_draws")
    eta = raw_dict.get("hyper_prior_draws")
    out_dict = dict(
        local_parameters=((theta_t - LOCAL_PRIOR_MEAN) / LOCAL_PRIOR_STD).astype(np.float32),
        hyper_parameters=((eta - GLOBAL_PRIOR_MEAN) / GLOBAL_PRIOR_STD).astype(np.float32),
        summary_conditions=summary_conditions.astype(np.float32),
    )
    return out_dict