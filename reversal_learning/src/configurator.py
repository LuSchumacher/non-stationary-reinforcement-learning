import numpy as np
from scipy.stats import halfnorm, uniform
from keras.utils import to_categorical

GLOBAL_PRIOR_MEAN = np.concatenate(
    [halfnorm(0, [0.02, 0.5]).mean().round(decimals=2),
    uniform(0, [0.04, 0.04]).mean().round(decimals=2)]
)
GLOBAL_PRIOR_STD = np.concatenate(
    [halfnorm(0, [0.02, 0.5]).std().round(decimals=2),
    uniform(0, [0.04, 0.04]).std().round(decimals=2)]
)
LOCAL_PRIOR_MEAN = np.array([0.43, 4.87])
LOCAL_PRIOR_STD = np.array([0.25, 3.63])

def configure_input(forward_dict):
    data = forward_dict.get("sim_data")
    context = np.array(forward_dict.get("sim_batchable_context"))[:, :, 1:]
    summary_conditions = np.c_[data, context]

    theta = forward_dict.get("local_prior_draws")
    eta = forward_dict.get("hyper_prior_draws")

    out_dict = dict(
        local_parameters=((theta - LOCAL_PRIOR_MEAN) / LOCAL_PRIOR_STD).astype(np.float32),
        hyper_parameters=((eta - GLOBAL_PRIOR_MEAN) / GLOBAL_PRIOR_STD).astype(np.float32),
        summary_conditions=summary_conditions.astype(np.float32),
    )
    return out_dict