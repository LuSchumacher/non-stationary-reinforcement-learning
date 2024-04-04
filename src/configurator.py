import numpy as np
from scipy.stats import halfnorm
from keras.utils import to_categorical

GLOBAL_PRIOR_MEAN = halfnorm(0, [0.05, 3]).mean().round(decimals=2)
GLOBAL_PRIOR_STD = halfnorm(0, [0.05, 3]).std().round(decimals=2)
LOCAL_PRIOR_MEAN = np.array([0.6, 20])
LOCAL_PRIOR_STD = np.array([0.3, 20])

def configure_input(raw_dict):
    """
    Process raw dictionary data into formatted input for a model.

    This function takes a dictionary containing raw simulation data and prior draws,
    and formats it into input data suitable for a machine learning model. It extracts simulation
    data, feedback, and options from the dictionary and encodes them appropriately. It also scales
    the local and hyper prior draws.

    Parameters
    ----------
    raw_dict : dict
        A dictionary containing the following keys:
        - "sim_data": Simulation data.
        - "sim_batchable_context": Simulation batchable context, including feedback and options.
        - "local_prior_draws": Local prior draws.
        - "hyper_prior_draws": Hyper prior draws.

    Returns
    -------
    dict
        A dictionary containing formatted input data for the model, with the following keys:
        - "local_parameters": Scaled local prior parameters.
        - "hyper_parameters": Scaled hyper prior parameters.
        - "summary_conditions": Encoded and formatted summary conditions for the model.

    Notes
    -----
    This function assumes that "sim_data", "sim_batchable_context", "local_prior_draws", and
    "hyper_prior_draws" are present in the raw dictionary.

    """
    data = raw_dict.get("sim_data")[:, :, None]
    feedback = np.array(raw_dict.get("sim_batchable_context"))[:, :, :2]
    cor_option = np.array(raw_dict.get("sim_batchable_context"))[:, :, 2][:, :, None]
    inc_option = np.array(raw_dict.get("sim_batchable_context"))[:, :, 3][:, :, None]
    summary_conditions = np.c_[
        data, feedback, to_categorical(cor_option), to_categorical(inc_option)
    ]
    theta_t = raw_dict.get("local_prior_draws")
    eta = raw_dict.get("hyper_prior_draws")
    out_dict = dict(
        local_parameters=((theta_t - LOCAL_PRIOR_MEAN) / LOCAL_PRIOR_STD).astype(np.float32),
        hyper_parameters=((eta - GLOBAL_PRIOR_MEAN) / GLOBAL_PRIOR_STD).astype(np.float32),
        summary_conditions=summary_conditions.astype(np.float32),
    )
    return out_dict