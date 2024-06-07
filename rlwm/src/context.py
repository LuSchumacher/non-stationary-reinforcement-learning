import numpy as np
import pandas as pd

RNG = np.random.default_rng()

DATA = pd.read_csv("../data/data_prepared.csv")
NUM_SUB = len(np.unique(DATA.id))

MIN_STEPS = 600
MAX_STEPS = 780

def generate_context(num_steps):
    idx = RNG.choice(
        np.arange(MAX_STEPS), MAX_STEPS - num_steps, replace=False
    )
    mask = np.full(MAX_STEPS, 1).astype(bool)
    mask[idx] = False
    sub = RNG.choice(np.unique(DATA.id))
    sub_data = DATA.loc[DATA.id == sub]
    return sub_data[["stim", "correct_resp", "block", "set_size"]].to_numpy()[mask, :]

def random_num_steps(min_obs=MIN_STEPS, max_obs=MAX_STEPS):
    return RNG.integers(low=min_obs, high=max_obs + 1)