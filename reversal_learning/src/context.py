import numpy as np
import pandas as pd

RNG = np.random.default_rng()

DATA = pd.read_csv("../data/data_prepared.csv")
SUBJECTS = np.unique(DATA.id)

def generate_context():
    random_sub = RNG.choice(
        SUBJECTS, 1, replace=False
    )[0]
    person_data = DATA.loc[DATA.id == random_sub]
    context = person_data[['stim_set', 'p_a', 'p_b']].to_numpy().astype(np.float32)
    return context