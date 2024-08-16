import numpy as np
import pandas as pd

RNG = np.random.default_rng()

DATA = pd.read_csv("../data/data_prepared.csv")
SUBJECTS = [
    3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 29, 30, 33, 34, 36, 40, 41, 42, 43, 47, 47, 48
]

def generate_context():
    random_sub = RNG.choice(
        SUBJECTS, 1, replace=False
    )[0]
    person_data = DATA.loc[DATA.id == random_sub]
    context = person_data[['stim_set', 'p_a', 'p_b']].to_numpy().astype(np.float32)
    return context