import numpy as np
import pandas as pd

RNG = np.random.default_rng()

DATA = pd.read_csv("../data/data_fontanesi_prep.csv")
DATA.f_cor = DATA.f_cor / 60
DATA.f_inc = DATA.f_inc / 60

RELEVANT_SUB = np.array([1, 3, 5, 6, 7, 8, 16, 20, 22, 23, 24, 26, 27])
# BLOCKS = np.arange(3)


def generate_context():
    """
    Generate contextual information from a random subject and block.

    Randomly selects a subject from the unique subjects in the data and a block number from a predefined
    set of blocks. Retrieves the corresponding data for the selected subject and block from the global
    DATA dataframe, and returns an array containing features 'f_cor', 'f_inc', 'cor_option', and 'inc_option'
    for the selected subject and block.

    Returns
    -------
    np.ndarray
        A numpy array of shape (80, 4) containing contextual information:
        - f_cor: Feedback correct option
        - f_inc: Feedback incorrect option
        - cor_option: Correct option
        - inc_option: Incorrect option
    """
    sub = RNG.choice(RELEVANT_SUB)
    # block = RNG.choice(BLOCKS)
    # sub_data = DATA.loc[(DATA.id == sub) & (DATA.block == block)]
    sub_data = DATA.loc[DATA.id == sub]
    return sub_data[['f_cor', 'f_inc', 'cor_option', 'inc_option']].to_numpy()