import numpy as np

RNG = np.random.default_rng()
NUM_STEPS = 200
STEPS_PER_CONDITION = 50

def generate_context():
    condition_a = RNG.normal(
        loc=(30, 34, 38), scale=5, size=(STEPS_PER_CONDITION, 3)
    )
    condition_a = np.c_[condition_a, np.repeat(0, STEPS_PER_CONDITION)[:, None]]
    condition_b = RNG.normal(
        loc=(30, 38, 46), scale=5, size=(STEPS_PER_CONDITION, 3)
    )
    condition_b = np.c_[condition_b, np.repeat(1, STEPS_PER_CONDITION)[:, None]]
    condition_c = RNG.normal(
        loc=(42, 46, 50), scale=5, size=(STEPS_PER_CONDITION, 3)
    )
    condition_c = np.c_[condition_c, np.repeat(2, STEPS_PER_CONDITION)[:, None]]
    condition_d = RNG.normal(
        loc=(34, 42, 50), scale=5, size=(STEPS_PER_CONDITION, 3)
    )
    condition_d = np.c_[condition_d, np.repeat(3, STEPS_PER_CONDITION)[:, None]]
    context = np.concatenate(
        [condition_a, condition_b, condition_c, condition_d]
    )
    idx = RNG.choice(
        np.arange(NUM_STEPS), NUM_STEPS, replace=False
    )
    return context[idx].astype(np.int32) / 30