import numpy as np


def rm_nan(x):
    if not np.isnan(x).any():
        return x
    if len(x.shape) == 1:
        return x[~np.isnan(x)]
    return x[~np.isnan(x).any(axis=1)]

# Keys that we ignore when reading ASE arrays
ASE_ARRAYS_KEYS = ["numbers", "positions", "tags", "momenta", "masses", "initial_magmoms", "initial_charges"]