import numpy as np


def rm_nan(x):
    if not np.isnan(x).any():
        return x
    if len(x.shape) == 1:
        return x[~np.isnan(x)]
    return x[~np.isnan(x).any(axis=1)]
