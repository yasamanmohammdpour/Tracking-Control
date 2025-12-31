import numpy as np


def func_rmse(a, b, time_start, time_end):
    """
    Python translation of func_rmse.m

    Computes RMSE between two time-series matrices over a time window.
    """

    a = np.asarray(a)
    b = np.asarray(b)

    # MATLAB logic:
    # [c, d] = size(a)
    # len = max(c, d)
    c, d = a.shape
    length = max(c, d)

    # Ensure shape is (dim, time)
    if a.shape[1] != length:
        a = a.T

    if b.shape[1] != length:
        b = b.T

    # MATLAB is 1-based, Python is 0-based
    # MATLAB: time_start:time_end
    # Python: time_start-1 : time_end
    diff = a[:, time_start - 1 : time_end] - b[:, time_start - 1 : time_end]

    rmse = np.sqrt(np.mean(np.sum(diff ** 2, axis=0)))

    return rmse
