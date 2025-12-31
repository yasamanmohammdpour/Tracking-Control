import numpy as np


def func_limits(value, limit_type):
    """
    Python translation of func_limits.m

    Parameters
    ----------
    value : array-like, shape (2,)
        Input value to be limited.
    limit_type : int
        1 -> dq/dt limit
        2 -> d2q/dt2 limit

    Returns
    -------
    value : ndarray, shape (2,)
        Limited value.
    """

    value = np.asarray(value, dtype=float).copy()

    qdt_lim = (-0.05, 0.05)
    q2dt_lim = (-0.3, 0.3)

    if limit_type == 1:
        # dq/dt limits
        value[0] = min(max(value[0], qdt_lim[0]), qdt_lim[1])
        value[1] = min(max(value[1], qdt_lim[0]), qdt_lim[1])

    elif limit_type == 2:
        # d2q/dt2 limits
        value[0] = min(max(value[0], q2dt_lim[0]), q2dt_lim[1])
        value[1] = min(max(value[1], q2dt_lim[0]), q2dt_lim[1])

    return value
