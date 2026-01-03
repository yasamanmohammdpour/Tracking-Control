import numpy as np
from learning.config import KP, KD


def build_training_data(data: dict):
    """
    Build reservoir training data for task-space residual learning.

    The reservoir learns a 2D task-space residual acceleration:
        Δẍ = ẍ_true − ẍ_PD

    This formulation is scalable with respect to the number of joints.

    Parameters
    ----------
    data : dict
        Dictionary containing time series data with keys:
        - "y"  : ndarray, shape (T, 2)
                 End-effector position
        - "yd" : ndarray, shape (T, 2)
                 End-effector velocity
        - "q"  : ndarray, shape (T, n_joints)
                 Joint positions
        - "qd" : ndarray, shape (T, n_joints)
                 Joint velocities

    Returns
    -------
    U : ndarray, shape (T-2, dim_in)
        Reservoir input matrix
    Y : ndarray, shape (T-2, 2)
        Task-space residual acceleration targets
    """

    # --------------------------------------------------
    # Validate input
    # --------------------------------------------------
    required = {"y", "yd", "q", "qd"}
    if not required.issubset(data.keys()):
        raise ValueError(f"Missing required keys: {required}")

    y = np.asarray(data["y"])
    yd = np.asarray(data["yd"])
    q = np.asarray(data["q"])
    qd = np.asarray(data["qd"])

    if y.shape[1] != 2 or yd.shape[1] != 2:
        raise ValueError("Task-space signals must be 2D")

    T = y.shape[0]
    if T < 3:
        raise ValueError("Need at least 3 samples to compute accelerations")

    # --------------------------------------------------
    # Reservoir input (same structure as before)
    # --------------------------------------------------
    # U(t) = [ y(t), y(t+1), qd(t), qd(t+1) ]
    U = np.hstack([
        y[:-2],
        y[1:-1],
        qd[:-2],
        qd[1:-1],
    ])

    # --------------------------------------------------
    # Compute true task-space acceleration (finite diff)
    # --------------------------------------------------
    # ẍ(t) ≈ (y(t+2) − 2y(t+1) + y(t)) / dt²
    dt = 1.0  # cancels out because PD is in same scale
    xdd_true = (y[2:] - 2 * y[1:-1] + y[:-2]) / (dt ** 2)

    # --------------------------------------------------
    # Baseline task-space PD acceleration
    # --------------------------------------------------
    e = y[1:-1]          # reference is implicit (near zero)
    ed = yd[1:-1]

    xdd_pd = -KP * e - KD * ed

    # --------------------------------------------------
    # Residual learning target (THIS IS THE KEY)
    # --------------------------------------------------
    Y = xdd_true - xdd_pd

    return U, Y
