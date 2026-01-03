import numpy as np


def compute_rmse(y: np.ndarray, y_ref: np.ndarray) -> float:
    """
    Compute root-mean-square error (RMSE) between actual and reference trajectories.

    Parameters
    ----------
    y : ndarray, shape (T, 2)
        Actual end-effector trajectory.
    y_ref : ndarray, shape (T, 2)
        Reference end-effector trajectory.

    Returns
    -------
    rmse : float
        Root-mean-square tracking error.
    """

    if y.shape != y_ref.shape:
        raise ValueError(
            f"Shape mismatch: y has shape {y.shape}, "
            f"but y_ref has shape {y_ref.shape}"
        )

    error = y - y_ref
    mse = np.mean(np.sum(error ** 2, axis=1))
    rmse = np.sqrt(mse)

    return rmse


def compute_divergence_time(
    y: np.ndarray,
    y_ref: np.ndarray,
    threshold: float = 1.0,
):
    """
    Compute the time index at which the tracking error exceeds a threshold.

    This is useful for identifying instability or loss of control.

    Parameters
    ----------
    y : ndarray, shape (T, 2)
        Actual end-effector trajectory.
    y_ref : ndarray, shape (T, 2)
        Reference end-effector trajectory.
    threshold : float
        Euclidean error threshold.

    Returns
    -------
    t_div : int or None
        Index where divergence occurs, or None if never diverged.
    """

    error_norm = np.linalg.norm(y - y_ref, axis=1)

    diverged = np.where(error_norm > threshold)[0]

    if len(diverged) == 0:
        return None

    return int(diverged[0])


def compute_mean_torque(
    tau: np.ndarray,
) -> float:
    """
    Compute mean torque magnitude over time.

    Parameters
    ----------
    tau : ndarray, shape (T, n_joints)
        Joint torque sequence.

    Returns
    -------
    mean_tau : float
        Mean L2 norm of torque.
    """

    tau_norm = np.linalg.norm(tau, axis=1)
    return float(np.mean(tau_norm))
