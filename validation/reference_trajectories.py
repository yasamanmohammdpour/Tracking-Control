import numpy as np


def generate_reference_trajectory(
    traj_type: str,
    T: int,
    dt: float,
    radius: float = 0.5,
    frequency: float = 0.2,
):
    """
    Generate 2D reference trajectories for closed-loop tracking.

    Parameters
    ----------
    traj_type : str
        Type of trajectory.
        Supported:
        - "circle"
        - "lissajous"
        - "figure8"
    T : int
        Number of time steps.
    dt : float
        Time step.
    radius : float
        Spatial scale of trajectory.
    frequency : float
        Temporal frequency.

    Returns
    -------
    y_ref : ndarray, shape (T, 2)
        Reference end-effector trajectory.
    """

    t = np.arange(T) * dt

    if traj_type == "circle":
        x = radius * np.cos(2 * np.pi * frequency * t)
        y = radius * np.sin(2 * np.pi * frequency * t)

    elif traj_type == "lissajous":
        x = radius * np.sin(2 * np.pi * frequency * t)
        y = radius * np.sin(4 * np.pi * frequency * t)

    elif traj_type == "figure8":
        x = radius * np.sin(2 * np.pi * frequency * t)
        y = radius * np.sin(2 * np.pi * frequency * t) * np.cos(
            2 * np.pi * frequency * t
        )

    else:
        raise ValueError(
            f"Unknown trajectory type '{traj_type}'. "
            "Supported: circle, lissajous, figure8"
        )

    return np.column_stack((x, y))
