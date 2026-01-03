import numpy as np
from simulator.multijoint_robot import MultiJointRobot


def generate_multijoint_data(
    n_joints,
    T=100000,
    dt=0.01,
    tau_scale=0.5
):
    """
    Generate training data for a multi-joint planar robot.

    The generated data is suitable for learning task-space
    residual dynamics (scalable across joint counts).

    Returns time series of:
    - end-effector position y
    - end-effector velocity yd
    - joint positions q
    - joint velocities qd
    - applied joint torques tau
    """

    robot = MultiJointRobot(n_joints, dt)

    # --------------------------------------------------
    # Logs
    # --------------------------------------------------
    y_log = []
    yd_log = []
    q_log = []
    qd_log = []
    tau_log = []

    y_prev = None

    # --------------------------------------------------
    # Simulation loop
    # --------------------------------------------------
    for t in range(T):
        # random bounded excitation
        tau = tau_scale * np.random.randn(n_joints)

        # integrate dynamics
        q, qd = robot.step(tau)

        # task-space position
        y = robot.end_effector()

        # task-space velocity (finite difference)
        if y_prev is None:
            yd = np.zeros(2)
        else:
            yd = (y - y_prev) / dt

        # log everything
        y_log.append(y)
        yd_log.append(yd)
        q_log.append(q)
        qd_log.append(qd)
        tau_log.append(tau)

        y_prev = y

    # --------------------------------------------------
    # Package data
    # --------------------------------------------------
    return {
        "y": np.asarray(y_log),      # (T, 2)
        "yd": np.asarray(yd_log),    # (T, 2)
        "q": np.asarray(q_log),      # (T, n_joints)
        "qd": np.asarray(qd_log),    # (T, n_joints)
        "tau": np.asarray(tau_log),  # (T, n_joints)
    }
