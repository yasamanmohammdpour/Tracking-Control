import os
import sys
import numpy as np

# ------------------------------------------------------------
# Add project root to PYTHONPATH
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulator.multijoint_robot import MultiJointRobot
from learning.reservoir import Reservoir
from learning.config import KP, KD
from validation.reference_trajectories import generate_reference_trajectory
from validation.metrics import compute_rmse, compute_divergence_time


# ------------------------------------------------------------
# Load trained reservoir
# ------------------------------------------------------------
def load_trained_reservoir(model_path: str) -> Reservoir:
    data = np.load(model_path)

    res = Reservoir(
        n_reservoir=data["W"].shape[0],
        dim_in=data["W_in"].shape[1],
        dim_out=data["W_out"].shape[0],  # MUST be 2
    )

    res.W = data["W"]
    res.W_in = data["W_in"]
    res.bias = data["bias"]
    res.W_out = data["W_out"]
    res.r_end = data["r_end"]

    return res


# ------------------------------------------------------------
# Closed-loop validation (task-space residual architecture)
# ------------------------------------------------------------
def run_closed_loop_validation(
    n_joints: int,
    T: int = 2000,
    dt: float = 0.01,
    traj_type: str = "circle",
):
    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------
    model_path = f"save_file/reservoir_n{n_joints}.npz"
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    reservoir = load_trained_reservoir(model_path)

    # --------------------------------------------------------
    # Initialize robot
    # --------------------------------------------------------
    robot = MultiJointRobot(n_joints=n_joints, dt=dt)

    # small initial velocities (break symmetry)
    robot.qd = 0.01 * np.random.randn(n_joints)

    # --------------------------------------------------------
    # Reference trajectory (task space)
    # --------------------------------------------------------
    y_ref = generate_reference_trajectory(
        traj_type=traj_type,
        T=T,
        dt=dt,
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    y_log = np.zeros((T, 2))
    y_ref_log = np.zeros((T, 2))
    tau_log = np.zeros((T, n_joints))

    # --------------------------------------------------------
    # Reservoir state
    # --------------------------------------------------------
    r = reservoir.r_end.copy()
    qd_prev = robot.qd.copy()

    y_prev = robot.end_effector()

    # --------------------------------------------------------
    # Control loop
    # --------------------------------------------------------
    for t in range(T - 1):
        # current state
        y = robot.end_effector()
        qd = robot.qd.copy()

        # task-space velocity
        yd = (y - y_prev) / dt

        # ----------------------------------------------------
        # Reservoir input
        # ----------------------------------------------------
        u = np.hstack([
            y,
            y_ref[t + 1],
            qd,
            qd_prev,
        ])

        # reservoir update
        r = (1 - reservoir.alpha) * r + reservoir.alpha * np.tanh(
            reservoir.W @ r + reservoir.W_in @ u + reservoir.bias
        )

        r_nl = r.copy()
        r_nl[1::2] = r_nl[1::2] ** 2

        # ----------------------------------------------------
        # Task-space PD (baseline stability)
        # ----------------------------------------------------
        e = y - y_ref[t]
        ed = yd

        u_task_pd = -KP * e - KD * ed

        # ----------------------------------------------------
        # Learned task-space residual (SCALABLE PART)
        # ----------------------------------------------------
        delta_u_task = reservoir.W_out @ r_nl   # shape (2,)

        u_task = u_task_pd + delta_u_task

        # ----------------------------------------------------
        # Map to joint torques
        # ----------------------------------------------------
        J = robot.jacobian()
        tau = J.T @ u_task

        robot.step(tau)

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        y_log[t] = y
        y_ref_log[t] = y_ref[t]
        tau_log[t] = tau

        qd_prev = qd.copy()
        y_prev = y.copy()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    rmse = compute_rmse(y_log, y_ref_log)
    t_div = compute_divergence_time(y_log, y_ref_log, threshold=1.0)

    print(f"[RESULT] Closed-loop RMSE ({traj_type}): {rmse:.6f}")
    print(f"[RESULT] Divergence time: {t_div}")

    return {
        "rmse": rmse,
        "divergence_time": t_div,
        "y": y_log,
        "y_ref": y_ref_log,
        "tau": tau_log,
    }


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    run_closed_loop_validation(
        n_joints=6,
        T=2000,
        dt=0.01,
        traj_type="circle",
    )
