import numpy as np
from numpy import sin, cos, pi
from scipy.io import loadmat
from scipy.interpolate import PchipInterpolator


def normalize_range(x, a, b):
    xmin, xmax = np.min(x), np.max(x)
    return (x - xmin) / (xmax - xmin) * (b - a) + a


def func_desired_traj(
    traj_type,
    bridge_type,
    time_infor,
    control_infor,
    properties,
    dt,
    plot_movie,
    traj_frequency,
):
    plot_figures_inside = False

    val_length = time_infor["val_length"]
    train_length = time_infor["train_length"]

    # Unpack robot parameters
    m1, m2, l1, l2, lc1, lc2, I1, I2 = properties

    q_control = control_infor["q_control"]
    qdt_control = control_infor["qdt_control"]
    q2dt_control = control_infor["q2dt_control"]
    tau_control = control_infor["tau_control"]

    value_q = q_control[0, :]
    x_start = l1 * cos(value_q[0]) + l2 * cos(value_q[0] + value_q[1])
    y_start = l1 * sin(value_q[0]) + l2 * sin(value_q[0] + value_q[1])

    t = np.arange(0, val_length + dt, dt)

    # --------------------------------------------------
    # Generate reference trajectory
    # --------------------------------------------------
    if traj_type == "circle":
        x = 0.5 * cos(2 * pi * t / traj_frequency)
        y = 0.5 * sin(2 * pi * t / traj_frequency)

    elif traj_type == "infty":
        x = 0.25 * sin(2 * pi * t / (2 * traj_frequency))
        y = 0.15 * sin(2 * pi * t / traj_frequency)

    elif traj_type == "astroid":
        x = 0.4 * cos(2 * pi * t / 250) ** 3
        y = 0.4 * sin(2 * pi * t / 250) ** 3

    elif traj_type == "heart":
        f = 250
        a = 0.4
        theta = 2 * pi * t / f
        r = 1 - sin(theta)
        x = normalize_range(r * cos(theta), -a, a)
        y = normalize_range(r * sin(theta), -a, a)

    elif traj_type == "epitrochoid":
        a, b, c, f = 5, 3, 5, 200
        x = (a + b) * cos(2 * pi * t / f) - c * cos((a / b + 1) * 2 * pi * t / f)
        y = (a + b) * sin(2 * pi * t / f) - c * sin((a / b + 1) * 2 * pi * t / f)
        x = normalize_range(x, -0.4, 0.4)
        y = normalize_range(y, -0.4, 0.4)

    elif traj_type == "fermat":
        f, a = 100, 0.5
        theta = 2 * pi * t / f
        r = np.sqrt(a ** 2 * theta)
        x = normalize_range(r * cos(theta), -4, 4)
        y = normalize_range(r * sin(theta), -4, 4)

    elif traj_type == "lissajous":
        f, a, b = 300, 1, 3
        x = sin(a * t * 2 * pi / f + pi / 4)
        y = sin(b * t * 2 * pi / f)
        x = normalize_range(x, -0.3, 0.3)
        y = normalize_range(y, -0.3, 0.3)

    elif traj_type in ["lorenz", "chua", "rossler", "sprott_1", "sprott_4", "mg17", "mg30", "lorenz96"]:
        data = loadmat(f"./read_data/{traj_type}.mat")
        ts = data["ts_train"]

        x = normalize_range(ts[: val_length * 2 + 1, 0], -0.5, 0.5)
        y = normalize_range(ts[: val_length * 2 + 1, 1], -0.5, 0.5)

    else:
        raise ValueError("Invalid traj_type")

    x = x[: 2 * val_length + 1]
    y = y[: 2 * val_length + 1]

    # --------------------------------------------------
    # Find closest point for bridge
    # --------------------------------------------------
    add_id = 0
    closest = np.inf
    for i in range(min(100000, val_length)):
        d = np.sqrt((x_start - x[i]) ** 2 + (y_start - y[i]) ** 2)
        if d < closest and not (round(x[i], 6) == 0 and round(y[i], 6) == 0):
            closest = d
            add_id = i

    bridge_point = np.array([x[add_id], y[add_id]])
    bridge_len = np.linalg.norm(bridge_point - np.array([x_start, y_start]))
    bridge_time = int(round(bridge_len / dt))

    # --------------------------------------------------
    # Build bridge (cubic only, as in MATLAB default)
    # --------------------------------------------------
    if bridge_type == "cubic":
        t_bg = np.arange(0, bridge_time * dt + dt, dt)
        theta0 = q_control[0, :]
        theta0_dot = qdt_control[0, :]

        q2_bg = np.arccos((x[add_id:add_id + 2] ** 2 + y[add_id:add_id + 2] ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))
        q1_bg = np.arctan2(y[add_id:add_id + 2], x[add_id:add_id + 2]) - np.arctan2(
            l2 * sin(q2_bg), l1 + l2 * cos(q2_bg)
        )

        theta1 = np.array([q1_bg[0], q2_bg[0]])
        theta1_dot = (np.array([q1_bg[1], q2_bg[1]]) - theta1) / dt

        T = bridge_time
        a0 = theta0
        a1 = theta0_dot
        a2 = 3 * (theta1 - theta0) / T ** 2 - 2 * theta0_dot / T - theta1_dot / T
        a3 = -2 * (theta1 - theta0) / T ** 3 + (theta1_dot + theta0_dot) / T ** 2

        q1_bridge = a0[0] + a1[0] * t_bg + a2[0] * t_bg ** 2 + a3[0] * t_bg ** 3
        q2_bridge = a0[1] + a1[1] * t_bg + a2[1] * t_bg ** 2 + a3[1] * t_bg ** 3

        x_bridge = l1 * cos(q1_bridge) + l2 * cos(q1_bridge + q2_bridge)
        y_bridge = l1 * sin(q1_bridge) + l2 * sin(q1_bridge + q2_bridge)

        x = np.concatenate([x_bridge[1:-1], x[add_id:]])
        y = np.concatenate([y_bridge[1:-1], y[add_id:]])

    else:
        raise NotImplementedError("Only cubic bridge implemented")

    # --------------------------------------------------
    # Inverse kinematics
    # --------------------------------------------------
    q2 = np.arccos((x ** 2 + y ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))
    q1 = np.arctan2(y, x) - np.arctan2(l2 * sin(q2), l1 + l2 * cos(q2))

    q_control[1:val_length + 1, :] = np.column_stack((q1[:val_length], q2[:val_length]))
    qdt_control[1:val_length + 1, :] = np.diff(q_control[: val_length + 1], axis=0) / dt
    q2dt_control[:val_length, :] = np.diff(qdt_control[: val_length + 1], axis=0) / dt

    # --------------------------------------------------
    # Torque computation
    # --------------------------------------------------
    for i in range(val_length):
        H11 = m1 * lc1 ** 2 + I1 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(q_control[i, 1])) + I2
        H12 = m2 * l1 * lc2 * cos(q_control[i, 1]) + m2 * lc2 ** 2 + I2
        H21 = H12
        H22 = m2 * lc2 ** 2 + I2
        h = m2 * l1 * lc2 * sin(q_control[i, 1])

        part1 = -h * qdt_control[i, 1] * qdt_control[i, 0] - h * (qdt_control[i, 0] + qdt_control[i, 1]) * qdt_control[i, 1]
        part2 = h * qdt_control[i, 0] ** 2

        tau_control[i, 0] = H11 * q2dt_control[i, 0] + H12 * q2dt_control[i, 1] + part1
        tau_control[i, 1] = H21 * q2dt_control[i, 0] + H22 * q2dt_control[i, 1] + part2

    control_infor["q_control"] = q_control
    control_infor["qdt_control"] = qdt_control
    control_infor["q2dt_control"] = q2dt_control
    control_infor["tau_control"] = tau_control

    time_infor["val_length"] = val_length

    return control_infor, time_infor
