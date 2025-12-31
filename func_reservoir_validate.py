# func_reservoir_validate.py

import numpy as np
from numpy import cos, sin, tanh


def func_reservoir_validate(
    traj_type,
    bridge_type,
    time_infor,
    input_infor,
    res_infor,
    start_info,
    properties,
    dim_in,
    dim_out,
    Wout,
    r_end,
    dt,
    disturbance,
    measurement_noise,
    plot_movie,
    save_rend,
    failure,
    blur,
    traj_frequency,
):

    """
    Faithful Python translation of func_reservoir_validate.m
    """

    # --------------------------------------------------
    # Unpack parameters
    # --------------------------------------------------
    if save_rend == 1:
        q = start_info["q"]
        qdt = start_info["qdt"]
        q2dt = start_info["q2dt"]
        tau = start_info["tau"]

    val_length = time_infor["val_length"]

    # Equivalent of matsplit(properties)
    m1, m2, l1, l2, lc1, lc2, I1, I2 = properties

    W_in = res_infor["W_in"]
    res_net = res_infor["res_net"]
    alpha = res_infor["alpha"]
    kb = res_infor["kb"]
    n = res_infor["n"]

    # --------------------------------------------------
    # Failure handling
    # --------------------------------------------------
    failure_type = failure["type"]
    if failure_type != "none":
        failure_amplitude = failure["amplitude"]
        if failure_type == "all":
            failure_amplitude_2 = failure["amplitude_2"]

    # --------------------------------------------------
    # Allocate memory
    # --------------------------------------------------
    val_pred_y = np.zeros((val_length, dim_out))
    val_real_y = np.zeros((val_length, dim_out))

    if save_rend == 1:
        r = r_end.reshape(n, 1)
    else:
        r = np.zeros((res_net.shape[0], 1))

    u = np.zeros((dim_in, 1))

    q_control = np.zeros((val_length + 100, 2))
    qdt_control = np.zeros((val_length + 100, 2))
    q2dt_control = np.zeros((val_length + 100, 2))
    tau_control = np.zeros((val_length + 100, 2))

    # --------------------------------------------------
    # Initial condition
    # --------------------------------------------------
    if save_rend == 1:
        q_control[0, :] = q
        qdt_control[0, :] = qdt
        q2dt_control[0, :] = q2dt
        tau_control[0, :] = tau
    else:
        if traj_type == "infty":
            q_control[0, 0] = (3 - 1) * np.random.rand() + 1
            q_control[0, 1] = (0.0 - 2.4) * np.random.rand()
        else:
            q_control[0, 0] = (6 - 4) * np.random.rand() + 4
            q_control[0, 1] = (0.0 - 2.4) * np.random.rand() - 0.1

    q_pred = q_control.copy()
    qdt_pred = qdt_control.copy()
    q2dt_pred = q2dt_control.copy()
    tau_pred = tau_control.copy()

    control_infor = {
        "q_control": q_control,
        "qdt_control": qdt_control,
        "q2dt_control": q2dt_control,
        "tau_control": tau_control,
    }

    # --------------------------------------------------
    # Desired trajectory
    # --------------------------------------------------
    from func_desired_traj import func_desired_traj

    control_infor, time_infor = func_desired_traj(
        traj_type,
        bridge_type,
        time_infor,
        control_infor,
        properties,
        dt,
        plot_movie,
        traj_frequency,
    )

    q_control = control_infor["q_control"]
    qdt_control = control_infor["qdt_control"]
    val_length = time_infor["val_length"]

    x_control = l1 * cos(q_control[:, 0]) + l2 * cos(q_control[:, 0] + q_control[:, 1])
    y_control = l1 * sin(q_control[:, 0]) + l2 * sin(q_control[:, 0] + q_control[:, 1])
    data_control = np.column_stack((x_control, y_control))
    control_infor["data_control"] = data_control

    # --------------------------------------------------
    # Input configuration
    # --------------------------------------------------
    input_infor_label = 0
    if (
        len(input_infor) == 2
        and input_infor[0] == "xy"
        and input_infor[1] == "qdt"
    ):
        input_infor_label = 1
        u[:, 0] = np.concatenate(
            [
                data_control[0, :],
                data_control[1, :],
                qdt_control[0, :],
                qdt_control[1, :],
            ]
        )

    data_pred = data_control.copy()

    # --------------------------------------------------
    # Noise & failure
    # --------------------------------------------------
    np.random.seed(None)

    disturbance_failure = np.zeros((2, val_length))
    measurement_failure = np.zeros((dim_in // 2, val_length))

    if failure_type == "disturbance":
        disturbance_failure = np.random.randn(2, val_length) * failure_amplitude
    elif failure_type == "measurement":
        measurement_failure = np.random.randn(dim_in // 2, val_length) * failure_amplitude
    elif failure_type == "all":
        disturbance_failure = np.random.randn(2, val_length) * failure_amplitude
        measurement_failure = (
            np.random.randn(dim_in // 2, val_length) * failure_amplitude_2
        )

    taudt_threshold = np.array([-5e-2, 5e-2])

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    for t_i in range(val_length - 3):
        r = (1 - alpha) * r + alpha * tanh(res_net @ r + W_in @ u + kb)

        r_out = r.copy()
        r_out[1::2, 0] = r_out[1::2, 0] ** 2

        predict_value = (Wout @ r_out).flatten()
        predict_value += predict_value * disturbance_failure[:, t_i]

        time_li = max(t_i - 1, 0)
        for li in range(2):
            delta = predict_value[li] - tau_pred[time_li, li]
            delta = np.clip(delta, taudt_threshold[0] * dt, taudt_threshold[1] * dt)
            predict_value[li] = tau_pred[time_li, li] + delta

        tau_pred[t_i, :] = predict_value

        H11 = (
            m1 * lc1**2
            + I1
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(q_pred[t_i, 1]))
            + I2
        )
        H12 = m2 * l1 * lc2 * cos(q_pred[t_i, 1]) + m2 * lc2**2 + I2
        H21 = H12
        H22 = m2 * lc2**2 + I2
        h = m2 * l1 * lc2 * sin(q_pred[t_i, 1])

        part_1 = -h * qdt_pred[t_i, 1] * qdt_pred[t_i, 0] - h * (
            qdt_pred[t_i, 0] + qdt_pred[t_i, 1]
        ) * qdt_pred[t_i, 1]
        part_2 = h * qdt_pred[t_i, 0] ** 2

        denominator = H12 * H21 - H11 * H22

        q2dt_pred[t_i, 0] = -(
            -part_1 * H22
            + H12 * part_2
            - H12 * predict_value[1]
            + H22 * predict_value[0]
        ) / denominator

        q2dt_pred[t_i, 1] = -(
            part_1 * H21
            - H11 * part_2
            + H11 * predict_value[1]
            - H21 * predict_value[0]
        ) / denominator

        q_pred[t_i + 1, :] = q_pred[t_i, :] + qdt_pred[t_i, :] * dt
        qdt_pred[t_i + 1, :] = qdt_pred[t_i, :] + q2dt_pred[t_i, :] * dt

        x_pred = l1 * cos(q_pred[t_i + 1, 0]) + l2 * cos(
            q_pred[t_i + 1, 0] + q_pred[t_i + 1, 1]
        )
        y_pred = l1 * sin(q_pred[t_i + 1, 0]) + l2 * sin(
            q_pred[t_i + 1, 0] + q_pred[t_i + 1, 1]
        )

        data_pred[t_i + 1, :] = [x_pred, y_pred]

        if input_infor_label == 1:
            u[0:2, 0] = [x_pred, y_pred]
            u[2:4, 0] = data_control[t_i + 2, :]
            u[4:6, 0] = qdt_pred[t_i + 1, :]
            u[6:8, 0] = qdt_control[t_i + 2, :]

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    output_infor = {
        "data_pred": data_pred,
        "q_pred": q_pred,
        "qdt_pred": qdt_pred,
        "q2dt_pred": q2dt_pred,
        "tau_pred": tau_pred,
    }

    return control_infor, output_infor, time_infor, r
