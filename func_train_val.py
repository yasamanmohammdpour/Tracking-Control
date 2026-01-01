# func_train_val.py

import numpy as np
import time

from robot_data_generator import robot_data_generator
from func_reservoir_train import func_reservoir_train
from func_reservoir_validate import func_reservoir_validate
from tools.func_rmse import func_rmse


def func_train_val(n, train_t, reset_t, noise_level, bias):
    """
    Python translation of func_train_val.m

    Returns:
        rmse_l, rmse_c, rmse_m, rmse_i, t_train
    """

    # --------------------------------------------------
    # Basic setup
    # --------------------------------------------------
    dt = 0.01
    input_infor = ["xy", "qdt"]

    disturbance = 0.0
    measurement_noise = 0.0

    dim_in = len(input_infor) * 4
    dim_out = 2

    # Robot properties
    m1 = 1.0
    m2 = 1.0
    l1 = 0.5
    l2 = 0.5
    lc1 = 0.25
    lc2 = 0.25
    I1 = 0.03
    I2 = 0.03

    properties = np.array([m1, m2, l1, l2, lc1, lc2, I1, I2])

    val_t = 500

    # --------------------------------------------------
    # Hyperparameters (Bayesian optimized)
    # --------------------------------------------------
    hyperpara_set = np.array([
        0.756250,
        0.756250,
        0.843750,
        -3.125,
        106.71875,
        bias,
    ])

    eig_rho = hyperpara_set[0]
    W_in_a = hyperpara_set[1]
    alpha = hyperpara_set[2]
    beta = 10 ** hyperpara_set[3]
    k = int(round(hyperpara_set[4] / 200 * n))
    kb = hyperpara_set[5]

    # --------------------------------------------------
    # Reservoir initialization
    # --------------------------------------------------
    W_in = W_in_a * (2 * np.random.rand(n, dim_in) - 1)

    # Symmetric sparse reservoir (MATLAB sprandsym)
    density = k / n
    res_net = np.random.randn(n, n)
    mask = np.random.rand(n, n) < density
    res_net = np.triu(res_net * mask)
    res_net = res_net + res_net.T

    # Spectral radius normalization
    eigvals = np.linalg.eigvals(res_net)
    eig_D = np.max(np.abs(eigvals))
    res_net = (eig_rho / eig_D) * res_net

    # --------------------------------------------------
    # Time parameters
    # --------------------------------------------------
    section_len = int(round(reset_t / dt))
    washup_length = int(round(1002 / dt))
    train_length = int(round(train_t / dt) + round(5 / dt))
    val_length = int(round(val_t / dt))

    time_length = (
        train_length
        + 2 * val_length
        + 3 * washup_length
        + 100
    )

    res_infor = {
        "W_in": W_in,
        "res_net": res_net,
        "alpha": alpha,
        "kb": kb,
        "beta": beta,
        "n": n,
    }

    time_infor = {
        "section_len": section_len,
        "washup_length": washup_length,
        "train_length": train_length,
        "val_length": val_length,
        "time_length": time_length,
    }

    # --------------------------------------------------
    # Generate training data
    # --------------------------------------------------
    xy, q, qdt, q2dt, tau = robot_data_generator(
        time_infor, noise_level, dt, properties
    )

    xy = xy[washup_length:, :]
    q = q[washup_length:, :]
    qdt = qdt[washup_length:, :]
    q2dt = q2dt[washup_length:, :]
    tau = tau[washup_length:, :]

    data_reservoir = {
        "xy": xy,
        "q": q,
        "qdt": qdt,
        "q2dt": q2dt,
        "tau": tau,
    }

    # --------------------------------------------------
    # Train reservoir
    # --------------------------------------------------
    t0 = time.time()
    Wout, r_end = func_reservoir_train(
        data_reservoir,
        time_infor,
        input_infor,
        res_infor,
        dim_in,
        dim_out,
    )
    t_train = time.time() - t0

    # --------------------------------------------------
    # Validation helper
    # --------------------------------------------------
    def evaluate(
        traj_type,
        r_end,
        time_infor,
        disturbance,
        measurement_noise,
    ):
        time_infor = dict(time_infor)   # copy
        time_infor["val_length"] = 150000

        start_info = {
            "q": 0,
            "qdt": 0,
            "q2dt": 0,
            "tau": 0,
        }

        control, output, time_infor, r_end = func_reservoir_validate(
            traj_type=traj_type,
            bridge_type="cubic",
            time_infor=time_infor,
            input_infor=input_infor,
            res_infor=res_infor,
            start_info=start_info,
            properties=properties,
            dim_in=dim_in,
            dim_out=dim_out,
            Wout=Wout,
            r_end=r_end,
            dt=dt,
            plot_movie=0,
            save_rend=0,
            failure={"type": "none"},
            blur={"blur": 0},
            traj_frequency=75,
            disturbance = disturbance,
            measurement_noise = measurement_noise,
        )

        data_pred = output["data_pred"]
        data_control = control["data_control"]

        rmse_start = int(round(time_infor["val_length"] * 3 / 5))
        rmse_end = time_infor["val_length"] - 100

        return func_rmse(data_pred, data_control, rmse_start, rmse_end), r_end

    # --------------------------------------------------
    # Evaluate all trajectories
    # --------------------------------------------------
    rmse_l, r_end = evaluate(
        "lorenz",
        r_end,
        time_infor,
        disturbance,
        measurement_noise,
    )
    rmse_c, r_end = evaluate("circle", r_end,
        time_infor,
        disturbance,
        measurement_noise,
    )
    rmse_m, r_end = evaluate("mg17", r_end,
        time_infor,
        disturbance,
        measurement_noise,
    )
    rmse_i, r_end = evaluate("infty", r_end,
        time_infor,
        disturbance,
        measurement_noise,
    )

    return rmse_l, rmse_c, rmse_m, rmse_i, t_train
