import numpy as np
import scipy.io as sio
from datetime import datetime
from pathlib import Path

from robot_data_generator import robot_data_generator
from func_reservoir_train import func_reservoir_train
from func_reservoir_validate import func_reservoir_validate
from tools.func_rmse import func_rmse
from val_and_update import val_and_update


def func_double_arm():
    # --------------------------------------------------
    # RNG + time
    # --------------------------------------------------
    np.random.seed(None)
    time_today = datetime.now().strftime("%m%d%Y")

    # --------------------------------------------------
    # Basic setup
    # --------------------------------------------------
    dt = 0.01
    input_infor = ["xy", "qdt"]

    dim_in = len(input_infor) * 4
    dim_out = 2

    # --------------------------------------------------
    # Robot properties
    # --------------------------------------------------
    m1 = m2 = 1.0
    l1 = l2 = 0.5
    lc1 = lc2 = 0.25
    I1 = I2 = 0.03

    properties = np.array([m1, m2, l1, l2, lc1, lc2, I1, I2])

    # --------------------------------------------------
    # Time parameters
    # --------------------------------------------------
    reset_t = 80
    train_t = 200_000
    val_t = 500

    noise_level = 2.0e-2
    disturbance = 0.0
    measurement_noise = 0.0

    # --------------------------------------------------
    # Reservoir hyperparameters
    # --------------------------------------------------
    n = 200
    hyperpara_set = [0.756250, 0.756250, 0.843750, -3.125, 106.71875, 2.0]

    eig_rho = hyperpara_set[0]
    W_in_a = hyperpara_set[1]
    alpha = hyperpara_set[2]
    beta = 10 ** hyperpara_set[3]
    k = round(hyperpara_set[4] / 200 * n)
    kb = hyperpara_set[5]

    # --------------------------------------------------
    # Reservoir construction
    # --------------------------------------------------
    W_in = W_in_a * (2 * np.random.rand(n, dim_in) - 1)

    # symmetric sparse matrix → dense
    res_net = np.random.randn(n, n)
    eig_D = np.linalg.eigvals(res_net)[0]
    res_net = (eig_rho / abs(eig_D)) * res_net

    # --------------------------------------------------
    # Time info
    # --------------------------------------------------
    section_len = round(reset_t / dt)
    washup_length = round(1002 / dt)
    train_length = round(train_t / dt) + round(5 / dt)
    val_length = round(val_t / dt)

    time_length = train_length + 2 * val_length + 3 * washup_length + 100

    time_infor = {
        "section_len": section_len,
        "washup_length": washup_length,
        "train_length": train_length,
        "val_length": val_length,
        "time_length": time_length,
    }

    res_infor = {
        "W_in": W_in,
        "res_net": res_net,
        "alpha": alpha,
        "kb": kb,
        "beta": beta,
        "n": n,
    }

    # --------------------------------------------------
    # Generate data
    # --------------------------------------------------
    xy, q, qdt, q2dt, tau = robot_data_generator(
        time_infor, noise_level, dt, properties
    )

    xy = xy[washup_length:]
    q = q[washup_length:]
    qdt = qdt[washup_length:]
    q2dt = q2dt[washup_length:]
    tau = tau[washup_length:]

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
    Wout, r_end = func_reservoir_train(
        data_reservoir,
        time_infor,
        input_infor,
        res_infor,
        dim_in,
        dim_out,
    )

    # --------------------------------------------------
    # Validation runs
    # --------------------------------------------------
    failure = {"type": "none"}
    blur = {"blur": 0}

    rmse_start_time = 200_000
    rmse_end_time = 300_000 - 100

    results = []

    for idx, traj_type in enumerate(["lorenz", "circle", "mg17", "infty"], start=1):
        time_infor["val_length"] = 300_000

        save_all_traj, _, _ = val_and_update(
            mat_data=None,
            traj_type=traj_type,
            bridge_type="cubic",
            time_infor=time_infor,
            input_infor=input_infor,
            res_infor=res_infor,
            properties=properties,
            dim_in=dim_in,
            dim_out=dim_out,
            Wout=Wout,
            dt=dt,
            plot_movie=0,
            save_rend=idx > 1,
            failure=failure,
            blur=blur,
            idx=idx,
            plot_val_and_update=0,
        )

        data_pred = save_all_traj[f"output_{idx}"]["data_pred"]
        data_control = save_all_traj[f"control_{idx}"]["data_control"]

        rmse = func_rmse(
            data_pred, data_control, rmse_start_time, rmse_end_time
        )
        results.append(rmse)

    a_rmse = np.mean(results)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    save_dir = Path("choose_file")
    save_dir.mkdir(exist_ok=True)

    save_path = (
        save_dir
        / f"all_traj_{time_today}_{np.random.randint(10000)}_{np.random.randint(10000)}.mat"
    )

    sio.savemat(
        save_path,
        {
            "time_infor": time_infor,
            "input_infor": input_infor,
            "res_infor": res_infor,
            "properties": properties,
            "dim_in": dim_in,
            "dim_out": dim_out,
            "Wout": Wout,
            "r_end": r_end,
            "dt": dt,
            "reset_t": reset_t,
            "noise_level": noise_level,
            "a_rmse": a_rmse,
        },
    )
