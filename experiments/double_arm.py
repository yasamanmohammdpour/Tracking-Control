import numpy as np
from scipy.sparse import random as sprandsym
from scipy.sparse.linalg import eigs
from pathlib import Path

from robot_data_generator import robot_data_generator
from func_reservoir_train import func_reservoir_train
from tools.func_rmse import func_rmse
from val_and_update import val_and_update


def func_double_arm():
    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------
    dt = 0.01
    input_infor = ["xy", "qdt"]

    dim_in = len(input_infor) * 4
    dim_out = 2

    # Double-arm physical properties
    properties = np.array([
        1.0, 1.0,     # m1, m2
        0.5, 0.5,     # l1, l2
        0.25, 0.25,   # lc1, lc2
        0.03, 0.03    # I1, I2
    ])

    reset_t = 80
    train_t = 200_000
    val_t = 500
    noise_level = 2.0e-2

    disturbance = 0.0
    measurement_noise = 0.0

    n = 200
    hyperpara_set = [0.756250, 0.756250, 0.843750, -3.125, 106.71875, 2.0]

    eig_rho, W_in_a, alpha, beta_log, k_raw, kb = hyperpara_set
    beta = 10 ** beta_log
    k = round(k_raw / 200 * n)

    # --------------------------------------------------
    # Reservoir construction
    # --------------------------------------------------
    W_in = W_in_a * (2 * np.random.rand(n, dim_in) - 1)

    res_net = sprandsym(n, k / n, data_rvs=np.random.randn)
    eig_D = eigs(res_net, k=1, return_eigenvectors=False)[0]
    res_net = (eig_rho / abs(eig_D)) * res_net
    res_net = res_net.toarray()

    section_len = round(reset_t / dt)
    washup_length = round(1002 / dt)
    train_length = round(train_t / dt) + round(5 / dt)
    val_length = round(val_t / dt)
    time_length = train_length + 2 * val_length + 3 * washup_length + 100

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
    # Data generation
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
    # Training
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
    # Load pretrained trajectory data
    # --------------------------------------------------
    save_path = Path("save_file/all_traj_06282022.mat")
    if not save_path.exists():
        raise FileNotFoundError("Missing all_traj_06282022.mat")

    # --------------------------------------------------
    # Validation runs
    # --------------------------------------------------
    failure = {"type": "none"}
    blur = {"blur": 0}

    plot_val_and_update = True
    plot_movie = 0

    rmse_list = []

    traj_configs = [
        ("lorenz", None),
        ("circle", 150),
        ("mg17", None),
        ("infty", 100),
    ]

    for idx, (traj_type, traj_frequency) in enumerate(traj_configs, start=1):
        time_infor["val_length"] = 250_000

        save_all_traj, start_info, r_end = val_and_update(
            mat_data=str(save_path),
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
            plot_movie=plot_movie,
            save_rend=idx > 1,
            failure=failure,
            blur=blur,
            idx=idx,
            plot_val_and_update=plot_val_and_update,
            traj_frequency=traj_frequency,
        )

        data_pred = save_all_traj[f"output_{idx}"]["data_pred"]
        data_control = save_all_traj[f"control_{idx}"]["data_control"]

        rmse_length = int(1000 / dt)
        rmse = func_rmse(
            data_pred,
            data_control,
            0,
            rmse_length,
        )
        rmse_list.append(rmse)

    avg_rmse = np.mean(rmse_list)
    print(f"Average RMSE over trajectories: {avg_rmse:.6f}")

    return avg_rmse
