import numpy as np
import matplotlib.pyplot as plt

from func_reservoir_validate import func_reservoir_validate


def val_and_update(
    mat_data,
    traj_type,
    bridge_type,
    time_infor,
    input_infor,
    res_infor,
    properties,
    dim_in,
    dim_out,
    Wout,
    dt,
    plot_movie,
    save_rend,
    failure,
    blur,
    idx,
    plot_val_and_update,
    traj_frequency=None,
):
    """
    Python translation of val_and_update.m
    """

    # -----------------------------
    # Validate phase
    # -----------------------------
    if save_rend == 0:
        start_info = {
            "q": 0,
            "qdt": 0,
            "q2dt": 0,
            "tau": 0,
        }

        r_end = np.zeros((res_infor["res_net"].shape[0], 1))
    else:
        start_info = {}
        r_end = None

    # -----------------------------
    # Trajectory frequency logic
    # -----------------------------
    if traj_frequency is None:
        if traj_type == "lorenz":
            traj_frequency = 100
        elif traj_type == "circle":   # MATLAB typo fixed: "cirlce"
            traj_frequency = 150
        else:
            traj_frequency = 75

    # -----------------------------
    # Random seed (rng('shuffle'))
    # -----------------------------
    np.random.seed(None)

    # -----------------------------
    # Reservoir validation
    # -----------------------------
    control_infor, output_infor, time_infor, r_end = func_reservoir_validate(
        traj_type=traj_type,
        bridge_type=bridge_type,
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
        plot_movie=plot_movie,
        save_rend=save_rend,
        failure=failure,
        blur=blur,
        traj_frequency=traj_frequency,
    )

    # -----------------------------
    # Update phase
    # -----------------------------
    data_pred = output_infor["data_pred"]
    q_pred = output_infor["q_pred"]
    qdt_pred = output_infor["qdt_pred"]
    q2dt_pred = output_infor["q2dt_pred"]
    tau_pred = output_infor["tau_pred"]

    q_control = control_infor["q_control"]
    qdt_control = control_infor["qdt_control"]
    q2dt_control = control_infor["q2dt_control"]
    tau_control = control_infor["tau_control"]
    data_control = control_infor["data_control"]

    val_length = time_infor["val_length"]

    start_info["q"] = q_pred[val_length - 3, :]
    start_info["qdt"] = qdt_pred[val_length - 3, :]
    start_info["q2dt"] = q2dt_pred[val_length - 3, :]
    start_info["tau"] = tau_pred[val_length - 3, :]

    save_all_traj = {}
    save_all_traj[f"control_{idx}"] = control_infor
    save_all_traj[f"output_{idx}"] = output_infor

    # -----------------------------
    # Plot trajectory
    # -----------------------------
    start_time = 0
    end_time = val_length - 100

    if plot_val_and_update:
        plt.figure()
        plt.plot(
            data_control[start_time:end_time, 0],
            data_control[start_time:end_time, 1],
            "r",
            label="desired trajectory",
        )
        plt.plot(
            data_pred[start_time:end_time, 0],
            data_pred[start_time:end_time, 1],
            "b--",
            label="pred trajectory",
        )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.axhline(0, color="black", linestyle="--")
        plt.axvline(0, color="black", linestyle="--")
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.legend()
        plt.show()

    return save_all_traj, start_info, r_end
