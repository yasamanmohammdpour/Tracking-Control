import numpy as np
import scipy.io as sio
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from func_reservoir_validate import func_reservoir_validate


def iter_noise(
    Wout,
    res_infor,
    time_infor,
    input_infor,
    properties,
    dim_in,
    dim_out,
    dt,
):
    project_root = Path(__file__).resolve().parents[1]
    save_dir = project_root / "save_data"
    save_dir.mkdir(exist_ok=True)

    time_today = datetime.now().strftime("%m%d%Y")

    # ==========================================================
    # ONLY DISTURBANCE
    # ==========================================================
    measurement_noise = 0.0
    disturbance_set = np.linspace(1, 5, 20)
    error_set = np.zeros(len(disturbance_set))

    rmse_length = int(1000 / dt)
    weight = np.arange(1, rmse_length + 1, dtype=float)
    weight /= np.sum(weight)

    for i, disturbance in enumerate(disturbance_set):
        traj_type = "infty"
        bridge_type = "cubic"
        plot_movie = 0

        time_infor["val_length"] = 120000

        control_infor, output_infor, time_infor = func_reservoir_validate(
            traj_type=traj_type,
            bridge_type=bridge_type,
            time_infor=time_infor,
            input_infor=input_infor,
            res_infor=res_infor,
            properties=properties,
            dim_in=dim_in,
            dim_out=dim_out,
            Wout=Wout,
            dt=dt,
            plot_movie=plot_movie,
            disturbance=disturbance,
            measurement_noise=measurement_noise,
        )

        data_pred = output_infor["data_pred"]
        data_control = control_infor["data_control"]

        error = np.abs(
            data_control[:rmse_length] - data_pred[:rmse_length]
        )
        error = np.sum(weight * np.mean(error ** 2, axis=1))
        error_set[i] = error

    error_set_plot = error_set.copy()
    nan_mask = np.isnan(error_set_plot)
    error_set_plot[nan_mask] = 5 * np.nanmax(error_set_plot)

    save_data = {
        "disturbance_set": disturbance_set,
        "disturbance_error_set": error_set,
        "disturbance_error_set_plot": error_set_plot,
    }

    sio.savemat(
        save_dir / f"disturbance_result_normscale_{time_today}.mat",
        {"save_data": save_data},
    )

    plt.figure()
    plt.plot(disturbance_set, error_set_plot, "o-")
    plt.xlabel("gaussian disturbance, σ")
    plt.ylabel("error")
    plt.show()

    # ==========================================================
    # ONLY MEASUREMENT NOISE
    # ==========================================================
    disturbance = 0.0
    measurement_set = np.exp(np.linspace(np.log(0.01), np.log(10), 20))
    error_set = np.zeros(len(measurement_set))

    for i, measurement_noise in enumerate(measurement_set):
        traj_type = "infty"
        bridge_type = "cubic"
        plot_movie = 0

        time_infor["val_length"] = 120000

        control_infor, output_infor, time_infor = func_reservoir_validate(
            traj_type=traj_type,
            bridge_type=bridge_type,
            time_infor=time_infor,
            input_infor=input_infor,
            res_infor=res_infor,
            properties=properties,
            dim_in=dim_in,
            dim_out=dim_out,
            Wout=Wout,
            dt=dt,
            plot_movie=plot_movie,
            disturbance=disturbance,
            measurement_noise=measurement_noise,
        )

        data_pred = output_infor["data_pred"]
        data_control = control_infor["data_control"]

        error = np.abs(
            data_control[:rmse_length] - data_pred[:rmse_length]
        )
        error = np.sum(weight * np.mean(error ** 2, axis=1))
        error_set[i] = error

    error_set_plot = error_set.copy()
    nan_mask = np.isnan(error_set_plot)
    error_set_plot[nan_mask] = 5 * np.nanmax(error_set_plot)

    save_data = {
        "measurement_set": measurement_set,
        "measurement_error_set": error_set,
        "measurement_error_set_plot": error_set_plot,
    }

    sio.savemat(
        save_dir / f"measurement_result_logscale_{time_today}.mat",
        {"save_data": save_data},
    )

    plt.figure()
    plt.semilogx(measurement_set, error_set_plot, "o-")
    plt.xlabel("gaussian measurement noise, σ")
    plt.ylabel("error")
    plt.show()

    # ==========================================================
    # DISTURBANCE + MEASUREMENT (HEATMAP)
    # ==========================================================
    disturbance_set = np.exp(np.linspace(np.log(0.01), np.log(10), 20))
    measurement_set = np.exp(np.linspace(np.log(0.01), np.log(10), 20))
    error_set = np.zeros((len(disturbance_set), len(measurement_set)))

    for i, disturbance in enumerate(disturbance_set):
        for j, measurement_noise in enumerate(measurement_set):
            traj_type = "infty"
            bridge_type = "cubic"
            plot_movie = 0

            time_infor["val_length"] = 120000

            control_infor, output_infor, time_infor = func_reservoir_validate(
                traj_type=traj_type,
                bridge_type=bridge_type,
                time_infor=time_infor,
                input_infor=input_infor,
                res_infor=res_infor,
                properties=properties,
                dim_in=dim_in,
                dim_out=dim_out,
                Wout=Wout,
                dt=dt,
                plot_movie=plot_movie,
                disturbance=disturbance,
                measurement_noise=measurement_noise,
            )

            data_pred = output_infor["data_pred"]
            data_control = control_infor["data_control"]

            error = np.abs(
                data_control[:rmse_length] - data_pred[:rmse_length]
            )
            error = np.sum(weight * np.mean(error ** 2, axis=1))
            error_set[i, j] = error

    error_set_plot = error_set.copy()
    error_set_plot[np.isnan(error_set_plot)] = 3 * np.nanmax(error_set_plot)

    save_data = {
        "disturbance_set_heat": disturbance_set,
        "measurement_set_heat": measurement_set,
        "heat_error_set": error_set,
        "heat_error_set_plot": error_set_plot,
    }

    sio.savemat(
        save_dir / f"heat_result_logscale_{time_today}.mat",
        {"save_data": save_data},
    )

    plt.figure()
    plt.contourf(disturbance_set, measurement_set, error_set_plot, 50)
    plt.xscale("log")
    plt.yscale("log")
    plt.colorbar(label="error")
    plt.xlabel("measurement noise, σ")
    plt.ylabel("disturbance noise, σ")
    plt.title("error")
    plt.show()
