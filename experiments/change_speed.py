import numpy as np
import scipy.io as sio
from pathlib import Path
from datetime import datetime

from val_and_update import val_and_update
from tools.func_rmse import func_rmse


def change_speed():
    # --------------------------------------------------
    # Setup (equivalent to clear / close / clc)
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    time_today = datetime.now().strftime("%m%d%Y")

    # MATLAB:
    # traj_frequency_set = round(linspace(1, 400, 20));
    traj_frequency_set = np.round(np.linspace(1, 400, 20)).astype(int)

    iteration = 50
    val_length_all = 300_000

    disturbance = 0.0
    measurement_noise = 0.0

    plot_movie = 0
    traj_type = "infty"
    bridge_type = "cubic"

    failure = {"type": "none"}
    blur = {"blur": 0}

    rmse_start_time = round(val_length_all * 3 / 5)
    rmse_end_time = val_length_all - 100

    # IMPORTANT: correct variable name
    rmse_frequency_set = np.zeros((len(traj_frequency_set), iteration))

    # --------------------------------------------------
    # Main experiment loop
    # --------------------------------------------------
    for tfs, traj_frequency in enumerate(traj_frequency_set):

        rmse_repeat_set = np.zeros(iteration)

        for repeat_i in range(iteration):
            # MATLAB: load('./save_file/all_traj_06282022.mat')
            mat = sio.loadmat(
                project_root / "save_file" / "all_traj_06282022.mat",
                simplify_cells=True,
            )

            np.random.seed(None)  # rng('shuffle')

            save_rend = 0
            idx = 1

            time_infor = mat["time_infor"]
            time_infor["val_length"] = val_length_all

            # Run validation (Python replacement for val_and_update.m)
            save_all_traj = val_and_update(
                traj_type=traj_type,
                bridge_type=bridge_type,
                time_infor=time_infor,
                input_infor=mat["input_infor"],
                res_infor=mat["res_infor"],
                properties=mat["properties"],
                dim_in=int(mat["dim_in"]),
                dim_out=int(mat["dim_out"]),
                Wout=mat["Wout"],
                dt=float(mat["dt"]),
                plot_movie=plot_movie,
                save_rend=save_rend,
                failure=failure,
                blur=blur,
                idx=idx,
                plot_val_and_update=0,
                traj_frequency=int(traj_frequency),
            )

            output = save_all_traj[f"output_{idx}"]
            control = save_all_traj[f"control_{idx}"]

            data_pred = output["data_pred"]
            data_control = control["data_control"]

            rmse_repeat_set[repeat_i] = func_rmse(
                data_pred,
                data_control,
                rmse_start_time,
                rmse_end_time,
            )

        # MATLAB: sort(rmse_repeat_set)
        rmse_frequency_set[tfs, :] = np.sort(rmse_repeat_set)

        print(f"traj frequency: {traj_frequency}")

    # --------------------------------------------------
    # Save results (exact MATLAB structure)
    # --------------------------------------------------
    save_frequency_iter = {
        "traj_type": traj_type,
        "val_length": val_length_all,
        "traj_frequency_set": traj_frequency_set,
        "rmse_frequency_set": rmse_frequency_set,
    }

    save_dir = project_root / "save_data"
    save_dir.mkdir(exist_ok=True)

    save_path = (
        save_dir
        / f"save_frequency_iter_{time_today}_{np.random.randint(1, 1000)}.mat"
    )

    sio.savemat(save_path, {"save_frequency_iter": save_frequency_iter})


if __name__ == "__main__":
    change_speed()
