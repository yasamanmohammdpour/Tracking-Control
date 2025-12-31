import numpy as np
import scipy.io as sio
from pathlib import Path
from datetime import datetime

from val_and_update import val_and_update
from tools.func_rmse import func_rmse


def normal_success_rate():
    # --------------------------------------------------
    # Setup (MATLAB: clear all; close all; clc)
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    # MATLAB: load('./save_file/all_traj_06282022.mat')
    mat = sio.loadmat(
        project_root / "save_file" / "all_traj_06282022.mat",
        simplify_cells=True,
    )

    time_today = datetime.now().strftime("%m%d%Y")

    plot_movie = 0

    # MATLAB:
    # if exist('traj_type','var') == 0
    #     traj_type = 'infty';
    if "traj_type" not in locals():
        traj_type = "infty"

    traj_set = [
        "lorenz", "circle", "mg17", "infty", "astroid", "fermat",
        "lissajous", "talbot", "heart", "chua", "rossler",
        "sprott_1", "sprott_4", "mg30", "epitrochoid",
    ]

    iteration = 100

    bridge_type = "cubic"

    time_infor = mat["time_infor"]
    time_infor["val_length"] = 250_000
    val_length_rm = time_infor["val_length"]

    # Failure / blur configuration
    blur = {
        "last_time": 1000,
        "recover_time": time_infor["val_length"],
        "blur": 0,
    }

    failure = {
        "type": "all",
        "amplitude": 0.1,
        "amplitude_2": 0.1,
    }

    rmse_start_time = int(round(val_length_rm * 3 / 5))
    rmse_end_time = time_infor["val_length"] - 100

    rmse_set = np.zeros((len(traj_set), iteration))

    idx = 1

    # --------------------------------------------------
    # Main experiment loop
    # --------------------------------------------------
    for traj_id, traj_type in enumerate(traj_set):

        if traj_type == "circle":
            traj_frequency = 150
        elif traj_type == "infty":
            traj_frequency = 75
        else:
            traj_frequency = 75

        rmse_repeat_set = np.zeros(iteration)

        for repeat_i in range(iteration):
            mat = sio.loadmat(
                project_root / "save_file" / "all_traj_06282022.mat",
                simplify_cells=True,
            )

            time_infor = mat["time_infor"]
            time_infor["val_length"] = val_length_rm

            save_rend = 0

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
                traj_frequency=traj_frequency,
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

        rmse_set[traj_id, :] = np.sort(rmse_repeat_set)

        print(f"Finished traj_type={traj_type}")

    # --------------------------------------------------
    # Save results (MATLAB structure)
    # --------------------------------------------------
    save_success = {
        "rmse_set": rmse_set,
        "val_length": time_infor["val_length"],
        "traj_set": traj_set,
    }

    # (MATLAB save commented out in original, so we keep it optional)
    # save_dir = project_root / "save_data"
    # save_dir.mkdir(exist_ok=True)
    # sio.savemat(
    #     save_dir / f"save_normal_success_rate_{time_today}_{np.random.randint(1,1000)}.mat",
    #     {"save_success": save_success},
    # )

    # --------------------------------------------------
    # Post-processing (success logic)
    # --------------------------------------------------
    rmse_threshold = 0.18

    rmse_logic = (rmse_set <= rmse_threshold).astype(int)
    rmse_count = rmse_logic.mean(axis=1)

    # MATLAB:
    # figure(); plot(rmse_count, 'o')
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(rmse_count, "o")
        plt.title("Normal Success Rate")
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    normal_success_rate()
