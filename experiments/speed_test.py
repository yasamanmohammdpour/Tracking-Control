import numpy as np
import scipy.io as sio
from pathlib import Path

from val_and_update import val_and_update
from tools.func_rmse import func_rmse


def speed_test(traj_type="infty", time_today=None):
    # --------------------------------------------------
    # Setup
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    # MATLAB: load('./save_file/all_traj_06282022.mat')
    mat = sio.loadmat(
        project_root / "save_file" / "all_traj_06282022.mat",
        simplify_cells=True,
    )

    plot_val_and_update = 0
    plot_movie = 0

    blur = {"blur": 0}

    # MATLAB default behavior
    if traj_type is None:
        traj_type = "infty"

    failure = {
        "type": "none",  # overwritten later in MATLAB, ends as 'none'
    }

    bridge_type = "cubic"

    time_infor = mat["time_infor"]
    time_infor["val_length"] = 250_000
    val_length_rm = time_infor["val_length"]

    rmse_start_time = int(round(val_length_rm * 3 / 5))
    rmse_end_time = time_infor["val_length"] - 100

    # --------------------------------------------------
    # Frequency set
    # --------------------------------------------------
    if traj_type == "circle":
        frequency_set = np.round(
            np.linspace(10, 500, 15)
        ).astype(int)
    elif traj_type == "infty":
        frequency_set = np.round(
            np.linspace(10, 500, 15) / 2
        ).astype(int)
    else:
        frequency_set = np.round(
            np.linspace(10, 500, 15)
        ).astype(int)

    iteration = 50
    rmse_set = np.zeros((len(frequency_set), iteration))

    idx = 1

    # --------------------------------------------------
    # Main experiment loop
    # --------------------------------------------------
    for f_idx, traj_frequency in enumerate(frequency_set):
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
                plot_val_and_update=plot_val_and_update,
                traj_frequency=int(traj_frequency),
            )

            output = save_all_traj[f"output_{idx}"]
            control = save_all_traj[f"control_{idx}"]

            rmse_repeat_set[repeat_i] = func_rmse(
                output["data_pred"],
                control["data_control"],
                rmse_start_time,
                rmse_end_time,
            )

        rmse_set[f_idx, :] = np.sort(rmse_repeat_set)

        print(
            f"Finished traj_type={traj_type}, frequency={traj_frequency}"
        )

    # --------------------------------------------------
    # Save results (MATLAB structure preserved)
    # --------------------------------------------------
    save_speed_iter = {
        "traj_type": traj_type,
        f"frequency_set{idx}": frequency_set,
        f"rmse_set{idx}": rmse_set,
    }

    save_dir = project_root / "save_data"
    save_dir.mkdir(exist_ok=True)

    save_path = (
        save_dir
        / f"save_speed_iter_{traj_type}_{time_today}_{np.random.randint(1,1000)}.mat"
    )

    sio.savemat(save_path, {"save_speed_iter": save_speed_iter})
