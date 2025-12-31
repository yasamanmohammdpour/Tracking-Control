import numpy as np
import scipy.io as sio
from pathlib import Path

from val_and_update import val_and_update
from tools.func_rmse import func_rmse


def uncertain_test_saferegion(traj_type="infty", time_today=None):
    # --------------------------------------------------
    # Setup paths
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    # --------------------------------------------------
    # Load pretrained data (MATLAB: load)
    # --------------------------------------------------
    mat = sio.loadmat(
        project_root / "save_file" / "all_traj_06282022.mat",
        simplify_cells=True,
    )

    plot_val_and_update = 0
    plot_movie = 0
    blur = {"blur": 0}

    bridge_type = "cubic"

    # --------------------------------------------------
    # Time configuration
    # --------------------------------------------------
    time_infor = mat["time_infor"]
    time_infor["val_length"] = 250_000
    val_length_rm = time_infor["val_length"]

    rmse_start_time = round(val_length_rm * 3 / 5)
    rmse_end_time = time_infor["val_length"] - 100

    # --------------------------------------------------
    # Trajectory frequency
    # --------------------------------------------------
    if traj_type == "circle":
        traj_frequency = 150
    elif traj_type == "infty":
        traj_frequency = 75
    else:
        traj_frequency = 75

    # --------------------------------------------------
    # Uncertainty parameters
    # --------------------------------------------------
    l1_set = np.linspace(0.5, 0.55, 10)
    l2_set = np.linspace(0.5, 0.55, 10)

    iteration = 50

    failure = {
        "type": "all",
        "amplitude": 0.1,
        "amplitude_2": 0.1,
    }

    uncertain_type = "l"
    idx = 2

    rmse_set = np.zeros((len(l1_set), len(l2_set), iteration))

    # --------------------------------------------------
    # Main experiment loop
    # --------------------------------------------------
    for l1_idx, l1 in enumerate(l1_set):
        for l2_idx, l2 in enumerate(l2_set):

            rmse_repeat_set = np.zeros(iteration)

            for repeat_i in range(iteration):
                # Reload data each iteration (MATLAB behavior)
                mat = sio.loadmat(
                    project_root / "save_file" / "all_traj_06282022.mat",
                    simplify_cells=True,
                )

                time_infor = mat["time_infor"]
                time_infor["val_length"] = val_length_rm

                properties = mat["properties"].copy()
                properties[2] = l1  # MATLAB: properties(3)
                properties[3] = l2  # MATLAB: properties(4)

                save_rend = 0

                save_all_traj = val_and_update(
                    traj_type=traj_type,
                    bridge_type=bridge_type,
                    time_infor=time_infor,
                    input_infor=mat["input_infor"],
                    res_infor=mat["res_infor"],
                    properties=properties,
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
                    traj_frequency=traj_frequency,
                )

                output = save_all_traj[f"output_{idx}"]
                control = save_all_traj[f"control_{idx}"]

                rmse_repeat_set[repeat_i] = func_rmse(
                    output["data_pred"],
                    control["data_control"],
                    rmse_start_time,
                    rmse_end_time,
                )

            rmse_set[l1_idx, l2_idx, :] = np.sort(rmse_repeat_set)

    # --------------------------------------------------
    # Save results (MATLAB structure preserved)
    # --------------------------------------------------
    save_uncertain_iter = {
        "traj_type": traj_type,
        f"type{idx}": uncertain_type,
        f"l1_set{idx}": l1_set,
        f"l2_set{idx}": l2_set,
        f"rmse_set{idx}": rmse_set,
    }

    save_dir = project_root / "save_data"
    save_dir.mkdir(exist_ok=True)

    save_path = (
        save_dir
        / f"save_uncertain_iter_{traj_type}_{time_today}_{np.random.randint(1, 1000)}.mat"
    )

    sio.savemat(save_path, {"save_uncertain_iter": save_uncertain_iter})

    # --------------------------------------------------
    # Visualization (same as MATLAB)
    # --------------------------------------------------
    rmse_set_l = rmse_set
    rmse_l_mean = np.mean(rmse_set_l, axis=2)

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(
            rmse_l_mean,
            origin="lower",
            extent=[l1_set[0], l1_set[-1], l2_set[0], l2_set[-1]],
            aspect="auto",
        )
        plt.colorbar()
        plt.title("RMSE mean (l-space)")

        l1_set_plot = (l1_set - 0.5) / 0.5
        l2_set_plot = (l2_set - 0.5) / 0.5
        rmse_l_mean_flip = np.flipud(rmse_l_mean)

        plt.figure()
        plt.imshow(
            rmse_l_mean_flip,
            origin="lower",
            extent=[
                l1_set_plot[0],
                l1_set_plot[-1],
                l2_set_plot[0],
                l2_set_plot[-1],
            ],
            aspect="auto",
        )
        plt.colorbar()
        plt.title("RMSE mean (normalized)")

        plt.show()

    except ImportError:
        pass
