import numpy as np
import scipy.io as sio
from datetime import datetime
from pathlib import Path

from val_and_update import val_and_update


def rand_traj_control():
    # --------------------------------------------------
    # Load pretrained data (MATLAB: load all_traj_06282022.mat)
    # --------------------------------------------------
    mat_path = Path("save_file/all_traj_06282022.mat")
    if not mat_path.exists():
        raise FileNotFoundError("save_file/all_traj_06282022.mat not found")

    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # Extract variables exactly as MATLAB workspace
    time_infor = mat["time_infor"].__dict__
    input_infor = list(mat["input_infor"])
    res_infor = mat["res_infor"].__dict__
    properties = mat["properties"]
    dim_in = int(mat["dim_in"])
    dim_out = int(mat["dim_out"])
    Wout = mat["Wout"]
    r_end = mat["r_end"]
    dt = float(mat["dt"])

    # --------------------------------------------------
    # Trajectory set (MATLAB cell array)
    # --------------------------------------------------
    traj_set = [
        "infty", "circle", "astroid", "fermat",
        "lissajous", "talbot", "heart", "lorenz",
        "chua", "rossler", "sprott_1", "sprott_4",
        "mg17", "mg30", "epitrochoid"
    ]

    # MATLAB: order = randperm(length(traj_set))
    order = np.random.permutation(len(traj_set))
    traj_set = [traj_set[i] for i in order]

    # --------------------------------------------------
    # Experiment parameters
    # --------------------------------------------------
    plot_val_and_update = True
    disturbance = 0.1
    measurement_noise = 0.1
    plot_movie = 0
    bridge_type = "cubic"

    failure = {"type": "none"}
    blur = {"blur": 0}

    idx = 1
    val_length_all = 150_000

    save_all_traj = {}

    # --------------------------------------------------
    # Main loop (MATLAB for ii = 1:length(traj_set))
    # --------------------------------------------------
    for ii, traj_type in enumerate(traj_set, start=1):
        np.random.seed(None)  # MATLAB: rng('shuffle')

        time_infor["val_length"] = val_length_all

        save_rend = 0 if ii == 1 else 1

        # MATLAB traj_frequency logic
        if traj_type == "lorenz":
            traj_frequency = 100
        elif traj_type == "circle":
            traj_frequency = 150
        else:
            traj_frequency = 75

        save_block, start_info, r_end = val_and_update(
            mat_data=str(mat_path),
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
            save_rend=save_rend,
            failure=failure,
            blur=blur,
            idx=idx,
            plot_val_and_update=plot_val_and_update,
            traj_frequency=traj_frequency,
        )

        save_all_traj.update(save_block)
        idx += 1

    # --------------------------------------------------
    # Optional save (commented in MATLAB)
    # --------------------------------------------------
    time_today = datetime.now().strftime("%m%d%Y")

    # Uncomment if you want to save
    # save_path = Path("save_data") / f"15traj_{time_today}_{np.random.randint(1,1000)}.mat"
    # sio.savemat(
    #     save_path,
    #     {
    #         "val_length_all": val_length_all,
    #         "save_all_traj": save_all_traj,
    #         "traj_set": np.array(traj_set, dtype=object),
    #     }
    # )

    print("Random trajectory control test completed.")


if __name__ == "__main__":
    rand_traj_control()
