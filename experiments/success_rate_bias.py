import numpy as np
import scipy.io as sio
from pathlib import Path
from datetime import datetime

from func_train_val import func_train_val


def success_rate_bias():
    # --------------------------------------------------
    # Setup (MATLAB: clear all; close all; clc)
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    # MATLAB:
    # bias_set = linspace(0, 3, 7);
    bias_set = np.linspace(0, 3, 7)

    time_today = datetime.now().strftime("%m%d%Y")

    # --------------------------------------------------
    # Fixed parameters
    # --------------------------------------------------
    reset_t = 80
    noise_level = 2.0e-2
    n = 200
    train_t = 150000

    iteration = 10

    # --------------------------------------------------
    # Allocate result arrays
    # --------------------------------------------------
    rmse_set_lorenz = np.zeros((len(bias_set), iteration))
    rmse_set_circle = np.zeros_like(rmse_set_lorenz)
    rmse_set_mg17 = np.zeros_like(rmse_set_lorenz)
    rmse_set_infty = np.zeros_like(rmse_set_lorenz)

    # --------------------------------------------------
    # Main experiment loop
    # --------------------------------------------------
    for bs, bias in enumerate(bias_set):

        rmse_parfor_l = np.zeros(iteration)
        rmse_parfor_c = np.zeros(iteration)
        rmse_parfor_m = np.zeros(iteration)
        rmse_parfor_i = np.zeros(iteration)

        for repeat_i in range(iteration):
            (
                rmse_l,
                rmse_c,
                rmse_m,
                rmse_i,
                _,
            ) = func_train_val(
                n=n,
                train_t=train_t,
                reset_t=reset_t,
                noise_level=noise_level,
                bias=bias,
            )

            rmse_parfor_l[repeat_i] = rmse_l
            rmse_parfor_c[repeat_i] = rmse_c
            rmse_parfor_m[repeat_i] = rmse_m
            rmse_parfor_i[repeat_i] = rmse_i

        rmse_set_lorenz[bs, :] = rmse_parfor_l
        rmse_set_circle[bs, :] = rmse_parfor_c
        rmse_set_mg17[bs, :] = rmse_parfor_m
        rmse_set_infty[bs, :] = rmse_parfor_i

        print(f"Finished bias={bias:.3f}")

    # --------------------------------------------------
    # Save results (exact MATLAB structure)
    # --------------------------------------------------
    save_success_rate = {
        "reset_t": reset_t,
        "noise_level": noise_level,
        "n": n,
        "train_t": train_t,
        "bias_set": bias_set,
        "rmse_set_lorenz": rmse_set_lorenz,
        "rmse_set_circle": rmse_set_circle,
        "rmse_set_mg17": rmse_set_mg17,
        "rmse_set_infty": rmse_set_infty,
    }

    save_dir = project_root / "save_data"
    save_dir.mkdir(exist_ok=True)

    save_path = (
        save_dir
        / f"save_success_rate_bias_{time_today}_{np.random.randint(1,1000)}.mat"
    )

    sio.savemat(save_path, {"save_success_rate": save_success_rate})


if __name__ == "__main__":
    success_rate_bias()
