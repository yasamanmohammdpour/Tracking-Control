import numpy as np
import scipy.io as sio
from pathlib import Path
from datetime import datetime

from func_train_val import func_train_val


def success_rate():
    # --------------------------------------------------
    # Setup (MATLAB: clear all; close all; clc)
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    # MATLAB:
    # network_set = round(exp(linspace(log(20), log(250), 10)));
    network_set = np.round(
        np.exp(np.linspace(np.log(20), np.log(250), 10))
    ).astype(int)

    # MATLAB:
    # training_length_set = round(exp(linspace(log(2000), log(150000), 10)));
    training_length_set = np.round(
        np.exp(np.linspace(np.log(2000), np.log(150000), 10))
    ).astype(int)

    # MATLAB:
    # reset_t_set = round(linspace(10, 150, 10));
    reset_t_set = np.round(np.linspace(10, 150, 10)).astype(int)

    # MATLAB:
    # noise_level_set = 10 .^ linspace(-3, 0, 10);
    noise_level_set = 10 ** np.linspace(-3, 0, 10)

    time_today = datetime.now().strftime("%m%d%Y")

    # --------------------------------------------------
    # Fixed test parameters (first experiment)
    # --------------------------------------------------
    reset_t = 80
    noise_level = 2.0e-2
    bias = 2.0

    iteration = 50

    # --------------------------------------------------
    # Allocate result arrays
    # --------------------------------------------------
    rmse_set_lorenz = np.zeros(
        (len(network_set), len(training_length_set), iteration)
    )
    rmse_set_circle = np.zeros_like(rmse_set_lorenz)
    rmse_set_mg17 = np.zeros_like(rmse_set_lorenz)
    rmse_set_infty = np.zeros_like(rmse_set_lorenz)

    time_set = np.zeros_like(rmse_set_lorenz)

    # --------------------------------------------------
    # Main experiment loop
    # --------------------------------------------------
    for ns, n in enumerate(network_set):
        for tls, train_t in enumerate(training_length_set):

            rmse_parfor_l = np.zeros(iteration)
            rmse_parfor_c = np.zeros(iteration)
            rmse_parfor_m = np.zeros(iteration)
            rmse_parfor_i = np.zeros(iteration)
            time_parfor = np.zeros(iteration)

            for repeat_i in range(iteration):
                (
                    rmse_l,
                    rmse_c,
                    rmse_m,
                    rmse_i,
                    t_repeat_i,
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
                time_parfor[repeat_i] = t_repeat_i

            rmse_set_lorenz[ns, tls, :] = rmse_parfor_l
            rmse_set_circle[ns, tls, :] = rmse_parfor_c
            rmse_set_mg17[ns, tls, :] = rmse_parfor_m
            rmse_set_infty[ns, tls, :] = rmse_parfor_i
            time_set[ns, tls, :] = time_parfor

            print(
                f"Finished n={n}, train_length={train_t}"
            )

    # --------------------------------------------------
    # Save results (exact MATLAB structure)
    # --------------------------------------------------
    save_success_rate = {
        "reset_t": reset_t,
        "noise_level": noise_level,
        "network_set": network_set,
        "training_length_set": training_length_set,
        "rmse_set_lorenz": rmse_set_lorenz,
        "rmse_set_circle": rmse_set_circle,
        "rmse_set_mg17": rmse_set_mg17,
        "rmse_set_infty": rmse_set_infty,
        "time_set": time_set,
    }

    save_dir = project_root / "save_data"
    save_dir.mkdir(exist_ok=True)

    save_path = (
        save_dir
        / f"save_success_rate_nt_{time_today}_{np.random.randint(1,1000)}.mat"
    )

    sio.savemat(save_path, {"save_success_rate": save_success_rate})


if __name__ == "__main__":
    success_rate()
