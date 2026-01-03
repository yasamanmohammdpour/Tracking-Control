import os
import json
import numpy as np
import sys

# ------------------------------------------------------------
# Add project root to PYTHONPATH
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulator.data_generator import generate_multijoint_data
from learning.reservoir import Reservoir
from learning.training_data import build_training_data
from learning.config import (
    N_JOINTS,
    DT,
    T_TRAIN,
    WASHOUT,
    N_RESERVOIR,
    LEAK_RATE,
    SPECTRAL_RADIUS,
    RIDGE_BETA,
    INPUT_SCALE,
    RANDOM_SEED,
)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    # ------------------------------------------------------------
    # 1. Reproducibility
    # ------------------------------------------------------------
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    # ------------------------------------------------------------
    # 2. Generate multi-joint simulation data
    # ------------------------------------------------------------
    print(f"[INFO] Generating training data for {N_JOINTS}-joint system")

    data = generate_multijoint_data(
        n_joints=N_JOINTS,
        T=T_TRAIN,
        dt=DT,
    )

    # ------------------------------------------------------------
    # 3. Build reservoir training matrices
    # ------------------------------------------------------------
    print("[INFO] Building reservoir training matrices")

    U, Y = build_training_data(data)

    dim_in = U.shape[1]
    dim_out = Y.shape[1]

    print(f"[INFO] Reservoir input dimension  : {dim_in}")
    print(f"[INFO] Reservoir output dimension : {dim_out}")

    # ------------------------------------------------------------
    # 4. Initialize reservoir
    # ------------------------------------------------------------
    print("[INFO] Initializing reservoir")

    reservoir = Reservoir(
        n_reservoir=N_RESERVOIR,
        dim_in=dim_in,
        dim_out=dim_out,
        spectral_radius=SPECTRAL_RADIUS,
        input_scale=INPUT_SCALE,
        leak_rate=LEAK_RATE,
        ridge_beta=RIDGE_BETA,
        seed=RANDOM_SEED,
    )

    # ------------------------------------------------------------
    # 5. Train reservoir
    # ------------------------------------------------------------
    print("[INFO] Training reservoir")

    reservoir.train(
        U=U,
        Y=Y,
        washout=WASHOUT,
    )

    # ------------------------------------------------------------
    # 6. Save trained model
    # ------------------------------------------------------------
    ensure_dir("save_file")

    model_path = f"save_file/reservoir_n{N_JOINTS}.npz"
    meta_path = f"save_file/reservoir_n{N_JOINTS}_meta.json"

    np.savez(
        model_path,
        W_out=reservoir.W_out,
        r_end=reservoir.r_end,
        W=reservoir.W,
        W_in=reservoir.W_in,
        bias=reservoir.bias,
    )

    metadata = {
        "n_joints": N_JOINTS,
        "dt": DT,
        "train_length": T_TRAIN,
        "washout": WASHOUT,
        "n_reservoir": N_RESERVOIR,
        "leak_rate": LEAK_RATE,
        "spectral_radius": SPECTRAL_RADIUS,
        "ridge_beta": RIDGE_BETA,
        "input_scale": INPUT_SCALE,
        "random_seed": RANDOM_SEED,
        "dim_in": dim_in,
        "dim_out": dim_out,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Training complete")
    print(f"[INFO] Model saved to {model_path}")
    print(f"[INFO] Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
