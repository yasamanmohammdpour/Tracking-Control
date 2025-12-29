# main.py

import numpy as np
import scipy.io as sio
import sys
from pathlib import Path

# -----------------------------
# Path setup (MATLAB addpath)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
TOOLS_DIR = PROJECT_ROOT / "tools"

sys.path.append(str(TOOLS_DIR))

# -----------------------------
# Imports from tools
# -----------------------------
from val_and_update import val_and_update

# -----------------------------
# Clear / reset equivalents
# -----------------------------
# MATLAB:
# clear all; close all; clc
#
# Python:
# - No workspace to clear
# - No figures open unless you opened them
# - No command window to clear
#
# We do nothing here on purpose.

# -----------------------------
# Load data
# -----------------------------
data_path = PROJECT_ROOT / "save_file" / "all_traj_06282022.mat"
mat_data = sio.loadmat(data_path)

# NOTE:
# MATLAB loads variables directly into workspace.
# Python loads a dict. We pass what we need explicitly.

# -----------------------------
# Choose reference trajectory
# -----------------------------
traj_type = "circle"
# traj_type = "lorenz"
# traj_type = "mg17"
# traj_type = "infty"
# traj_type = "fermat"
# traj_type = "astroid"
# traj_type = "heart"
# traj_type = "epitrochoid"
# traj_type = "lissajous"
# traj_type = "talbot"
# traj_type = "chua"
# traj_type = "rossler"
# traj_type = "sprott_1"
# traj_type = "sprott_4"
# traj_type = "mg30"
# traj_type = "lorenz96"

# -----------------------------
# Parameters
# -----------------------------
time_infor = {
    "val_length": 200_000
}

bridge_type = "cubic"

failure = {
    "type": "none"
}

disturbance = 0.0
measurement_noise = 0.0

plot_movie = False

blur = {
    "blur": 0
}

save_rend = False
idx = 0

plot_val_and_update = True

# -----------------------------
# Run validation + update
# -----------------------------
val_and_update(
    mat_data=mat_data,
    traj_type=traj_type,
    time_infor=time_infor,
    bridge_type=bridge_type,
    failure=failure,
    disturbance=disturbance,
    measurement_noise=measurement_noise,
    plot_movie=plot_movie,
    blur=blur,
    save_rend=save_rend,
    idx=idx,
    plot_val_and_update=plot_val_and_update,
)
