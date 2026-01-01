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

# print(mat_data.keys())

# -----------------------------
# Choose reference trajectory
# -----------------------------
# traj_type = "circle"
# traj_type = "lorenz"
# traj_type = "mg17"
traj_type = "infty"
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

# -----------------------------
# Extract MATLAB variables
# -----------------------------
input_infor = [s[0] for s in mat_data["input_infor"].squeeze()]

res_raw = mat_data["res_infor"]
res_infor = {
    "W_in": res_raw["W_in"][0, 0],
    "res_net": res_raw["res_net"][0, 0],
    "alpha": float(res_raw["alpha"][0, 0]),
    "kb": float(res_raw["kb"][0, 0]),
    "beta": float(res_raw["beta"][0, 0]),
    "n": int(res_raw["n"][0, 0]),
}

properties = mat_data["properties"].squeeze()
dim_in = int(mat_data["dim_in"].squeeze())
dim_out = int(mat_data["dim_out"].squeeze())
Wout = mat_data["Wout"]
dt = float(mat_data["dt"].squeeze())

# time_infor from MAT file, override val_length only
time_raw = mat_data["time_infor"]
time_infor = {}

for field in time_raw.dtype.names:
    time_infor[field] = int(time_raw[field][0, 0])

time_infor["val_length"] = 200_000



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
    disturbance=disturbance,
    measurement_noise=measurement_noise,
    plot_movie=plot_movie,
    save_rend=save_rend,
    failure=failure,
    blur=blur,
    idx=idx,
    plot_val_and_update=plot_val_and_update,
)

