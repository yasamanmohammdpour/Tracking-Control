"""
Global configuration for multi-joint reservoir training.

This file centralizes all hyperparameters to ensure:
- reproducibility
- clean experiments
- easy scaling across joint counts

Change values here, not inside scripts.
"""

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
RANDOM_SEED = 42  # set to None for non-deterministic runs


# ------------------------------------------------------------
# Robot / Simulation parameters
# ------------------------------------------------------------
N_JOINTS = 6         # number of joints (change this for scaling studies)
DT = 0.01             # simulation time step
T_TRAIN = 60000       # total simulation steps for training data


# ------------------------------------------------------------
# Reservoir training parameters
# ------------------------------------------------------------
WASHOUT = 2000        # wash-up length (discard initial transient)

N_RESERVOIR = 800     # number of reservoir neurons
LEAK_RATE = 0.3       # alpha in leaky ESN
SPECTRAL_RADIUS = 0.95
INPUT_SCALE = 1.0
RIDGE_BETA = 1e-6     # ridge regularization coefficient


# ------------------------------------------------------------
# Sanity checks (fail early, not silently)
# ------------------------------------------------------------
assert N_JOINTS >= 2, "N_JOINTS must be >= 2"
assert DT > 0, "DT must be positive"
assert T_TRAIN > WASHOUT + 100, "Training length must exceed washout"
assert N_RESERVOIR > 10, "Reservoir too small to be meaningful"

# ------------------------------------------------------------
# Baseline PD controller (joint space)
# ------------------------------------------------------------
KP = 25.0
KD = 5.0
