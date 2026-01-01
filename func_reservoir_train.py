# func_reservoir_train.py

import numpy as np
from numpy import tanh


def func_reservoir_train(
    data_reservoir,
    time_infor,
    input_infor,
    res_infor,
    dim_in,
    dim_out,
):
    """
    Python translation of func_reservoir_train.m
    Trains Wout using ridge regression.
    """

    # --------------------------------------------------
    # Random seed (rng('shuffle'))
    # --------------------------------------------------
    np.random.seed(None)

    # --------------------------------------------------
    # Unpack training data
    # --------------------------------------------------
    xy = data_reservoir["xy"]
    q = data_reservoir["q"]
    qdt = data_reservoir["qdt"]
    q2dt = data_reservoir["q2dt"]
    tau = data_reservoir["tau"]

    washup_length = time_infor["washup_length"]
    train_length = time_infor["train_length"]

    # --------------------------------------------------
    # Reservoir parameters
    # --------------------------------------------------
    W_in = res_infor["W_in"]
    res_net = res_infor["res_net"]
    alpha = res_infor["alpha"]
    kb = res_infor["kb"]
    beta = res_infor["beta"]
    n = res_infor["n"]

    # --------------------------------------------------
    # Allocate training matrices
    # --------------------------------------------------
    train_x = np.zeros((train_length, dim_in))
    train_y = np.zeros((train_length, dim_out))

    # --------------------------------------------------
    # Build input features (exact MATLAB logic)
    # --------------------------------------------------
    if (
        len(input_infor) == 2
        and input_infor[0] == "xy"
        and input_infor[1] == "qdt"
    ):
        train_x[:] = np.hstack(
            [
                xy[:train_length, :],
                xy[1 : train_length + 1, :],
                qdt[:train_length, :],
                qdt[1 : train_length + 1, :],
            ]
        )

    elif len(input_infor) == 1 and input_infor[0] == "q":
        train_x[:] = np.hstack(
            [
                q[:train_length, :],
                q[1 : train_length + 1, :],
            ]
        )

    elif (
        len(input_infor) == 2
        and input_infor[0] == "q"
        and input_infor[1] == "qdt"
    ):
        train_x[:] = np.hstack(
            [
                q[:train_length, :],
                q[1 : train_length + 1, :],
                qdt[:train_length, :],
                qdt[1 : train_length + 1, :],
            ]
        )

    elif (
        len(input_infor) == 3
        and input_infor[0] == "xy"
        and input_infor[1] == "qdt"
        and input_infor[2] == "q2dt"
    ):
        train_x[:] = np.hstack(
            [
                q[:train_length, :],
                q[1 : train_length + 1, :],
                qdt[:train_length, :],
                qdt[1 : train_length + 1, :],
                q2dt[:train_length, :],
                q2dt[1 : train_length + 1, :],
            ]
        )

    else:
        raise ValueError("Unsupported input_infor configuration")

    train_y[:] = tau[:train_length, :]

    # MATLAB transposes here
    train_x = train_x.T      # dim_in x T
    train_y = train_y.T      # dim_out x T

    # --------------------------------------------------
    # Reservoir forward pass
    # --------------------------------------------------
    r_all = np.zeros((n, train_length + 1))

    for ti in range(train_length):
        r_all[:, ti + 1] = (
            (1 - alpha) * r_all[:, ti]
            + alpha * tanh(
                res_net @ r_all[:, ti]
                + W_in @ train_x[:, ti]
                + kb * np.ones(n)
            )
        )

    # --------------------------------------------------
    # Collect states after washup
    # --------------------------------------------------
    r_out = r_all[:, washup_length + 1 :]

    # Square even indices (MATLAB: 2:2:end)
    r_out[1::2, :] = r_out[1::2, :] ** 2

    r_end = r_all[:, -1].reshape(-1, 1)

    r_train = r_out
    y_train = train_y[:, washup_length:]

    # --------------------------------------------------
    # Ridge regression (closed form)
    # --------------------------------------------------
    # Wout = Y R^T (R R^T + beta I)^(-1)
    RRt = r_train @ r_train.T
    Wout = y_train @ r_train.T @ np.linalg.inv(RRt + beta * np.eye(n))

    return Wout, r_end
