import numpy as np
import matplotlib.pyplot as plt

from func_plot_movie import func_plot_movie


def func_plot_figures(control_infor, output_infor, val_length, plot_movie, properties=None):
    """
    Python translation of func_plot_figures.m
    """

    # -----------------------------
    # Extract data
    # -----------------------------
    data_pred = output_infor["data_pred"]
    q_pred = output_infor["q_pred"]
    qdt_pred = output_infor["qdt_pred"]
    q2dt_pred = output_infor["q2dt_pred"]
    tau_pred = output_infor["tau_pred"]

    q_control = control_infor["q_control"]
    qdt_control = control_infor["qdt_control"]
    q2dt_control = control_infor["q2dt_control"]
    tau_control = control_infor["tau_control"]
    data_control = control_infor["data_control"]

    start_time = 0
    end_time = val_length - 10_000

    # -----------------------------
    # Plot trajectory
    # -----------------------------
    plt.figure()
    plt.plot(
        data_control[start_time:end_time, 0],
        data_control[start_time:end_time, 1],
        "r",
        label="desired trajectory",
    )
    plt.plot(
        data_pred[start_time:end_time, 0],
        data_pred[start_time:end_time, 1],
        "b--",
        label="pred trajectory",
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # -----------------------------
    # Plot q (mod pi)
    # -----------------------------
    q_control_plot = np.mod(q_control, np.pi)
    q_pred_plot = np.mod(q_pred, np.pi)

    plt.figure()
    plt.plot(q_control_plot[start_time:end_time, 0], "r", label="desired")
    plt.plot(q_pred_plot[start_time:end_time, 0], "b", label="pred")
    plt.xlabel("time step")
    plt.ylabel("q(1)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(q_control_plot[start_time:end_time, 1], "r", label="desired")
    plt.plot(q_pred_plot[start_time:end_time, 1], "b", label="pred")
    plt.xlabel("time step")
    plt.ylabel("q(2)")
    plt.legend()
    plt.show()

    # -----------------------------
    # Plot dq/dt
    # -----------------------------
    plt.figure()
    plt.plot(qdt_control[start_time:end_time, 0], "r", label="desired")
    plt.plot(qdt_pred[start_time:end_time, 0], "b", label="pred")
    plt.xlabel("time step")
    plt.ylabel("dq/dt(1)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(qdt_control[start_time:end_time, 1], "r", label="desired")
    plt.plot(qdt_pred[start_time:end_time, 1], "b", label="pred")
    plt.xlabel("time step")
    plt.ylabel("dq/dt(2)")
    plt.legend()
    plt.show()

    # -----------------------------
    # Plot d2q/dt2 and tau (prediction only)
    # -----------------------------
    remove_transient = 1000

    plt.figure()
    plt.plot(
        q2dt_pred[start_time + remove_transient : end_time, 0],
        "b",
        label="pred",
    )
    plt.xlabel("time step")
    plt.ylabel("d2q/dt2(1)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(
        q2dt_pred[start_time + remove_transient : end_time, 1],
        "b",
        label="pred",
    )
    plt.xlabel("time step")
    plt.ylabel("d2q/dt2(2)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(
        tau_pred[start_time + remove_transient : end_time, 0],
        "b",
        label="pred",
    )
    plt.xlabel("time step")
    plt.ylabel("tau(1)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(
        tau_pred[start_time + remove_transient : end_time, 1],
        "b",
        label="pred",
    )
    plt.xlabel("time step")
    plt.ylabel("tau(2)")
    plt.legend()
    plt.show()

    # -----------------------------
    # Optional movie plot
    # -----------------------------
    if plot_movie == 1 and properties is not None:
        start_step = start_time
        movie_step = 500
        time_all = end_time
        line_property = "dotted"

        q1 = q_pred[:, 0]
        q2 = q_pred[:, 1]

        func_plot_movie(
            start_step,
            movie_step,
            time_all,
            q1,
            q2,
            properties,
            line_property,
        )
