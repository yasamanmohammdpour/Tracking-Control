import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path


def func_plot_movie(
    start_step,
    movie_step,
    time_all,
    q1,
    q2,
    properties,
    line_prop,
):
    """
    Python translation of func_plot_movie.m
    Creates an animated GIF of the double-arm motion.
    """

    # -----------------------------
    # Robot properties
    # -----------------------------
    l1 = properties[2]  # MATLAB index 3
    l2 = properties[3]  # MATLAB index 4

    # End-effector trajectory
    value_x_test = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    value_y_test = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

    # First arm endpoint
    value_x1 = l1 * np.cos(q1)
    value_y1 = l1 * np.sin(q1)

    # -----------------------------
    # Line style
    # -----------------------------
    if line_prop == "solid":
        linestyle = "-"
    elif line_prop == "dotted":
        linestyle = "--"
    else:
        raise ValueError("line_prop must be 'solid' or 'dotted'")

    # -----------------------------
    # Output GIF
    # -----------------------------
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    filename = results_dir / "double_arm.gif"

    frames = []

    trace = 1000

    # -----------------------------
    # Animation loop
    # -----------------------------
    for i in range(start_step, time_all, movie_step):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])

        # Plot trajectory trace
        if i > trace * movie_step:
            start_idx = i - trace * movie_step
        else:
            start_idx = 0

        ax.plot(
            value_x_test[start_idx:i],
            value_y_test[start_idx:i],
            color="blue",
            linestyle=linestyle,
            linewidth=1,
        )

        # Plot arms
        ax.plot(
            [0, value_x1[i]],
            [0, value_y1[i]],
            linewidth=3,
        )
        ax.plot(
            [value_x1[i], value_x_test[i]],
            [value_y1[i], value_y_test[i]],
            linewidth=3,
        )

        # Axes lines
        ax.axhline(0, color="black", linestyle="--")
        ax.axvline(0, color="black", linestyle="--")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"time = {i * 0.01:.2f}s")

        # Render frame
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close(fig)

    # -----------------------------
    # Write GIF
    # -----------------------------
    imageio.mimsave(
        filename,
        frames,
        fps=10,
        loop=0,
    )
