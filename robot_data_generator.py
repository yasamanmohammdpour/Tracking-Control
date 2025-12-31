import numpy as np
from numpy import sin, cos, pi
from scipy.ndimage import gaussian_filter1d


def robot_data_generator(time_infor, noise_level, dt, properties):
    """
    Python translation of robot_data_generator.m

    Generates training/validation data for a 2-link robot arm
    using noisy torque inputs and forward dynamics.
    """

    # --------------------------------------------------
    # Unpack robot parameters (MATLAB matsplit)
    # --------------------------------------------------
    m1, m2, l1, l2, lc1, lc2, I1, I2 = properties

    section_len = time_infor["section_len"]
    time_length = time_infor["time_length"]

    # --------------------------------------------------
    # Generate smoothed noise (control torque)
    # --------------------------------------------------
    noise_interval = noise_level
    pert_length = time_length * 2

    pert = -noise_interval + 2 * noise_interval * np.random.rand(pert_length, 2)

    # MATLAB: pert = pert(100:end, :)
    pert = pert[99:, :]

    # MATLAB: smoothdata(..., 'gaussian', 50)
    pert = gaussian_filter1d(pert, sigma=50, axis=0)

    # --------------------------------------------------
    # Allocate arrays
    # --------------------------------------------------
    q = np.zeros((time_length, 2))
    qdt = np.zeros((time_length, 2))
    q2dt = np.zeros((time_length, 2))
    tau = np.zeros((time_length, 2))

    tau[:, :] = pert[:time_length, :]

    np.random.seed(None)

    # --------------------------------------------------
    # Forward dynamics simulation
    # --------------------------------------------------
    for t_i in range(time_length - 1):

        # Reset every section_len steps
        if t_i % section_len == 0:
            q[t_i, 0] = 2 * pi * np.random.rand()
            q[t_i, 1] = 2 * pi * np.random.rand() - pi
            qdt[t_i, :] = 0.0

        H11 = (
            m1 * lc1**2
            + I1
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(q[t_i, 1]))
            + I2
        )
        H12 = m2 * l1 * lc2 * cos(q[t_i, 1]) + m2 * lc2**2 + I2
        H21 = H12
        H22 = m2 * lc2**2 + I2
        h = m2 * l1 * lc2 * sin(q[t_i, 1])

        part_1 = (
            -h * qdt[t_i, 1] * qdt[t_i, 0]
            - h * (qdt[t_i, 0] + qdt[t_i, 1]) * qdt[t_i, 1]
        )
        part_2 = h * qdt[t_i, 0] ** 2

        denominator = H12 * H21 - H11 * H22

        q2dt[t_i, 0] = -(
            -part_1 * H22
            + H12 * part_2
            - H12 * tau[t_i, 1]
            + H22 * tau[t_i, 0]
        ) / denominator

        q2dt[t_i, 1] = -(
            part_1 * H21
            - H11 * part_2
            + H11 * tau[t_i, 1]
            - H21 * tau[t_i, 0]
        ) / denominator

        # Zero out dynamics at reset points
        if t_i % section_len == 0:
            q2dt[t_i, :] = 0.0
            tau[t_i, :] = 0.0

        # Integrate
        q[t_i + 1, :] = q[t_i, :] + qdt[t_i, :] * dt
        qdt[t_i + 1, :] = qdt[t_i, :] + q2dt[t_i, :] * dt

    # --------------------------------------------------
    # Forward kinematics
    # --------------------------------------------------
    x = l1 * cos(q[:, 0]) + l2 * cos(q[:, 0] + q[:, 1])
    y = l1 * sin(q[:, 0]) + l2 * sin(q[:, 0] + q[:, 1])

    xy = np.column_stack((x, y))

    return xy, q, qdt, q2dt, tau
