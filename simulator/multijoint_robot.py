#multijoint_robot.py
import numpy as np


class MultiJointRobot:
    """
    Planar multi-joint robot with simple joint-space dynamics and
    physically motivated constraints for stable closed-loop control.
    """

    def __init__(self, n_joints, dt=0.01):
        # --------------------------------------------------
        # Basic configuration
        # --------------------------------------------------
        self.n_joints = n_joints
        self.dt = dt

        # --------------------------------------------------
        # Joint states
        # --------------------------------------------------
        self.q = np.zeros(n_joints)     # joint positions
        self.qd = np.zeros(n_joints)    # joint velocities

        # --------------------------------------------------
        # Kinematic parameters (planar serial arm)
        # --------------------------------------------------
        self.link_lengths = np.ones(n_joints) * 0.5

        # --------------------------------------------------
        # Joint-space dynamics parameters
        # --------------------------------------------------
        self.D = 0.1 * np.eye(n_joints)   # linear damping
        self.K = 0.5 * np.eye(n_joints)   # weak elastic restoring torque

        # weak nonlinear coupling between joints
        self.C = 0.05 * (np.ones((n_joints, n_joints)) - np.eye(n_joints))

        # --------------------------------------------------
        # Physical constraints (CRUCIAL for multijoint stability)
        # --------------------------------------------------
        self.tau_max = 2.0          # actuator torque limit
        self.qd_max = 5.0           # velocity limit (rad/s)

        self.q_max = np.pi / 2      # soft joint angle limit
        self.q_limit_gain = 5.0     # strength of joint limit restoring torque

    # ------------------------------------------------------
    # Forward dynamics integration
    # ------------------------------------------------------
    def step(self, tau):
        """
        Integrate joint dynamics forward by one time step.

        Parameters
        ----------
        tau : ndarray, shape (n_joints,)
            Joint torques

        Returns
        -------
        q : ndarray
            Joint positions after step
        qd : ndarray
            Joint velocities after step
        """

        # --------------------------------------------------
        # Torque saturation (actuator limits)
        # --------------------------------------------------
        tau = np.clip(tau, -self.tau_max, self.tau_max)

        # --------------------------------------------------
        # Soft joint-limit restoring torque
        # --------------------------------------------------
        q_limit_torque = -self.q_limit_gain * np.tanh(self.q / self.q_max)

        # --------------------------------------------------
        # Joint accelerations
        # --------------------------------------------------
        qdd = (
            tau
            - self.D @ self.qd
            - self.K @ np.sin(self.q)
            - self.C @ np.tanh(self.qd)
            + q_limit_torque
        )

        # --------------------------------------------------
        # Integrate dynamics
        # --------------------------------------------------
        self.qd += qdd * self.dt

        # velocity saturation
        self.qd = np.clip(self.qd, -self.qd_max, self.qd_max)

        self.q += self.qd * self.dt

        return self.q.copy(), self.qd.copy()

    # ------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------
    def end_effector(self):
        """
        Compute planar end-effector position.

        Returns
        -------
        y : ndarray, shape (2,)
            End-effector position [x, y]
        """
        x, y = 0.0, 0.0
        angle_sum = 0.0

        for i in range(self.n_joints):
            angle_sum += self.q[i]
            x += self.link_lengths[i] * np.cos(angle_sum)
            y += self.link_lengths[i] * np.sin(angle_sum)

        return np.array([x, y])

    # ------------------------------------------------------
    # Jacobian (task-space)
    # ------------------------------------------------------
    def jacobian(self):
        """
        Compute planar Jacobian for end-effector position.

        Returns
        -------
        J : ndarray, shape (2, n_joints)
            Jacobian matrix
        """
        n = self.n_joints
        J = np.zeros((2, n))

        theta = np.cumsum(self.q)
        l = self.link_lengths

        for k in range(n):
            s = 0.0
            c = 0.0
            for i in range(k, n):
                s += l[i] * np.sin(theta[i])
                c += l[i] * np.cos(theta[i])

            J[0, k] = -s
            J[1, k] = c

        return J
