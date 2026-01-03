import numpy as np


class Reservoir:
    """
    Echo State Network (Reservoir Computing) for inverse control learning.

    This implementation faithfully mirrors the MATLAB training logic:
    - Leaky reservoir update
    - tanh nonlinearity
    - Squared expansion on half of reservoir units
    - Ridge regression output layer

    The reservoir is fully dimension-agnostic and supports multi-joint systems.
    """

    def __init__(
        self,
        n_reservoir: int,
        dim_in: int,
        dim_out: int,
        spectral_radius: float = 0.95,
        input_scale: float = 1.0,
        leak_rate: float = 0.3,
        ridge_beta: float = 1e-6,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        n_reservoir : int
            Number of reservoir neurons.
        dim_in : int
            Dimension of reservoir input.
        dim_out : int
            Dimension of reservoir output (number of joints).
        spectral_radius : float
            Desired spectral radius of reservoir weight matrix.
        input_scale : float
            Scaling factor for input weights.
        leak_rate : float
            Leaky integration rate (alpha).
        ridge_beta : float
            Ridge regularization coefficient.
        seed : int or None
            Random seed for reproducibility.
        """

        if seed is not None:
            np.random.seed(seed)

        self.n = n_reservoir
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.alpha = leak_rate
        self.beta = ridge_beta

        # Input weight matrix
        self.W_in = input_scale * (2.0 * np.random.rand(n_reservoir, dim_in) - 1.0)

        # Reservoir weight matrix
        W = 2.0 * np.random.rand(n_reservoir, n_reservoir) - 1.0
        eigvals = np.linalg.eigvals(W)
        W /= np.max(np.abs(eigvals)) / spectral_radius
        self.W = W

        # Bias term (equivalent to kb * ones(n,1) in MATLAB)
        self.bias = np.ones(n_reservoir)

        # Output weights (learned during training)
        self.W_out = None

        # Final reservoir state (used for continuation)
        self.r_end = np.zeros(n_reservoir)

    # ------------------------------------------------------------------
    # Internal reservoir update
    # ------------------------------------------------------------------
    def _update_state(self, r: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Single-step reservoir state update.
        """
        return (1.0 - self.alpha) * r + self.alpha * np.tanh(
            self.W @ r + self.W_in @ u + self.bias
        )

    def _nonlinear_expand(self, r: np.ndarray) -> np.ndarray:
        """
        Apply nonlinear feature expansion:
        square every second reservoir unit (MATLAB: r_out(2:2:end).^2)
        """
        r_exp = r.copy()
        r_exp[1::2] = r_exp[1::2] ** 2
        return r_exp

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        washout: int,
    ):
        """
        Train the reservoir output layer using ridge regression.

        Parameters
        ----------
        U : ndarray, shape (T, dim_in)
            Reservoir input time series.
        Y : ndarray, shape (T, dim_out)
            Target output (joint torques).
        washout : int
            Number of initial samples discarded (wash-up period).
        """

        T = U.shape[0]

        r = np.zeros(self.n)
        R_collect = []
        Y_collect = []

        for t in range(T):
            r = self._update_state(r, U[t])
            r_nl = self._nonlinear_expand(r)

            if t >= washout:
                R_collect.append(r_nl)
                Y_collect.append(Y[t])

        R = np.array(R_collect).T    # shape: (n_reservoir, samples)
        Yt = np.array(Y_collect).T   # shape: (dim_out, samples)

        # Ridge regression (exact MATLAB equivalent)
        self.W_out = Yt @ R.T @ np.linalg.inv(
            R @ R.T + self.beta * np.eye(self.n)
        )

        # Store final reservoir state
        self.r_end = r.copy()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        U: np.ndarray,
        r_init: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict outputs for a given input sequence.

        Parameters
        ----------
        U : ndarray, shape (T, dim_in)
            Reservoir input sequence.
        r_init : ndarray or None
            Initial reservoir state. If None, uses stored r_end.

        Returns
        -------
        Y_pred : ndarray, shape (T, dim_out)
            Predicted output sequence.
        """

        T = U.shape[0]
        r = self.r_end.copy() if r_init is None else r_init.copy()

        Y_pred = np.zeros((T, self.dim_out))

        for t in range(T):
            r = self._update_state(r, U[t])
            r_nl = self._nonlinear_expand(r)
            Y_pred[t] = self.W_out @ r_nl

        return Y_pred
