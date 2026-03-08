import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array


class MP_PCA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sigma_estimator="median",
        n_components=None,
        regression=False,
        verbose=True,
    ):
        """
        Marchenko-Pastur PCA (high-dimensional).

        Parameters
        ----------
        sigma_estimator : str
            Method to estimate noise variance:
            "median", "bulk_mean", "min_eigen", "mp_fit"
        n_components : int or None
            Maximum number of components to keep
        regression : bool
            Whether to fit LinearRegression on Y in fit()
        verbose : bool
            Print info
        """
        self.sigma_estimator = sigma_estimator
        self.n_components = n_components
        self.regression = regression
        self.verbose = verbose

    def fit(self, X, Y=None):
        X = check_array(X, ensure_2d=True, dtype=float)
        n, p = X.shape
        self.n_samples_ = n
        self.n_features_ = p
        self.gamma_ = p / n

        # Center the data
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_

        # Economy SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        eigenvals = (S**2) / n

        self.U_ = U
        self.S_ = S
        self.Vt_ = Vt
        self.eigenvalues_ = eigenvals

        # Estimate sigma^2
        if self.sigma_estimator == "median":
            sigma2_hat = np.median(eigenvals) / (1 + self.gamma_)
        elif self.sigma_estimator == "bulk_mean":
            sigma2_hat = np.mean(eigenvals)
        elif self.sigma_estimator == "min_eigen":
            sigma2_hat = np.min(eigenvals) / (1 - np.sqrt(self.gamma_)) ** 2
        elif self.sigma_estimator == "mp_fit":
            # simple MP fit using KL divergence or least-squares of bulk histogram
            sigma2_hat = self._fit_mp_bulk(eigenvals, self.gamma_)
        else:
            raise ValueError(f"Unknown sigma_estimator {self.sigma_estimator}")

        self.sigma2_ = sigma2_hat
        self.lambda_mp_max_ = sigma2_hat * (1 + np.sqrt(self.gamma_)) ** 2

        # Retain spikes
        kept_mask = eigenvals > self.lambda_mp_max_
        kept_idx = np.where(kept_mask)[0]

        if self.n_components is not None:
            kept_idx = kept_idx[: self.n_components]

        self.kept_idx_ = kept_idx
        V = Vt.T
        self.pcs_ = V[:, kept_idx] if len(kept_idx) > 0 else np.zeros((p, 0))
        self.scores_ = Xc @ self.pcs_ if len(kept_idx) > 0 else np.zeros((n, 0))

        # Optional regression on Y
        self.reg_ = None
        self.r2_ = None
        if Y is not None and self.regression:
            Y = np.asarray(Y).reshape(-1)
            if self.scores_.shape[1] > 0:
                self.reg_ = LinearRegression()
                self.reg_.fit(self.scores_, Y)
                Y_pred = self.reg_.predict(self.scores_)
                self.r2_ = r2_score(Y, Y_pred)
            elif self.verbose:
                print("No MP components retained, regression skipped")

        if self.verbose:
            print(f"Estimated sigma^2: {self.sigma2_:.6g}")
            print(f"MP upper edge: {self.lambda_mp_max_:.6g}")
            print(f"Number of components retained: {self.pcs_.shape[1]}")

        return self

    def transform(self, X):
        """Return the denoised reconstruction with the same shape as X."""
        X = check_array(X, ensure_2d=True, dtype=float)
        Xc = X - self.mean_
        scores = Xc @ self.pcs_          # (n, k)
        return scores @ self.pcs_.T + self.mean_  # (n, m) — reconstructed

    def fit_transform(self, X, Y=None):
        """Fit and return the denoised reconstruction with the same shape as X."""
        self.fit(X, Y)
        return self.scores_ @ self.pcs_.T + self.mean_  # (n, m) — reconstructed

    # -------------------------------
    # Private helper: simple MP fit
    # -------------------------------
    def _fit_mp_bulk(self, eigenvals, gamma):
        """
        Fit sigma^2 to match the bulk of eigenvalues to MP upper edge using least squares.
        This is a simple implementation.
        """

        def loss(sigma2):
            sigma2 = float(sigma2)
            lambda_max = sigma2 * (1 + np.sqrt(gamma)) ** 2
            bulk = eigenvals[eigenvals <= lambda_max]
            # we try to fit median of bulk = sigma2*(1+gamma)
            target = sigma2 * (1 + gamma)
            return (np.median(bulk) - target) ** 2

        res = minimize(
            loss, x0=[np.median(eigenvals) / (1 + gamma)], bounds=[(1e-12, None)]
        )
        return float(res.x)

    def get_params(self, deep=True):
        return {
            "sigma_estimator": self.sigma_estimator,
            "n_components": self.n_components,
            "regression": self.regression,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
