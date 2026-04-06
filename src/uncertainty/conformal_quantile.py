"""
Conformalized quantile regression (CQR) as a scikit-learn meta-estimator.

``ConformalizedQuantileRegressor.fit`` requires explicit calibration arrays.
For cross-validation with the same outer folds as a standard benchmark, use
:func:`cross_validate_conformal_quantile`, which splits each training fold into
fit vs calibration.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import check_cv, train_test_split
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ConformalizedQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible conformalized quantile regressor.

    Parameters
    ----------
    estimator : estimator
        A sklearn regressor supporting quantile regression. It must accept either:

        - ``loss="quantile"`` and ``quantile=<float in (0, 1)>`` (e.g.
          :class:`~sklearn.ensemble.HistGradientBoostingRegressor`), or
        - ``loss="quantile"`` and ``alpha=<float in (0, 1)>`` (e.g.
          :class:`~sklearn.ensemble.GradientBoostingRegressor`).

    alpha : float, default=0.1
        Target miscoverage. ``alpha=0.1`` implies nominal ~90% marginal coverage
        for the conformalized interval (under exchangeability).

    Attributes
    ----------
    lower_estimator_ : estimator
        Fitted lower quantile model.
    median_estimator_ : estimator
        Fitted median model (used for point predictions).
    upper_estimator_ : estimator
        Fitted upper quantile model.
    conformal_offset_ : float
        Non-negative score quantile applied symmetrically to widen the raw
        quantile interval on calibration data.
    """

    def __init__(self, estimator, *, alpha: float = 0.1):
        self.estimator = estimator
        self.alpha = alpha

    def fit(self, X, y, X_cal, y_cal):
        """
        Fit quantile models on (X, y) and calibrate conformal offset on (X_cal, y_cal).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        X_cal : array-like of shape (n_cal, n_features)
            Calibration features (held out from quantile fitting).
        y_cal : array-like of shape (n_cal,)
            Calibration targets.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=False)

        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal).ravel()
        if X_cal.shape[0] == 0:
            raise ValueError("Calibration set must be non-empty.")
        X_cal, y_cal = check_X_y(
            X_cal, y_cal, y_numeric=True, multi_output=False
        )

        if X_cal.shape[0] != y_cal.shape[0]:
            raise ValueError("X_cal and y_cal must have the same number of rows.")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be strictly between 0 and 1.")

        lower_q = self.alpha / 2.0
        upper_q = 1.0 - self.alpha / 2.0
        median_q = 0.5

        self.lower_estimator_ = self._make_quantile_estimator(lower_q)
        self.median_estimator_ = self._make_quantile_estimator(median_q)
        self.upper_estimator_ = self._make_quantile_estimator(upper_q)

        self.lower_estimator_.fit(X, y)
        self.median_estimator_.fit(X, y)
        self.upper_estimator_.fit(X, y)

        q_lo = self.lower_estimator_.predict(X_cal)
        q_hi = self.upper_estimator_.predict(X_cal)

        scores = np.maximum.reduce(
            [q_lo - y_cal, y_cal - q_hi, np.zeros_like(y_cal, dtype=float)]
        )

        n = int(scores.shape[0])
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = float(min(level, 1.0))

        self.conformal_offset_ = float(np.quantile(scores, level, method="higher"))
        return self

    def predict(self, X):
        """Point prediction: median quantile."""
        check_is_fitted(self, "median_estimator_")
        X = check_array(X)
        return self.median_estimator_.predict(X)

    def predict_interval(self, X):
        """Return ``(n_samples, 2)`` array of [lower, upper] conformalized bounds."""
        check_is_fitted(self, "conformal_offset_")
        X = check_array(X)
        q_lo = self.lower_estimator_.predict(X)
        q_hi = self.upper_estimator_.predict(X)
        lower = q_lo - self.conformal_offset_
        upper = q_hi + self.conformal_offset_
        return np.column_stack([lower, upper])

    def predict_full(self, X):
        """Return ``(y_pred, interval)`` with median prediction and conformal interval."""
        y_pred = self.predict(X)
        interval = self.predict_interval(X)
        return y_pred, interval

    def _make_quantile_estimator(self, q: float):
        est = clone(self.estimator)
        params = est.get_params(deep=True)

        if "loss" in params and "quantile" in params:
            est.set_params(loss="quantile", quantile=q)
            return est

        if "loss" in params and "alpha" in params:
            est.set_params(loss="quantile", alpha=q)
            return est

        # Pipelines and other composites expose nested keys like model__loss.
        for key in params:
            if not key.endswith("__quantile"):
                continue
            prefix = key[: -len("__quantile")]
            loss_key = f"{prefix}__loss" if prefix else "loss"
            if loss_key in params:
                est.set_params(**{loss_key: "quantile", key: q})
                return est

        for key in params:
            if not key.endswith("__alpha"):
                continue
            prefix = key[: -len("__alpha")]
            loss_key = f"{prefix}__loss" if prefix else "loss"
            if loss_key in params:
                est.set_params(**{loss_key: "quantile", key: q})
                return est

        raise ValueError(
            "Base estimator must support quantile regression via either "
            "`loss='quantile', quantile=q` or `loss='quantile', alpha=q` "
            "(including nested params on the final regressor in a Pipeline)."
        )


def cross_validate_conformal_quantile(
    estimator: ConformalizedQuantileRegressor,
    X,
    y,
    cv,
    *,
    calibration_fraction: float = 0.2,
    random_state: int | None = 0,
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """
    Out-of-fold scores using the same CV splits as ``sklearn.model_selection.cross_validate``,
    with a nested fit/calibration split inside each training fold for CQR.

    Point metrics (RMSE, MAE, R²) use median predictions on the held-out fold.

    Parameters
    ----------
    estimator : ConformalizedQuantileRegressor
        Unfitted template; cloned per fold.
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    cv : int or CV splitter
        Same object you pass to ``cross_validate`` (e.g. ``KFold(...)``).
    calibration_fraction : float, default=0.2
        Fraction of each **training fold** reserved for conformal calibration
        (the rest fits the three quantile models).
    random_state : int or None, default=0
        Base seed for ``train_test_split`` inside each fold; actual seed is
        ``random_state + fold_index`` when ``random_state`` is an int.
    n_jobs : int, default=1
        Parallel folds via joblib (``1`` disables parallelism).

    Returns
    -------
    dict
        Keys ``test_rmse``, ``test_mae``, ``test_r2`` — each a ``(n_folds,)`` array
        of **positive** RMSE and standard MAE / R².
    """
    X, y = check_X_y(X, y, y_numeric=True, multi_output=False)
    if not (0.0 < calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be strictly between 0 and 1.")

    cv = check_cv(cv, y, classifier=False)
    folds = list(cv.split(X, y))

    def _one_fold(
        fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray
    ) -> tuple[float, float, float]:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        n_tr = X_tr.shape[0]
        if n_tr < 2:
            raise ValueError(
                "CQR CV requires at least 2 training samples per fold; "
                f"got {n_tr}."
            )
        n_cal = max(1, min(int(round(calibration_fraction * n_tr)), n_tr - 1))
        rs = None if random_state is None else int(random_state) + int(fold_idx)
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_tr,
            y_tr,
            test_size=n_cal,
            random_state=rs,
            shuffle=True,
        )
        est = clone(estimator)
        est.fit(X_fit, y_fit, X_cal, y_cal)
        y_pred = est.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae = float(mean_absolute_error(y_te, y_pred))
        r2 = float(r2_score(y_te, y_pred))
        return rmse, mae, r2

    if n_jobs == 1:
        scores = [_one_fold(i, tr, te) for i, (tr, te) in enumerate(folds)]
    else:
        scores = Parallel(n_jobs=n_jobs)(
            delayed(_one_fold)(i, tr, te) for i, (tr, te) in enumerate(folds)
        )

    rmse = np.array([s[0] for s in scores], dtype=float)
    mae = np.array([s[1] for s in scores], dtype=float)
    r2 = np.array([s[2] for s in scores], dtype=float)
    return {"test_rmse": rmse, "test_mae": mae, "test_r2": r2}
