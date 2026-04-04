from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from features_data import build_descriptor_matrix, list_descriptor_names


def prepare_training_data(
    train_df: pd.DataFrame,
    mols: list,
    y_col: str = "pEC50",
) -> tuple[np.ndarray, list, np.ndarray]:
    """Drop rows with invalid target or missing molecule."""
    y = train_df[y_col].to_numpy(dtype=float)
    mol_ok = np.array([m is not None for m in mols], dtype=bool)
    mask = np.isfinite(y) & mol_ok
    n_drop = int((~mask).sum())
    if n_drop:
        print(f"prepare_training_data: dropped {n_drop} rows (invalid y or mol).")
    mols_f = [m for m, ok in zip(mols, mask) if ok]
    return y[mask], mols_f, mask


def _needs_scaling(model_name: str) -> bool:
    return model_name in ("ridge", "elasticnet", "svr")


def make_regressor_pipeline(model_name: str, estimator: Any) -> Pipeline:
    imputer = SimpleImputer(strategy="mean")
    if _needs_scaling(model_name):
        return Pipeline(
            [
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    return Pipeline([("imputer", imputer), ("model", estimator)])


def default_regressors(random_state: int = 0) -> dict[str, Any]:
    return {
        "ridge": Ridge(alpha=1.0),
        "elasticnet": ElasticNet(
            alpha=0.1, random_state=random_state, max_iter=8000
        ),
        "rf": RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        "hgb": HistGradientBoostingRegressor(
            max_iter=200, random_state=random_state
        ),
        "svr": SVR(kernel="rbf", C=1.0),
    }


@dataclass
class BaselineCVConfig:
    y_col: str = "pEC50"
    n_splits: int = 5
    shuffle: bool = True
    cv_random_state: int = 0
    model_random_state: int = 0


def run_baseline_cv(
    train_df: pd.DataFrame,
    mols: list,
    *,
    descriptor_names: list[str] | None = None,
    regressors: dict[str, Any] | None = None,
    config: BaselineCVConfig | None = None,
) -> pd.DataFrame:
    """
    Cross-validate each (descriptor, regressor) pair; return sorted results table.
    """
    cfg = config or BaselineCVConfig()
    y, mols_f, _ = prepare_training_data(train_df, mols, y_col=cfg.y_col)
    names = descriptor_names or list_descriptor_names()
    regs = regressors or default_regressors(cfg.model_random_state)

    cv = KFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=cfg.cv_random_state,
    )
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    rows: list[dict[str, Any]] = []
    for desc in names:
        X = build_descriptor_matrix(desc, mols_f).astype(np.float64)
        for model_name, est in regs.items():
            pipe = make_regressor_pipeline(model_name, est)
            out = cross_validate(
                pipe,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
            rmse_scores = -out["test_rmse"]
            mae_scores = -out["test_mae"]
            rows.append(
                {
                    "descriptor": desc,
                    "model": model_name,
                    "mean_rmse": float(rmse_scores.mean()),
                    "std_rmse": float(rmse_scores.std()),
                    "mean_mae": float(mae_scores.mean()),
                    "std_mae": float(mae_scores.std()),
                    "mean_r2": float(out["test_r2"].mean()),
                    "std_r2": float(out["test_r2"].std()),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values("mean_rmse", ascending=True).reset_index(drop=True)


def fit_best_on_full_train(
    train_df: pd.DataFrame,
    mols: list,
    descriptor: str,
    model_name: str,
    *,
    regressors: dict[str, Any] | None = None,
    config: BaselineCVConfig | None = None,
) -> Pipeline:
    """Fit chosen pipeline on all valid training rows."""
    cfg = config or BaselineCVConfig()
    y, mols_f, _ = prepare_training_data(train_df, mols, y_col=cfg.y_col)
    regs = regressors or default_regressors(cfg.model_random_state)
    if model_name not in regs:
        raise KeyError(f"Unknown model {model_name!r}")
    est = clone(regs[model_name])
    pipe = make_regressor_pipeline(model_name, est)
    X = build_descriptor_matrix(descriptor, mols_f).astype(np.float64)
    pipe.fit(X, y)
    return pipe


def cross_val_predict_baseline(
    train_df: pd.DataFrame,
    mols: list,
    descriptor: str,
    model_name: str,
    *,
    regressors: dict[str, Any] | None = None,
    config: BaselineCVConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions for diagnostics (same CV as run_baseline_cv)."""
    cfg = config or BaselineCVConfig()
    y, mols_f, _ = prepare_training_data(train_df, mols, y_col=cfg.y_col)
    regs = regressors or default_regressors(cfg.model_random_state)
    if model_name not in regs:
        raise KeyError(f"Unknown model {model_name!r}")
    est = clone(regs[model_name])
    pipe = make_regressor_pipeline(model_name, est)
    X = build_descriptor_matrix(descriptor, mols_f).astype(np.float64)
    cv = KFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=cfg.cv_random_state,
    )
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    return y, y_pred
