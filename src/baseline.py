from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from features_data import build_descriptor_matrix, list_descriptor_names
from uncertainty.conformal_quantile import (
    ConformalizedQuantileRegressor,
    cross_validate_conformal_quantile,
)

# Bump when cache key semantics change (invalidates old JSON entries).
CV_RESULT_CACHE_SCHEMA = "fp_v2_cv_1"
CV_RESULT_CACHE_SCHEMA_CQR = "fp_v2_cqr_cv_1"


def default_cv_cache_path() -> Path:
    return Path(__file__).resolve().parent.parent / "outputs" / "baseline_cv_cache.json"


def default_cqr_cv_cache_path() -> Path:
    return Path(__file__).resolve().parent.parent / "outputs" / "baseline_cqr_cv_cache.json"


def train_set_id(train_df: pd.DataFrame, y_col: str) -> str:
    """Stable hash of SMILES + target column (order-independent)."""
    if y_col not in train_df.columns:
        raise KeyError(y_col)
    smi = train_df["SMILES"].astype(str).fillna("")
    y = train_df[y_col]
    order = smi.argsort(kind="mergesort")
    smi_s = smi.iloc[order].values
    y_s = y.iloc[order].values
    lines = []
    for s, v in zip(smi_s, y_s):
        if pd.isna(v):
            lines.append(f"{s}\tnan")
        else:
            lines.append(f"{s}\t{float(v)}")
    payload = "\n".join(lines)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _cv_cache_key(
    train_id: str, desc: str, model: str, cfg: BaselineCVConfig
) -> str:
    return (
        f"{CV_RESULT_CACHE_SCHEMA}|{train_id}|{desc}|{model}|"
        f"{cfg.n_splits}|{cfg.shuffle}|{cfg.cv_random_state}|"
        f"{cfg.model_random_state}|{cfg.y_col}"
    )


def _cv_cache_key_cqr(
    train_id: str,
    desc: str,
    cfg: BaselineCVConfig,
    *,
    alpha: float,
    calibration_fraction: float,
    cqr_random_state: int | None,
) -> str:
    return (
        f"{CV_RESULT_CACHE_SCHEMA_CQR}|{train_id}|{desc}|cqr_hgb|"
        f"{cfg.n_splits}|{cfg.shuffle}|{cfg.cv_random_state}|"
        f"{cfg.model_random_state}|{cfg.y_col}|alpha={alpha}|"
        f"cal={calibration_fraction}|cqr_rs={cqr_random_state}"
    )


def _load_cqr_cv_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": CV_RESULT_CACHE_SCHEMA_CQR, "train_id": None, "entries": {}}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != CV_RESULT_CACHE_SCHEMA_CQR:
        return {"schema": CV_RESULT_CACHE_SCHEMA_CQR, "train_id": None, "entries": {}}
    return data


def _load_cv_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": CV_RESULT_CACHE_SCHEMA, "train_id": None, "entries": {}}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != CV_RESULT_CACHE_SCHEMA:
        return {"schema": CV_RESULT_CACHE_SCHEMA, "train_id": None, "entries": {}}
    return data


def _save_cv_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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


def default_cqr_hgb_regressor(
    config: BaselineCVConfig | None = None,
    *,
    alpha: float = 0.1,
) -> ConformalizedQuantileRegressor:
    """
    HistGradientBoosting quantile loss inside the same imputer-only pipeline as ``hgb``.
    """
    cfg = config or BaselineCVConfig()
    base = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.5,
        max_iter=200,
        random_state=cfg.model_random_state,
    )
    pipe = make_regressor_pipeline("hgb", base)
    return ConformalizedQuantileRegressor(estimator=pipe, alpha=alpha)


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
    cv_cache_path: Path | None = None,
    use_cv_cache: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Cross-validate each (descriptor, regressor) pair; return sorted results table.

    If ``use_cv_cache`` and ``cv_cache_path`` (default: ``outputs/baseline_cv_cache.json``)
    are set, reuse stored metrics for the same training set hash and cache key; only
    missing pairs run ``cross_validate``.

    Set ``show_progress=False`` to disable the tqdm progress bar (e.g. in tests).
    """
    cfg = config or BaselineCVConfig()
    y, mols_f, mask = prepare_training_data(train_df, mols, y_col=cfg.y_col)
    train_f = train_df.loc[mask].reset_index(drop=True)
    train_id = train_set_id(train_f, cfg.y_col)
    names = descriptor_names or list_descriptor_names()
    regs = regressors or default_regressors(cfg.model_random_state)

    cache_file = cv_cache_path if cv_cache_path is not None else default_cv_cache_path()

    cache_data: dict[str, Any] = {"schema": CV_RESULT_CACHE_SCHEMA, "train_id": train_id, "entries": {}}
    if use_cv_cache:
        loaded = _load_cv_cache(cache_file)
        if loaded.get("train_id") == train_id and loaded.get("schema") == CV_RESULT_CACHE_SCHEMA:
            cache_data["entries"] = dict(loaded.get("entries", {}))
        else:
            cache_data["entries"] = {}

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
    n_hit = 0
    n_run = 0
    X_by_desc: dict[str, np.ndarray] = {}
    pairs = list(product(names, regs.keys()))
    pair_iter = tqdm(
        pairs,
        desc="baseline_cv",
        unit="pair",
        disable=not show_progress,
    )
    for desc, model_name in pair_iter:
        est = regs[model_name]
        ck = _cv_cache_key(train_id, desc, model_name, cfg)
        if use_cv_cache and ck in cache_data["entries"]:
            rows.append(dict(cache_data["entries"][ck]))
            n_hit += 1
            continue
        if desc not in X_by_desc:
            X_by_desc[desc] = build_descriptor_matrix(desc, mols_f).astype(np.float64)
        X = X_by_desc[desc]
        n_run += 1
        short = desc if len(desc) <= 32 else f"{desc[:31]}..."
        pair_iter.set_postfix(desc=short, model=model_name)
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
        row = {
            "descriptor": desc,
            "model": model_name,
            "mean_rmse": float(rmse_scores.mean()),
            "std_rmse": float(rmse_scores.std()),
            "mean_mae": float(mae_scores.mean()),
            "std_mae": float(mae_scores.std()),
            "mean_r2": float(out["test_r2"].mean()),
            "std_r2": float(out["test_r2"].std()),
        }
        rows.append(row)
        cache_data["entries"][ck] = row

    if use_cv_cache and n_run:
        _save_cv_cache(cache_file, cache_data)

    if use_cv_cache and (n_hit or n_run):
        print(
            f"baseline_cv cache: train_id={train_id} "
            f"hits={n_hit} new_runs={n_run} path={cache_file}"
        )

    df = pd.DataFrame(rows)
    return df.sort_values("mean_rmse", ascending=True).reset_index(drop=True)


def run_baseline_cqr_cv(
    train_df: pd.DataFrame,
    mols: list,
    *,
    descriptor_names: list[str] | None = None,
    config: BaselineCVConfig | None = None,
    alpha: float = 0.1,
    calibration_fraction: float = 0.2,
    cqr_random_state: int | None = 0,
    cv_cache_path: Path | None = None,
    use_cv_cache: bool = True,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Cross-validate CQR (median point prediction) with the same outer ``KFold`` as
    :func:`run_baseline_cv`. Each training fold is split into fit vs calibration
    for conformal scores (nested split; test fold is untouched).

    Results use the same columns as ``run_baseline_cv`` with ``model`` fixed to
    ``cqr_hgb``.
    """
    cfg = config or BaselineCVConfig()
    y, mols_f, mask = prepare_training_data(train_df, mols, y_col=cfg.y_col)
    train_f = train_df.loc[mask].reset_index(drop=True)
    train_id = train_set_id(train_f, cfg.y_col)
    names = descriptor_names or list_descriptor_names()

    cache_file = cv_cache_path if cv_cache_path is not None else default_cqr_cv_cache_path()

    cache_data: dict[str, Any] = {
        "schema": CV_RESULT_CACHE_SCHEMA_CQR,
        "train_id": train_id,
        "entries": {},
    }
    if use_cv_cache:
        loaded = _load_cqr_cv_cache(cache_file)
        if loaded.get("train_id") == train_id and loaded.get("schema") == CV_RESULT_CACHE_SCHEMA_CQR:
            cache_data["entries"] = dict(loaded.get("entries", {}))
        else:
            cache_data["entries"] = {}

    cv = KFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=cfg.cv_random_state,
    )

    rows: list[dict[str, Any]] = []
    n_hit = 0
    n_run = 0
    X_by_desc: dict[str, np.ndarray] = {}
    desc_iter = tqdm(
        names,
        desc="baseline_cqr_cv",
        unit="desc",
        disable=not show_progress,
    )
    for desc in desc_iter:
        ck = _cv_cache_key_cqr(
            train_id,
            desc,
            cfg,
            alpha=alpha,
            calibration_fraction=calibration_fraction,
            cqr_random_state=cqr_random_state,
        )
        if use_cv_cache and ck in cache_data["entries"]:
            rows.append(dict(cache_data["entries"][ck]))
            n_hit += 1
            continue
        if desc not in X_by_desc:
            X_by_desc[desc] = build_descriptor_matrix(desc, mols_f).astype(np.float64)
        X = X_by_desc[desc]
        n_run += 1
        short = desc if len(desc) <= 32 else f"{desc[:31]}..."
        desc_iter.set_postfix(desc=short, model="cqr_hgb")
        cqr = default_cqr_hgb_regressor(cfg, alpha=alpha)
        out = cross_validate_conformal_quantile(
            cqr,
            X,
            y,
            cv,
            calibration_fraction=calibration_fraction,
            random_state=cqr_random_state,
            n_jobs=n_jobs,
        )
        rmse_scores = out["test_rmse"]
        mae_scores = out["test_mae"]
        row = {
            "descriptor": desc,
            "model": "cqr_hgb",
            "mean_rmse": float(rmse_scores.mean()),
            "std_rmse": float(rmse_scores.std()),
            "mean_mae": float(mae_scores.mean()),
            "std_mae": float(mae_scores.std()),
            "mean_r2": float(out["test_r2"].mean()),
            "std_r2": float(out["test_r2"].std()),
        }
        rows.append(row)
        cache_data["entries"][ck] = row

    if use_cv_cache and n_run:
        _save_cv_cache(cache_file, cache_data)

    if use_cv_cache and (n_hit or n_run):
        print(
            f"baseline_cqr_cv cache: train_id={train_id} "
            f"hits={n_hit} new_runs={n_run} path={cache_file}"
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
