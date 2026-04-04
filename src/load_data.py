"""
Load PXR challenge tables from the Hugging Face Hub.

Files are cached on disk by ``huggingface_hub`` (default: ``~/.cache/huggingface/hub``).
Set ``OPENADNET_HF_CACHE`` to override the cache directory.

In-process: repeated loads of the same file return the same DataFrame (``lru_cache``).
Use ``clear_data_cache()`` to drop in-memory tables (e.g. in tests).
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "openadmet/pxr-challenge-train-test"
REPO_TYPE = "dataset"

_FILENAMES = {
    "train": "pxr-challenge_TRAIN.csv",
    "test": "pxr-challenge_TEST_BLINDED.csv",
    "train_counter": "pxr-challenge_counter-assay_TRAIN.csv",
    "test_structure": "pxr-challenge_structure_TEST_BLINDED.csv",
    "train_single": "pxr-challenge_single_concentration_TRAIN.csv",
}


def _hub_download_kwargs(filename: str) -> dict:
    out: dict = {
        "repo_id": REPO_ID,
        "filename": filename,
        "repo_type": REPO_TYPE,
    }
    cache = os.environ.get("OPENADNET_HF_CACHE")
    if cache:
        out["cache_dir"] = cache
    return out


@lru_cache(maxsize=len(_FILENAMES))
def _read_cached_csv(filename: str) -> pd.DataFrame:
    path = hf_hub_download(**_hub_download_kwargs(filename))
    return pd.read_csv(path)


def clear_data_cache() -> None:
    """Clear in-memory DataFrame cache (does not remove Hugging Face disk cache)."""
    _read_cached_csv.cache_clear()


def get_train() -> pd.DataFrame:
    return _read_cached_csv(_FILENAMES["train"])


def get_test() -> pd.DataFrame:
    return _read_cached_csv(_FILENAMES["test"])


def get_train_counter() -> pd.DataFrame:
    return _read_cached_csv(_FILENAMES["train_counter"])


def get_test_structure() -> pd.DataFrame:
    return _read_cached_csv(_FILENAMES["test_structure"])


def get_train_single() -> pd.DataFrame:
    return _read_cached_csv(_FILENAMES["train_single"])


# Module-level aliases (same names as before; now backed by hub + LRU cache)
train = get_train()
test = get_test()
train_counter = get_train_counter()
test_structure = get_test_structure()
train_single = get_train_single()
