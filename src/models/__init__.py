"""PyTorch deep learning models for regression (transformers and GNNs).

Requires optional dependencies: ``pip install openadnet[dl]``.
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "EnsembleQuantileRegressor",
    "EnsembleRegressor",
    "HuggingFaceRegressor",
    "GNNRegressor",
    "PyGMoleculeRegressor",
    "pinball_loss",
]


def __getattr__(name: str) -> Any:
    if name == "EnsembleQuantileRegressor":
        from models.ensemble import EnsembleQuantileRegressor

        return EnsembleQuantileRegressor
    if name == "EnsembleRegressor":
        from models.ensemble import EnsembleRegressor

        return EnsembleRegressor
    if name == "pinball_loss":
        from models.ensemble import pinball_loss

        return pinball_loss
    if name == "HuggingFaceRegressor":
        from models.hf_regression import HuggingFaceRegressor

        return HuggingFaceRegressor
    if name == "GNNRegressor":
        from models.gnn_regression import GNNRegressor

        return GNNRegressor
    if name == "PyGMoleculeRegressor":
        from models.nn.pyg_regressor import PyGMoleculeRegressor

        return PyGMoleculeRegressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from models.ensemble import (
        EnsembleQuantileRegressor as EnsembleQuantileRegressor,
        EnsembleRegressor as EnsembleRegressor,
        pinball_loss as pinball_loss,
    )
    from models.hf_regression import HuggingFaceRegressor as HuggingFaceRegressor
    from models.gnn_regression import GNNRegressor as GNNRegressor
    from models.nn.pyg_regressor import PyGMoleculeRegressor as PyGMoleculeRegressor
