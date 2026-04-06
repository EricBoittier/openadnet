"""PyTorch deep learning models for regression (transformers and GNNs).

Requires optional dependencies: ``pip install openadnet[dl]``.
"""

from typing import TYPE_CHECKING, Any

__all__ = ["EnsembleRegressor", "HuggingFaceRegressor", "GNNRegressor", "PyGMoleculeRegressor"]


def __getattr__(name: str) -> Any:
    if name == "EnsembleRegressor":
        from models.ensemble import EnsembleRegressor

        return EnsembleRegressor
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
    from models.ensemble import EnsembleRegressor as EnsembleRegressor
    from models.hf_regression import HuggingFaceRegressor as HuggingFaceRegressor
    from models.gnn_regression import GNNRegressor as GNNRegressor
    from models.nn.pyg_regressor import PyGMoleculeRegressor as PyGMoleculeRegressor
