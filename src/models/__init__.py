"""PyTorch deep learning models for regression (transformers and GNNs).

Requires optional dependencies: ``pip install openadnet[dl]``.
"""

from typing import TYPE_CHECKING, Any

__all__ = ["HuggingFaceRegressor", "GNNRegressor"]


def __getattr__(name: str) -> Any:
    if name == "HuggingFaceRegressor":
        from models.hf_regression import HuggingFaceRegressor

        return HuggingFaceRegressor
    if name == "GNNRegressor":
        from models.gnn_regression import GNNRegressor

        return GNNRegressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from models.hf_regression import HuggingFaceRegressor as HuggingFaceRegressor
    from models.gnn_regression import GNNRegressor as GNNRegressor
