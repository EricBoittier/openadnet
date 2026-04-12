"""PyTorch deep learning models for regression (transformers and GNNs).

Requires optional dependencies: ``pip install openadnet[dl]``.
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "EnsembleQuantileRegressor",
    "EnsembleRegressor",
    "FingerprintEnsembleMember",
    "FingerprintQuantileMember",
    "PhysGatedMorganQuantileMoE",
    "cross_validate_phys_gated_morgan_quantile_moe",
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
    if name == "FingerprintEnsembleMember":
        from models.ensemble import FingerprintEnsembleMember

        return FingerprintEnsembleMember
    if name == "FingerprintQuantileMember":
        from models.moe_quantile import FingerprintQuantileMember

        return FingerprintQuantileMember
    if name == "PhysGatedMorganQuantileMoE":
        from models.moe_quantile import PhysGatedMorganQuantileMoE

        return PhysGatedMorganQuantileMoE
    if name == "cross_validate_phys_gated_morgan_quantile_moe":
        from models.moe_quantile import cross_validate_phys_gated_morgan_quantile_moe

        return cross_validate_phys_gated_morgan_quantile_moe
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
        FingerprintEnsembleMember as FingerprintEnsembleMember,
        pinball_loss as pinball_loss,
    )
    from models.moe_quantile import (
        FingerprintQuantileMember as FingerprintQuantileMember,
        PhysGatedMorganQuantileMoE as PhysGatedMorganQuantileMoE,
        cross_validate_phys_gated_morgan_quantile_moe as cross_validate_phys_gated_morgan_quantile_moe,
    )
    from models.hf_regression import HuggingFaceRegressor as HuggingFaceRegressor
    from models.gnn_regression import GNNRegressor as GNNRegressor
    from models.nn.pyg_regressor import PyGMoleculeRegressor as PyGMoleculeRegressor
