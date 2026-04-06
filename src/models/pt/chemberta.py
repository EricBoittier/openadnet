"""ChemBERTa-style SMILES regression (RoBERTa + PubChem BPE), without DeepChem."""

from __future__ import annotations

from typing import Optional

import torch

from models.hf_regression import HuggingFaceRegressor


class ChembertaRegressor(HuggingFaceRegressor):
    """Regression on SMILES using a RoBERTa sequence classifier and ChemBERTa-style tokenization.

    Defaults follow common ChemBERTa setups: PubChem10M byte-pair tokenizer
    (`seyonec/PubChem10M_SMILES_BPE_60k`) and a pretrained encoder checkpoint on the Hub.
    Pass ``tokenizer_path=None`` to load the tokenizer from the same repo as ``model_name_or_path``.

    Parameters
    ----------
    model_name_or_path
        Hugging Face model id or local directory with weights (regression head is resized to ``n_tasks``).
    n_tasks
        Number of regression targets.
    tokenizer_path
        Hub id or path for the tokenizer. Defaults to PubChem SMILES BPE. Use ``None`` to use
        ``model_name_or_path`` for both model and tokenizer.
    max_length
        Maximum sequence length for tokenization.
    device
        Torch device; default is CUDA if available else CPU.

    References
    ----------
    Chithrananda et al., ChemBERTa (2020). Ahmad et al., ChemBERTa-2 (2022).
    """

    def __init__(
        self,
        model_name_or_path: str = "DeepChem/ChemBERTa-77M-MLM",
        n_tasks: int = 1,
        *,
        tokenizer_path: Optional[str] = "seyonec/PubChem10M_SMILES_BPE_60k",
        max_length: Optional[int] = 512,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            model_name_or_path,
            n_tasks=n_tasks,
            tokenizer_name_or_path=tokenizer_path,
            max_length=max_length,
            device=device,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "DeepChem/ChemBERTa-77M-MLM",
        n_tasks: int = 1,
        *,
        tokenizer_path: Optional[str] = "seyonec/PubChem10M_SMILES_BPE_60k",
        max_length: Optional[int] = 512,
        device: Optional[torch.device] = None,
    ) -> "ChembertaRegressor":
        return cls(
            model_name_or_path,
            n_tasks=n_tasks,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            device=device,
        )

    def load_from_hf(
        self,
        model_name_or_path: str,
        *,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        """Reload model from the Hub or a local directory.

        If ``tokenizer_path`` is omitted, the tokenizer is loaded from ``model_name_or_path``.
        """
        super().load_from_hf(
            model_name_or_path,
            tokenizer_name_or_path=tokenizer_path,
        )
