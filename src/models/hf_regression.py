"""Hugging Face transformers for regression-only finetuning."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)

from models.data.transformer import SmilesRegressionDataset, smiles_regression_collate_fn

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def _default_device(requested: Optional[torch.device]) -> torch.device:
    if requested is not None:
        return requested
    if os.environ.get("OPENADNET_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync_tokenizer_embeddings(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> None:
    """Resize word embeddings when tokenizer length differs (avoids OOB indices on GPU)."""
    n_tok = len(tokenizer)
    n_emb = int(model.get_input_embeddings().weight.shape[0])
    if n_tok != n_emb:
        model.resize_token_embeddings(n_tok)


def _place_model_on_device(model: "PreTrainedModel", device: torch.device) -> torch.device:
    """Move model to ``device``; fall back to CPU if CUDA raises (bad driver, prior assert, etc.)."""
    try:
        model.to(device)
        return device
    except Exception as e:  # noqa: BLE001 — surface CUDA/Accelerator failures as CPU fallback
        if device.type != "cuda":
            raise
        warnings.warn(
            f"Failed to place model on {device} ({type(e).__name__}: {e!s}). "
            "Retrying on CPU. Set OPENADNET_FORCE_CPU=1 to skip GPU, or use device=torch.device('cpu').",
            UserWarning,
            stacklevel=3,
        )
        cpu = torch.device("cpu")
        model.to(cpu)
        return cpu


def _ensure_pad_token(tokenizer: "PreTrainedTokenizer") -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError(
                "Tokenizer has no pad_token; set pad_token or use a model that defines eos/unk."
            )


class HuggingFaceRegressor:
    """Regression using ``AutoModelForSequenceClassification`` (``problem_type='regression'``).

    **Device:** Defaults to CUDA when available. Set environment variable
    ``OPENADNET_FORCE_CPU=1`` or pass ``device=torch.device("cpu")`` if you hit
    CUDA device-side asserts (common with tokenizer/model vocab skew; we also
    call ``resize_token_embeddings`` when the tokenizer length differs from the
    loaded checkpoint).
    """

    def __init__(
        self,
        model_name_or_path: str,
        n_tasks: int = 1,
        *,
        tokenizer_name_or_path: Optional[str] = None,
        max_length: Optional[int] = 512,
        device: Optional[torch.device] = None,
    ) -> None:
        if n_tasks < 1:
            raise ValueError("n_tasks must be >= 1")
        self.n_tasks = n_tasks
        self.max_length = max_length
        self.model_name_or_path = model_name_or_path
        self.device = _default_device(device)
        tok_src = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        _ensure_pad_token(self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=n_tasks,
            problem_type="regression",
            ignore_mismatched_sizes=True,
        )
        _sync_tokenizer_embeddings(self.model, self.tokenizer)
        if self.tokenizer.pad_token_id is not None and self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.device = _place_model_on_device(self.model, self.device)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        n_tasks: int = 1,
        *,
        tokenizer_name_or_path: Optional[str] = None,
        max_length: Optional[int] = 512,
        device: Optional[torch.device] = None,
    ) -> "HuggingFaceRegressor":
        """Load tokenizer and model from a Hugging Face hub id or local directory."""
        return cls(
            model_name_or_path,
            n_tasks=n_tasks,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_length=max_length,
            device=device,
        )

    def _collate(self, return_labels: bool) -> Any:
        return smiles_regression_collate_fn(
            self.tokenizer,
            max_length=self.max_length,
            return_labels=return_labels,
        )

    def _batch_to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def fit(
        self,
        train_dataset: SmilesRegressionDataset,
        *,
        epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        val_dataset: Optional[SmilesRegressionDataset] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        show_progress: bool = True,
    ) -> List[float]:
        """Train on a :class:`~models.data.transformer.SmilesRegressionDataset`.

        When ``val_dataset`` is set, validation loss is computed after each epoch,
        the best checkpoint is restored at the end, and early stopping applies
        unless ``early_stopping_patience=0`` (``None`` defaults to patience ``10``).
        """
        if train_dataset.n_tasks != self.n_tasks:
            raise ValueError(
                f"dataset has n_tasks={train_dataset.n_tasks}, model expects {self.n_tasks}"
            )
        patience_eff = early_stopping_patience
        if val_dataset is not None and patience_eff is None:
            patience_eff = 10
        elif val_dataset is None:
            patience_eff = 0

        self.model.train()
        collate = self._collate(return_labels=True)
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        losses: List[float] = []
        best_val = float("inf")
        best_state: Optional[dict[str, torch.Tensor]] = None
        epochs_no_improve = 0
        epoch_bar = tqdm(range(epochs), disable=not show_progress, desc="epoch")
        for _ in epoch_bar:
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                batch = self._batch_to_device(batch)
                opt.zero_grad()
                out = self.model(**batch)
                loss = out.loss
                if loss is None:
                    raise RuntimeError("model returned no loss; check labels in batch")
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            losses.append(avg)
            postfix: dict[str, float] = {"loss": avg}
            if val_dataset is not None:
                vloss = self.evaluate_loss(val_dataset, batch_size=batch_size)
                postfix["val_loss"] = vloss
                if vloss < best_val - early_stopping_min_delta:
                    best_val = vloss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if patience_eff > 0 and epochs_no_improve >= patience_eff:
                    if show_progress:
                        tqdm.write(
                            f"early stopping: no val improvement for {patience_eff} epochs"
                        )
                    break
            epoch_bar.set_postfix(**postfix)

        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
        return losses

    @torch.no_grad()
    def evaluate_loss(
        self,
        dataset: SmilesRegressionDataset,
        *,
        batch_size: int = 8,
    ) -> float:
        self.model.eval()
        collate = self._collate(return_labels=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        total = 0.0
        n_batches = 0
        for batch in loader:
            batch = self._batch_to_device(batch)
            out = self.model(**batch)
            if out.loss is None:
                raise RuntimeError("model returned no loss")
            total += float(out.loss.cpu())
            n_batches += 1
        self.model.train()
        return total / max(n_batches, 1)

    @torch.no_grad()
    def predict(
        self,
        dataset: SmilesRegressionDataset,
        *,
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return predictions of shape ``(n_samples, n_tasks)``."""
        self.model.eval()
        collate = self._collate(return_labels=False)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        chunks: List[np.ndarray] = []
        it = tqdm(loader, disable=not show_progress, desc="predict")
        for batch in it:
            batch = self._batch_to_device(batch)
            out = self.model(**batch)
            logits = out.logits.detach().cpu().numpy()
            chunks.append(logits)
        self.model.train()
        return np.concatenate(chunks, axis=0)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save model and tokenizer to ``save_directory``."""
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_from_hf(
        self,
        model_name_or_path: str,
        *,
        tokenizer_name_or_path: Optional[str] = None,
    ) -> None:
        """Reload weights from a hub id or directory (same task: regression, ``n_tasks``)."""
        tok_src = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        _ensure_pad_token(self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=self.n_tasks,
            problem_type="regression",
            ignore_mismatched_sizes=True,
        )
        _sync_tokenizer_embeddings(self.model, self.tokenizer)
        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.device = _place_model_on_device(self.model, self.device)
        self.model_name_or_path = model_name_or_path
