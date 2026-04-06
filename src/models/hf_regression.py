"""Hugging Face transformers for regression-only finetuning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.data.transformer import SmilesRegressionDataset, smiles_regression_collate_fn

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


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
    """Regression using ``AutoModelForSequenceClassification`` (``problem_type='regression'``)."""

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
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        tok_src = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        _ensure_pad_token(self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=n_tasks,
            problem_type="regression",
            ignore_mismatched_sizes=True,
        )
        if self.tokenizer.pad_token_id is not None and self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)

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
        show_progress: bool = True,
    ) -> List[float]:
        """Train on a :class:`~models.data.transformer.SmilesRegressionDataset`."""
        if train_dataset.n_tasks != self.n_tasks:
            raise ValueError(
                f"dataset has n_tasks={train_dataset.n_tasks}, model expects {self.n_tasks}"
            )
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
        epoch_bar = tqdm(range(epochs), disable=not show_progress, desc="epoch")
        for _ in epoch_bar:
            batch_bar = tqdm(loader, disable=not show_progress, leave=False, desc="train")
            epoch_loss = 0.0
            n_batches = 0
            for batch in batch_bar:
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
            epoch_bar.set_postfix(loss=avg)

        if val_dataset is not None:
            self.model.eval()
            vloss = self.evaluate_loss(val_dataset, batch_size=batch_size)
            if show_progress:
                tqdm.write(f"validation loss: {vloss:.6f}")
            self.model.train()
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
        self.model.to(self.device)
        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model_name_or_path = model_name_or_path
