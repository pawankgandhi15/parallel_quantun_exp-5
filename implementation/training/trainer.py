"""
trainer.py
==========
Training and evaluation loop for QC-CNN-Parallel.

Paper: "A Parallel Hybrid Quantum-Classical Convolutional Design Using
        Parameterized Quantum Circuits for Image Classification"
        Quantum Engineering (2026), article 6643049.

References
----------
- Table 4 (page 10)   : Optimizer=Adam, LR=0.01, Seed=42, Epochs=50, Batch=32
- Section 3.5         : Learning process / gradient update (Eqs. 15–18)
- EXPERIMENT_SETUP.md : Full training protocol
- RESULTS.md          : Metrics to record (accuracy, loss, macro-F1)
"""

from __future__ import annotations

import os
import json
import time
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix


# ---------------------------------------------------------------------------
# Reproducibility  (Table 4: seed = 42)
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Set seeds for Python, NumPy, and PyTorch (Table 4, page 10)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------
def _train_epoch(model: nn.Module,
                 loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 device: torch.device) -> Tuple[float, float]:
    """Run one training epoch.  Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)                    # forward
        loss   = loss_fn(logits, labels)          # cross-entropy (Eq. 15)
        loss.backward()                           # param-shift via autograd
        optimizer.step()                          # Adam update (Eq. 18)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def _eval_epoch(model: nn.Module,
                loader: DataLoader,
                loss_fn: nn.Module,
                device: torch.device) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate model.  Returns (avg_loss, accuracy, macro_f1, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = loss_fn(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy   = (all_preds == all_labels).mean()
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return total_loss / len(all_labels), accuracy, macro_f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(model: nn.Module,
          train_loader: DataLoader,
          test_loader: DataLoader,
          num_epochs: int = 50,
          lr: float = 0.01,
          seed: int = 42,
          device: Optional[torch.device] = None,
          save_dir: str = "results",
          model_name: str = "qc_cnn_parallel",
          dataset_name: str = "mnist",
          resume: bool = True) -> Dict:
    """
    Full training loop matching paper Table 4:
      - Optimizer : Adam, lr=0.01
      - Loss      : CrossEntropyLoss
      - Epochs    : 50
      - Seed      : 42

    Supports checkpoint-based resume: if training is interrupted (error,
    disconnection, crash), re-running the same command will automatically
    resume from the last completed epoch instead of starting over.

    Parameters
    ----------
    model        : instantiated model (QCCNNParallel or ClassicalCNN)
    train_loader : DataLoader for training split
    test_loader  : DataLoader for test split
    num_epochs   : default 50 (Table 4)
    lr           : learning rate, default 0.01 (Table 4)
    seed         : random seed, default 42 (Table 4)
    device       : CPU or CUDA (auto-detected if None)
    save_dir     : directory to save checkpoints and results
    model_name   : identifier string for file names
    dataset_name : dataset identifier for file names
    resume       : if True, resume from checkpoint if one exists (default True)

    Returns
    -------
    history : dict with lists of per-epoch train/test loss and accuracy
    """
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    save_path = Path(save_dir) / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_path / f"{model_name}_checkpoint.pt"

    # ------------------------------------------------------------------
    # Resume from checkpoint if available
    # ------------------------------------------------------------------
    start_epoch = 1
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss" : [], "test_acc" : [], "test_f1"  : [],
        "epoch_time": [],
    }
    best_acc     = 0.0
    best_weights = None

    if resume and checkpoint_path.exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location=device,
                              weights_only=False)
            start_epoch  = ckpt["epoch"] + 1
            history      = ckpt["history"]
            best_acc     = ckpt["best_acc"]
            best_weights = ckpt["best_weights"]
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print(f"\n  ⟳ Resuming from checkpoint — epoch {start_epoch}/{num_epochs}")
            print(f"    (best_acc so far: {best_acc:.4f})")
        except Exception as e:
            print(f"\n  ⚠ Failed to load checkpoint ({e}). Starting from scratch.")
            start_epoch = 1
            history = {
                "train_loss": [], "train_acc": [],
                "test_loss" : [], "test_acc" : [], "test_f1"  : [],
                "epoch_time": [],
            }
            best_acc     = 0.0
            best_weights = None

    # If all epochs are already done, skip training entirely
    if start_epoch > num_epochs:
        print(f"\n  ✓ Training already completed ({num_epochs} epochs). Skipping.")
        if best_weights is not None:
            model.load_state_dict(best_weights)
        fin_loss, fin_acc, fin_f1, preds, labels = _eval_epoch(
            model, test_loader, loss_fn, device)
        cm = confusion_matrix(labels, preds)
        summary = {
            "model"       : model_name,
            "dataset"     : dataset_name,
            "best_test_acc": float(best_acc),
            "final_test_acc": float(fin_acc),
            "final_test_f1" : float(fin_f1),
            "final_test_loss": float(fin_loss),
            "total_train_time_s": sum(history["epoch_time"]),
        }
        # Clean up checkpoint since training is done
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        return history, summary, model

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}  |  Dataset: {dataset_name}")
    print(f"  Epochs={num_epochs}, LR={lr}, Seed={seed}, Device={device}")
    if start_epoch > 1:
        print(f"  Resuming from epoch {start_epoch}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, loss_fn, device)
        te_loss, te_acc, te_f1, _, _ = _eval_epoch(model, test_loader, loss_fn, device)

        epoch_t = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["train_acc" ].append(tr_acc)
        history["test_loss" ].append(te_loss)
        history["test_acc"  ].append(te_acc)
        history["test_f1"   ].append(te_f1)
        history["epoch_time"].append(epoch_t)

        print(f"  Epoch {epoch:3d}/{num_epochs} | "
              f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Test  loss={te_loss:.4f} acc={te_acc:.4f} F1={te_f1:.4f} | "
              f"Time={epoch_t:.1f}s")

        # Save best model weights
        if te_acc > best_acc:
            best_acc     = te_acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_weights,
                       save_path / f"{model_name}_best.pt")

        # ------------------------------------------------------------------
        # Save epoch checkpoint (enables resume after crash/disconnect)
        # ------------------------------------------------------------------
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history":              history,
            "best_acc":             best_acc,
            "best_weights":         best_weights,
        }, checkpoint_path)

    # Restore best weights and run final evaluation
    model.load_state_dict(best_weights)
    fin_loss, fin_acc, fin_f1, preds, labels = _eval_epoch(
        model, test_loader, loss_fn, device)
    cm = confusion_matrix(labels, preds)

    # Save results
    with open(save_path / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    np.save(save_path / f"{model_name}_confusion_matrix.npy", cm)

    summary = {
        "model"       : model_name,
        "dataset"     : dataset_name,
        "best_test_acc": float(best_acc),
        "final_test_acc": float(fin_acc),
        "final_test_f1" : float(fin_f1),
        "final_test_loss": float(fin_loss),
        "total_train_time_s": sum(history["epoch_time"]),
    }
    with open(save_path / f"{model_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Clean up checkpoint — training completed successfully
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  ✓ Checkpoint cleaned up (training complete)")

    print(f"\n  ✓ Best test accuracy : {best_acc:.4f}")
    print(f"  ✓ Final macro-F1     : {fin_f1:.4f}")
    print(f"  ✓ Results saved to   : {save_path}")

    return history, summary, model
