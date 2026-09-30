"""
trainer.py
==========
Training and evaluation loop for QC-CNN-Parallel with robust batch-level
checkpointing, atomic file operations, intra-epoch resume, and walltime
budget awareness for 48-hour cluster queues (e.g. PBS Pro, OpenPBS, SLURM).

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
import sys
import json
import time
import random
import signal
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix


# ---------------------------------------------------------------------------
# Exceptions & Signals for Walltime Budget Management
# ---------------------------------------------------------------------------
class TimeBudgetExceeded(Exception):
    """Raised when training approaches the cluster queue walltime limit."""
    pass


class TimeBudgetManager:
    """
    Monitors elapsed wall-clock time and system termination signals (SIGTERM).
    Ensures safe shutdown before cluster queue limits (e.g. 48 hours) kill the job.
    """
    def __init__(self,
                 max_runtime_hours: Optional[float] = None,
                 start_time: Optional[float] = None):
        # Read from argument, or fallback to environment variables
        if max_runtime_hours is None:
            env_val = os.environ.get("MAX_HOURS") or os.environ.get("WALLTIME_HOURS")
            if env_val:
                try:
                    max_runtime_hours = float(env_val)
                except ValueError:
                    max_runtime_hours = None

        self.max_runtime_hours = max_runtime_hours
        self.max_runtime_seconds = (max_runtime_hours * 3600.0) if max_runtime_hours is not None else None
        self.start_time = start_time if start_time is not None else time.time()
        self.signal_received = False
        self.signal_name = None

        # Register signal handlers if running on an OS that supports SIGTERM (e.g. Linux on H100)
        self._register_signals()

    def _register_signals(self):
        def _handler(sig, frame):
            self.signal_received = True
            try:
                self.signal_name = signal.Signals(sig).name
            except Exception:
                self.signal_name = str(sig)
            print(f"\n  [ALERT] Caught termination signal {self.signal_name}! Preparing graceful checkpoint...",
                  file=sys.stderr, flush=True)

        for sig_name in ["SIGTERM", "SIGINT", "SIGUSR1", "SIGUSR2"]:
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _handler)
                except (ValueError, OSError):
                    pass

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def elapsed_hours(self) -> float:
        return self.elapsed_seconds / 3600.0

    def should_stop(self) -> bool:
        if self.signal_received:
            return True
        if self.max_runtime_seconds is not None:
            if self.elapsed_seconds >= self.max_runtime_seconds:
                return True
        return False


# ---------------------------------------------------------------------------
# Reproducibility & RNG State Management
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Set seeds for Python, NumPy, and PyTorch (Table 4, page 10)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    """Capture full RNG states across all libraries for exact checkpoint resume."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_state(states: Optional[Dict[str, Any]]):
    """Restore RNG states from checkpoint."""
    if not states:
        return
    try:
        if "python" in states:
            random.setstate(states["python"])
        if "numpy" in states:
            np.random.set_state(states["numpy"])
        if "torch" in states:
            torch.set_rng_state(states["torch"])
        if "cuda" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["cuda"])
    except Exception as e:
        print(f"  [WARN] Could not fully restore RNG states: {e}")


# ---------------------------------------------------------------------------
# Atomic Checkpoint Persistence
# ---------------------------------------------------------------------------
def safe_torch_save(data: Any, target_path: Path):
    """
    Atomically saves data by writing to a temporary file then renaming.
    Also preserves a .bak copy to prevent corruption if interrupted mid-write.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    bak_path = target_path.with_suffix(target_path.suffix + ".bak")

    # Save to temporary path first
    torch.save(data, tmp_path)

    # Rotate existing file to .bak for safety
    if target_path.exists():
        try:
            if bak_path.exists():
                bak_path.unlink()
            target_path.rename(bak_path)
        except Exception:
            pass

    # Atomically replace target
    tmp_path.replace(target_path)


def safe_torch_load(target_path: Path, device: torch.device) -> Dict[str, Any]:
    """
    Safely load a PyTorch checkpoint, falling back to .bak if primary is corrupted.
    """
    target_path = Path(target_path)
    bak_path = target_path.with_suffix(target_path.suffix + ".bak")

    try:
        return torch.load(target_path, map_location=device, weights_only=False)
    except Exception as e1:
        if bak_path.exists():
            print(f"  [WARN] Primary checkpoint {target_path.name} failed ({e1}). Trying backup .bak...")
            return torch.load(bak_path, map_location=device, weights_only=False)
        raise e1


# ---------------------------------------------------------------------------
# Single epoch helpers with intra-epoch batch checkpointing
# ---------------------------------------------------------------------------
def _train_epoch(model: nn.Module,
                 loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 device: torch.device,
                 start_batch_idx: int = 0,
                 initial_total_loss: float = 0.0,
                 initial_correct: int = 0,
                 initial_total: int = 0,
                 batch_checkpoint_cb: Optional[Callable[[int, float, int, int], None]] = None,
                 time_budget_mgr: Optional[TimeBudgetManager] = None) -> Tuple[float, float, bool, int, float, int, int]:
    """
    Run one training epoch with intra-epoch batch resuming and periodic saving.
    Returns:
        (avg_loss, accuracy, interrupted, last_batch_idx, running_loss, running_correct, running_total)
    """
    model.train()
    total_loss = initial_total_loss
    correct    = initial_correct
    total      = initial_total
    interrupted = False
    last_batch_idx = start_batch_idx - 1

    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        # Skip batches already completed in previous run
        if batch_idx < start_batch_idx:
            continue

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)                    # forward
        loss   = loss_fn(logits, labels)          # cross-entropy (Eq. 15)
        loss.backward()                           # param-shift via autograd
        optimizer.step()                          # Adam update (Eq. 18)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += batch_size
        last_batch_idx = batch_idx

        # Intra-epoch periodic checkpoint callback
        if batch_checkpoint_cb is not None:
            batch_checkpoint_cb(batch_idx, total_loss, correct, total)

        # Check if 48-hour walltime limit or SIGTERM has arrived
        if time_budget_mgr is not None and time_budget_mgr.should_stop():
            interrupted = True
            print(f"\n  [WALLTIME] Budget limit reached or signal caught during epoch! "
                  f"Paused at batch {batch_idx + 1}/{total_batches}.")
            break

    avg_loss = total_loss / max(total, 1)
    acc      = correct / max(total, 1)
    return avg_loss, acc, interrupted, last_batch_idx, total_loss, correct, total


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

    return total_loss / max(len(all_labels), 1), accuracy, macro_f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main training loop with robust batch-level checkpointing
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
          resume: bool = True,
          checkpoint_interval_batches: int = 25,
          checkpoint_interval_seconds: float = 900.0,
          max_runtime_hours: Optional[float] = None,
          time_budget_mgr: Optional[TimeBudgetManager] = None) -> Tuple[Dict, Dict, nn.Module]:
    """
    Full training loop matching paper Table 4 with high-reliability checkpointing:
      - Optimizer : Adam, lr=0.01
      - Loss      : CrossEntropyLoss
      - Epochs    : 50
      - Seed      : 42

    Features:
      - Intra-epoch batch-level checkpointing every `checkpoint_interval_batches`
        or `checkpoint_interval_seconds`.
      - Atomic checkpoint persistence (zero risk of file corruption).
      - Walltime budget detection (automatically saves and halts when 48h limit nears).
      - Seamless automatic continuation on subsequent cluster jobs.

    Returns
    -------
    history : dict with lists of per-epoch train/test loss and accuracy
    summary : dict with final metrics
    model   : trained PyTorch model with restored best weights
    """
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if time_budget_mgr is None:
        time_budget_mgr = TimeBudgetManager(max_runtime_hours=max_runtime_hours)

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
    start_batch_idx = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss" : [], "test_acc" : [], "test_f1"  : [],
        "epoch_time": [],
    }
    best_acc     = 0.0
    best_weights = None

    if resume and (checkpoint_path.exists() or checkpoint_path.with_suffix(".pt.bak").exists()):
        try:
            ckpt = safe_torch_load(checkpoint_path, device=device)
            saved_epoch = ckpt.get("epoch", 1)
            saved_batch = ckpt.get("batch_idx", -1)

            history      = ckpt.get("history", history)
            best_acc     = ckpt.get("best_acc", 0.0)
            best_weights = ckpt.get("best_weights", None)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            restore_rng_state(ckpt.get("rng_state"))

            if saved_batch >= 0 and saved_batch < len(train_loader) - 1:
                # Interrupted mid-epoch at saved_batch: resume within that epoch
                start_epoch     = saved_epoch
                start_batch_idx = saved_batch + 1
                running_loss    = ckpt.get("running_loss", 0.0)
                running_correct = ckpt.get("running_correct", 0)
                running_total   = ckpt.get("running_total", 0)
                print(f"\n  ⟳ Resuming from intra-epoch batch checkpoint:")
                print(f"    Epoch {start_epoch}/{num_epochs}, continuing from batch {start_batch_idx + 1}/{len(train_loader)}")
                print(f"    (best_acc so far: {best_acc:.4f})")
            else:
                # Epoch was fully completed
                start_epoch     = saved_epoch + 1
                start_batch_idx = 0
                running_loss    = 0.0
                running_correct = 0
                running_total   = 0
                print(f"\n  ⟳ Resuming from epoch checkpoint — starting epoch {start_epoch}/{num_epochs}")
                print(f"    (best_acc so far: {best_acc:.4f})")

        except Exception as e:
            print(f"\n  ⚠ Failed to load checkpoint ({e}). Starting fresh.")
            start_epoch = 1
            start_batch_idx = 0
            running_loss = 0.0
            running_correct = 0
            running_total = 0

    # If all epochs are already done, finalize and return
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
            "total_train_time_s": sum(history.get("epoch_time", [])),
        }
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        bak_file = checkpoint_path.with_suffix(".pt.bak")
        if bak_file.exists():
            bak_file.unlink()
        return history, summary, model

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}  |  Dataset: {dataset_name}")
    print(f"  Epochs={num_epochs}, LR={lr}, Seed={seed}, Device={device}")
    if time_budget_mgr.max_runtime_hours:
        print(f"  Walltime budget: {time_budget_mgr.max_runtime_hours:.2f} hours (Cluster queue safe)")
    if start_epoch > 1 or start_batch_idx > 0:
        print(f"  Resuming at epoch {start_epoch}, batch {start_batch_idx}")
    print(f"{'='*60}")

    last_checkpoint_time = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        curr_start_batch = start_batch_idx if epoch == start_epoch else 0
        curr_run_loss    = running_loss if epoch == start_epoch else 0.0
        curr_run_corr    = running_correct if epoch == start_epoch else 0
        curr_run_tot     = running_total if epoch == start_epoch else 0

        # Helper callback for intra-epoch batch checkpointing
        def batch_save_cb(b_idx: int, cur_loss: float, cur_corr: int, cur_tot: int):
            nonlocal last_checkpoint_time
            now = time.time()
            time_due = (now - last_checkpoint_time) >= checkpoint_interval_seconds
            step_due = ((b_idx + 1) % checkpoint_interval_batches == 0)

            if step_due or time_due:
                last_checkpoint_time = now
                ckpt_data = {
                    "epoch": epoch,
                    "batch_idx": b_idx,
                    "running_loss": cur_loss,
                    "running_correct": cur_corr,
                    "running_total": cur_tot,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": capture_rng_state(),
                    "history": history,
                    "best_acc": best_acc,
                    "best_weights": best_weights,
                    "timestamp": now,
                }
                safe_torch_save(ckpt_data, checkpoint_path)

        tr_loss, tr_acc, interrupted, last_b_idx, run_l, run_c, run_t = _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            start_batch_idx=curr_start_batch,
            initial_total_loss=curr_run_loss,
            initial_correct=curr_run_corr,
            initial_total=curr_run_tot,
            batch_checkpoint_cb=batch_save_cb,
            time_budget_mgr=time_budget_mgr,
        )

        if interrupted:
            # Save intra-epoch checkpoint immediately upon walltime limit / interruption
            ckpt_data = {
                "epoch": epoch,
                "batch_idx": last_b_idx,
                "running_loss": run_l,
                "running_correct": run_c,
                "running_total": run_t,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": capture_rng_state(),
                "history": history,
                "best_acc": best_acc,
                "best_weights": best_weights,
                "timestamp": time.time(),
            }
            safe_torch_save(ckpt_data, checkpoint_path)
            print(f"\n  [CHECKPOINT SAVED] Interrupted at Epoch {epoch}, Batch {last_b_idx + 1}/{len(train_loader)}.")
            print(f"  Elapsed wall-clock time: {time_budget_mgr.elapsed_hours:.2f}h.")
            print(f"  State safely persisted to {checkpoint_path}.")
            raise TimeBudgetExceeded(
                f"Cluster walltime limit reached ({time_budget_mgr.elapsed_hours:.2f}h). "
                f"Checkpoint saved at Epoch {epoch}, Batch {last_b_idx + 1}."
            )

        # Full epoch evaluation
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
            safe_torch_save(best_weights, save_path / f"{model_name}_best.pt")

        # Save completed epoch checkpoint (batch_idx = -1 indicates full epoch completed)
        ckpt_data = {
            "epoch": epoch,
            "batch_idx": -1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "history": history,
            "best_acc": best_acc,
            "best_weights": best_weights,
            "timestamp": time.time(),
        }
        safe_torch_save(ckpt_data, checkpoint_path)

        # Check if time budget is exhausted after full epoch evaluation
        if time_budget_mgr.should_stop():
            print(f"\n  [WALLTIME] Budget limit reached after epoch {epoch}!")
            print(f"  Elapsed: {time_budget_mgr.elapsed_hours:.2f}h. Checkpoint saved.")
            raise TimeBudgetExceeded(
                f"Cluster walltime limit reached after epoch {epoch} ({time_budget_mgr.elapsed_hours:.2f}h)."
            )

    # Restore best weights and run final evaluation
    if best_weights is not None:
        model.load_state_dict(best_weights)
    fin_loss, fin_acc, fin_f1, preds, labels = _eval_epoch(
        model, test_loader, loss_fn, device)
    cm = confusion_matrix(labels, preds)

    # Save final results atomically
    history_file = save_path / f"{model_name}_history.json"
    with open(history_file, "w") as f:
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
    bak_file = checkpoint_path.with_suffix(".pt.bak")
    if bak_file.exists():
        bak_file.unlink()
    print(f"  ✓ Checkpoint cleaned up (training complete)")

    print(f"\n  ✓ Best test accuracy : {best_acc:.4f}")
    print(f"  ✓ Final macro-F1     : {fin_f1:.4f}")
    print(f"  ✓ Results saved to   : {save_path}")

    return history, summary, model
