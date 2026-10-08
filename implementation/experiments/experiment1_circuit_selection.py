"""
experiment1_circuit_selection.py
=================================
Experiment 1: Analysis of the Performance Indicators of Different Quantum Circuits
(Paper Section 4.3.1, Tables 2 and 3)

Reproduces:
  Table 2 — Expressibility, Entangling Capability, Discreteness for 11 circuits
  Table 3 — Classification accuracy of each circuit on MNIST and Fashion-MNIST

The paper's methodology:
  • Each circuit uses 4 qubits and is evaluated with 5,000 numerical simulations.
  • A simplified hybrid model (1 quantum conv layer + 1 linear layer) is used
    to isolate the PQC contribution.
  • The winning circuit (Circuit 11, 16 params) is selected based on balancing
    expressibility, discreteness, and entangling capability.

Usage:
  python experiments/experiment1_circuit_selection.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
from pathlib import Path

from models.quantum_circuit import NUM_QUBITS
from datasets.dataloader    import get_mnist, get_fashion_mnist
from training.trainer import (
    set_seed, _train_epoch, _eval_epoch,
    TimeBudgetManager, TimeBudgetExceeded,
    safe_torch_save, safe_torch_load,
    capture_rng_state, restore_rng_state,
)
from utils.circuit_metrics  import run_circuit_analysis


# ---------------------------------------------------------------------------
# Hyper-parameters (Table 4 / Section 4.3.1)
# ---------------------------------------------------------------------------
SEED          = 42
LR            = 0.01
BATCH_SIZE    = 32
NUM_EPOCHS    = 70      # same 70 epochs as main experiment
N_SIMS        = 5000   # circuit analysis: 5,000 simulations
RESULTS_DIR   = Path("results/experiment1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
dev = qml.device("default.qubit", wires=NUM_QUBITS)


# ---------------------------------------------------------------------------
# Circuit factory helpers
# ---------------------------------------------------------------------------
def _encode(inputs):
    """Angle encoding (H then RY, paper Eq. 6)."""
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)
        qml.RY(inputs[i] * np.pi, wires=i)   # scale by π


def _rx_linear(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RX(weights[i], wires=i)
    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _rx_circle(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RX(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _rx_alltoall(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RX(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        for j in range(NUM_QUBITS):
            if i != j:
                qml.CNOT(wires=[i, j])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _ry_linear(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RY(weights[i], wires=i)
    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _ry_circle(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RY(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _ry_alltoall(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RY(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        for j in range(NUM_QUBITS):
            if i != j:
                qml.CNOT(wires=[i, j])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _rz_linear(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RZ(weights[i], wires=i)
    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _rz_circle(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RZ(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _rz_alltoall(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RZ(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        for j in range(NUM_QUBITS):
            if i != j:
                qml.CNOT(wires=[i, j])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _circuit10(inputs, weights):
    """
    Circuit 10 (Figure 5a, Equations 19–21): RX+RZ rotations + all-to-all CRX.
    28 parameters: 2 rot layers × 4 qubits × 2 gates + 4×(4-1) CRX = 16+12=28.
    """
    _encode(inputs)
    # Variational layer 1: RX + RZ per qubit (8 params)
    for i in range(NUM_QUBITS):
        qml.RX(weights[i],              wires=i)
        qml.RZ(weights[NUM_QUBITS + i], wires=i)
    # Entangling: all ordered pairs (i,j), i≠j  (12 CRX params for 4 qubits)
    idx = 8
    for i in range(NUM_QUBITS):
        for j in range(NUM_QUBITS):
            if i != j:
                qml.CRX(weights[idx], wires=[i, j])
                idx += 1
    # Variational layer 2: RX + RZ per qubit (8 params)
    for i in range(NUM_QUBITS):
        qml.RX(weights[idx],     wires=i); idx += 1
        qml.RZ(weights[idx],     wires=i); idx += 1
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _circuit11(inputs, weights):
    """
    Circuit 11 — the SELECTED circuit (Figure 5b, Equations 22–24).
    16 parameters: 2 rot layers × 4 RY + 2 ent layers × 4 CRX.
    """
    _encode(inputs)
    w_rot1, w_ent1, w_rot2, w_ent2 = (
        weights[0:4], weights[4:8], weights[8:12], weights[12:16]
    )
    # Rot layer 1
    for i in range(NUM_QUBITS):
        qml.RY(w_rot1[i], wires=i)
    # Ent layer 1: circle 0→1→2→3→0
    for i in range(NUM_QUBITS):
        qml.CRX(w_ent1[i], wires=[i, (i + 1) % NUM_QUBITS])
    # Rot layer 2
    for i in range(NUM_QUBITS):
        qml.RY(w_rot2[i], wires=i)
    # Ent layer 2: circle shifted 1→2→3→0→1
    for i in range(NUM_QUBITS):
        qml.CRX(w_ent2[i], wires=[(i + 1) % NUM_QUBITS, (i + 2) % NUM_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


# ---------------------------------------------------------------------------
# Build QNodes
# ---------------------------------------------------------------------------
CIRCUIT_DEFS = {
    "RX-Linear"    : (_rx_linear,    4),
    "RX-Circle"    : (_rx_circle,    4),
    "RX-All-to-All": (_rx_alltoall,  4),
    "RY-Linear"    : (_ry_linear,    4),
    "RY-Circle"    : (_ry_circle,    4),
    "RY-All-to-All": (_ry_alltoall,  4),
    "RZ-Linear"    : (_rz_linear,    4),
    "RZ-Circle"    : (_rz_circle,    4),
    "RZ-All-to-All": (_rz_alltoall,  4),
    "Circuit-10"   : (_circuit10,   28),
    "Circuit-11"   : (_circuit11,   16),
}

QNODES = {
    name: (qml.QNode(fn, dev, interface="torch"), n_params)
    for name, (fn, n_params) in CIRCUIT_DEFS.items()
}


# ---------------------------------------------------------------------------
# Simplified hybrid model for Experiment 1
# (one quantum conv layer + one linear classifier)
# ---------------------------------------------------------------------------
class SimpleHybrid(nn.Module):
    """
    Minimal hybrid model to isolate the PQC contribution (Section 4.3.1).
    Quantum conv (Circuit X) → flatten → Linear(4×14×14, num_classes)
    """
    def __init__(self, qnode, num_params: int, num_classes: int = 10):
        super().__init__()
        self.weights    = nn.Parameter(torch.randn(num_params) * 0.01)
        self._qnode     = qnode
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        out_H, out_W = H // 2, W // 2
        out = torch.zeros((B, 4, out_H, out_W), device=x.device, dtype=x.dtype)
        for b in range(B):
            for i in range(out_H):
                for j in range(out_W):
                    patch = x[b, 0, i*2:i*2+2, j*2:j*2+2].flatten()
                    result = torch.stack(self._qnode(patch, self.weights))
                    out[b, :, i, j] = result
        flat   = out.view(B, -1)
        return self.classifier(flat)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_experiment1(run_metrics: bool = True, n_sims: int = N_SIMS,
                    resume: bool = True,
                    max_runtime_hours: float | None = None,
                    time_budget_mgr: TimeBudgetManager | None = None,
                    checkpoint_interval_batches: int = 25):
    set_seed(SEED)
    device = torch.device("cpu")

    if time_budget_mgr is None:
        time_budget_mgr = TimeBudgetManager(max_runtime_hours=max_runtime_hours)

    print("=" * 60)
    print("  Experiment 1: PQC Selection Study  (Table 2 & 3)")
    if time_budget_mgr.max_runtime_hours:
        print(f"  Walltime budget: {time_budget_mgr.max_runtime_hours:.2f} hours (Cluster queue safe)")
    print("=" * 60)

    # Table 2 is the independent numerical circuit study
    table2_path = RESULTS_DIR / "table2_circuit_metrics.json"
    table2_results = {}
    if run_metrics:
        if resume and table2_path.exists():
            print("  ⟳ Table 2 metrics already computed — loading from checkpoint.")
            with open(table2_path, "r") as f:
                table2_results = json.load(f)
        else:
            circuits = {
                name: {"fn": fn, "num_params": n_params}
                for name, (fn, n_params) in CIRCUIT_DEFS.items()
            }
            table2_results = run_circuit_analysis(circuits, n_sims=n_sims)
            with open(table2_path, "w") as f:
                json.dump(table2_results, f, indent=2)

    if time_budget_mgr.should_stop():
        raise TimeBudgetExceeded("Time budget reached after Table 2 metrics calculation.")

    train_mnist,   test_mnist   = get_mnist(batch_size=BATCH_SIZE)
    train_fashion, test_fashion = get_fashion_mnist(batch_size=BATCH_SIZE)

    # Resume: load partial Table 3 results if they exist
    table3_path = RESULTS_DIR / "table3_circuit_classification.json"
    results_table3 = {}
    if resume and table3_path.exists():
        try:
            with open(table3_path, "r") as f:
                results_table3 = json.load(f)
            if results_table3:
                completed_count = sum(len(v) for v in results_table3.values() if isinstance(v, dict))
                print(f"  ⟳ Loaded checkpoint: {completed_count} circuit-dataset benchmark(s) completed.")
        except Exception:
            results_table3 = {}

    for circuit_name, (qnode, n_params) in QNODES.items():
        if circuit_name not in results_table3:
            results_table3[circuit_name] = {}

        # If both datasets already evaluated for this circuit, skip
        if (resume and "MNIST" in results_table3[circuit_name]
                and "Fashion-MNIST" in results_table3[circuit_name]):
            print(f"\n--- Circuit: {circuit_name} — ✓ already completed, skipping ---")
            continue

        print(f"\n--- Circuit: {circuit_name} ({n_params} params) ---")

        for ds_name, (tr_loader, te_loader) in [
            ("MNIST", (train_mnist, test_mnist)),
            ("Fashion-MNIST", (train_fashion, test_fashion)),
        ]:
            if resume and ds_name in results_table3[circuit_name]:
                print(f"  ✓ {ds_name} already completed: best_acc = {results_table3[circuit_name][ds_name]}")
                continue

            ckpt_file = RESULTS_DIR / f"ckpt_{circuit_name}_{ds_name}.pt"
            set_seed(SEED)
            model   = SimpleHybrid(qnode, n_params).to(device)
            opt     = torch.optim.Adam(model.parameters(), lr=LR)
            loss_fn = nn.CrossEntropyLoss()

            start_epoch = 1
            best_acc = 0.0

            if resume and ckpt_file.exists():
                try:
                    cdata = safe_torch_load(ckpt_file, device=device)
                    start_epoch = cdata.get("epoch", 0) + 1
                    best_acc = cdata.get("best_acc", 0.0)
                    model.load_state_dict(cdata["model_state_dict"])
                    opt.load_state_dict(cdata["opt_state_dict"])
                    restore_rng_state(cdata.get("rng_state"))
                    print(f"  ⟳ Resumed {circuit_name} on {ds_name} at epoch {start_epoch} (best: {best_acc:.4f})")
                except Exception as ex:
                    print(f"  ⚠ Failed to load {ckpt_file.name} ({ex}). Starting fresh.")
                    start_epoch = 1
                    best_acc = 0.0

            for epoch in range(start_epoch, NUM_EPOCHS + 1):
                tr_loss, tr_acc, interrupted, last_b, _, _, _ = _train_epoch(
                    model=model, loader=tr_loader, optimizer=opt, loss_fn=loss_fn,
                    device=device, time_budget_mgr=time_budget_mgr,
                )

                if interrupted:
                    safe_torch_save({
                        "epoch": epoch,
                        "best_acc": best_acc,
                        "model_state_dict": model.state_dict(),
                        "opt_state_dict": opt.state_dict(),
                        "rng_state": capture_rng_state(),
                    }, ckpt_file)
                    print(f"\n  [CHECKPOINT] Circuit {circuit_name} on {ds_name} paused at epoch {epoch}.")
                    raise TimeBudgetExceeded(
                        f"Walltime limit reached on Circuit {circuit_name} ({ds_name}, Epoch {epoch})."
                    )

                _, acc, _, _, _ = _eval_epoch(model, te_loader, loss_fn, device)
                if acc > best_acc:
                    best_acc = acc

                print(f"  [{circuit_name}|{ds_name}] Epoch {epoch}/{NUM_EPOCHS} "
                      f"— Loss: {tr_loss:.4f}, Val Acc: {acc:.4f} (Best: {best_acc:.4f})", flush=True)

                # Save epoch checkpoint
                safe_torch_save({
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "model_state_dict": model.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                    "rng_state": capture_rng_state(),
                }, ckpt_file)
                import gc; gc.collect()

                if time_budget_mgr.should_stop():
                    raise TimeBudgetExceeded(
                        f"Walltime limit reached after epoch {epoch} of {circuit_name} on {ds_name}."
                    )

            results_table3[circuit_name][ds_name] = round(best_acc, 4)
            print(f"  {ds_name}: best_acc = {best_acc:.4f}")

            # Save Table 3 atomically
            tmp_t3 = table3_path.with_suffix(".json.tmp")
            with open(tmp_t3, "w") as f:
                json.dump(results_table3, f, indent=2)
            tmp_t3.replace(table3_path)

            # Clean up per-dataset checkpoint
            if ckpt_file.exists():
                ckpt_file.unlink()
            bak_f = ckpt_file.with_suffix(".pt.bak")
            if bak_f.exists():
                bak_f.unlink()

    # Print comparison against paper Table 3
    print("\n" + "=" * 60)
    print("  Table 3 Results vs Paper")
    print("=" * 60)
    paper_table3 = {
        "RX-Linear":     {"MNIST": 0.6560, "Fashion-MNIST": 0.7364},
        "RX-Circle":     {"MNIST": 0.6066, "Fashion-MNIST": 0.6864},
        "RX-All-to-All": {"MNIST": 0.7495, "Fashion-MNIST": 0.7647},
        "RY-Linear":     {"MNIST": 0.7530, "Fashion-MNIST": 0.7626},
        "RY-Circle":     {"MNIST": 0.7602, "Fashion-MNIST": 0.7763},
        "RY-All-to-All": {"MNIST": 0.8006, "Fashion-MNIST": 0.7846},
        "RZ-Linear":     {"MNIST": 0.5496, "Fashion-MNIST": 0.6229},
        "RZ-Circle":     {"MNIST": 0.5481, "Fashion-MNIST": 0.6213},
        "RZ-All-to-All": {"MNIST": 0.7249, "Fashion-MNIST": 0.6881},
        "Circuit-10":    {"MNIST": 0.8254, "Fashion-MNIST": 0.7946},
        "Circuit-11":    {"MNIST": 0.8057, "Fashion-MNIST": 0.7778},
    }
    print(f"{'Circuit':<20} {'MNIST (paper)':>14} {'MNIST (ours)':>14} "
          f"{'FashMNIST (paper)':>18} {'FashMNIST (ours)':>18}")
    print("-" * 90)
    for name in results_table3:
        p  = paper_table3.get(name, {})
        r  = results_table3[name]
        print(f"{name:<20} {p.get('MNIST',0):>14.4f} {r.get('MNIST',0):>14.4f} "
              f"{p.get('Fashion-MNIST',0):>18.4f} {r.get('Fashion-MNIST',0):>18.4f}")

    return {"table2": table2_results, "table3": results_table3}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Skip the 5,000-simulation Table 2 analysis.")
    parser.add_argument("--n-sims", type=int, default=N_SIMS,
                        help="Simulations per Table 2 metric (paper: 5000).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore checkpoints and start from scratch.")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Maximum hours before pausing and saving checkpoint (for 48h queues).")
    parser.add_argument("--checkpoint-interval-batches", type=int, default=25,
                        help="Frequency of intra-epoch batch checkpointing.")
    args = parser.parse_args()
    run_experiment1(
        run_metrics=not args.skip_metrics,
        n_sims=args.n_sims,
        resume=not args.no_resume,
        max_runtime_hours=args.max_hours,
        checkpoint_interval_batches=args.checkpoint_interval_batches,
    )
