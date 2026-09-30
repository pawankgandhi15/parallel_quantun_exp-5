"""
run_all.py
==========
Master entry point to run all five paper experiments with automated
checkpoint management, intra-epoch batch resuming, and walltime budget
orchestration for 48-hour cluster queues (PBS Pro, OpenPBS, SLURM).

Paper: "A Parallel Hybrid Quantum-Classical Convolutional Design Using
        Parameterized Quantum Circuits for Image Classification"
        Quantum Engineering (2026), article 6643049.

Usage:
  # Run Experiment 2 only (classification)
  python run_all.py --exp 2 --dataset mnist

  # Run all experiments sequentially under 48h queue (safely saves and pauses at 47h)
  python run_all.py --exp 1 2 3 4 5 --max_hours 47.0

  # Run with custom batch checkpoint interval
  python run_all.py --exp 2 --checkpoint_interval_batches 20

  # Smoke test (model & gradient check)
  python run_all.py --smoke_test
"""

from __future__ import annotations

import argparse
import sys
import os
import json
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))

from training.trainer import (
    TimeBudgetManager,
    TimeBudgetExceeded,
)


# ---------------------------------------------------------------------------
# Master Progress Manifest
# ---------------------------------------------------------------------------
class MasterProgressTracker:
    """Tracks completion of entire experiment suite across multiple cluster job cycles."""
    def __init__(self, status_path: Path):
        self.status_path = status_path
        self.data = self._load()

    def _load(self) -> dict:
        if self.status_path.exists():
            try:
                with open(self.status_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed_experiments": [], "history": []}

    def is_completed(self, exp_num: int) -> bool:
        return exp_num in self.data.get("completed_experiments", [])

    def mark_completed(self, exp_num: int):
        completed = set(self.data.get("completed_experiments", []))
        completed.add(exp_num)
        self.data["completed_experiments"] = sorted(list(completed))
        self.data["history"].append({
            "exp": exp_num,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self.save()

    def save(self):
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        tmp.replace(self.status_path)


# ---------------------------------------------------------------------------
# Smoke test (no real data needed — verifies model and gradient flow)
# ---------------------------------------------------------------------------
def run_smoke_test():
    """Verify the model forward and backward pass on dummy data."""
    from models import QCCNNParallel, ClassicalCNN
    from models.quantum_circuit import make_noisy_circuit

    print("\n" + "=" * 60)
    print("  SMOKE TEST — verifying model, optimizer, gradient flow")
    print("=" * 60)

    device  = torch.device("cpu")
    B       = 2       # tiny batch
    images  = torch.rand(B, 1, 28, 28)
    labels  = torch.randint(0, 10, (B,))
    loss_fn = torch.nn.CrossEntropyLoss()

    for name, model in [("QCCNNParallel", QCCNNParallel(10)),
                         ("ClassicalCNN",  ClassicalCNN(10))]:
        model = model.to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=0.01)
        opt.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ {name:<20}  loss={loss.item():.4f}  total_params={params:,}")

    # Noisy circuit test
    print("\n  Testing noisy circuits (default.mixed)...")
    for nt in ["bit_flip", "phase_flip", "depolarizing"]:
        qnode  = make_noisy_circuit(nt, 0.1)
        model  = QCCNNParallel(10, qnode=qnode)
        logits = model(images[:1])           # batch=1 for mixed simulator
        loss   = loss_fn(logits, labels[:1])
        print(f"  ✓ Noisy ({nt:<14})  loss={loss.item():.4f}")

    print("\n  ✓ All smoke tests passed!\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="QC-CNN-Parallel: run paper experiments"
    )
    parser.add_argument(
        "--exp", nargs="+", type=int, default=[2],
        choices=[1, 2, 3, 4, 5],
        help="Which experiments to run (default: 2)",
    )
    parser.add_argument(
        "--dataset", nargs="+",
        default=["mnist", "fashion_mnist", "overhead_mnist"],
        choices=["mnist", "fashion_mnist", "overhead_mnist"],
        help="Datasets for Experiment 2 (default: all three)",
    )
    parser.add_argument(
        "--dataset_exp4", nargs="+",
        default=["mnist", "fashion_mnist", "overhead_mnist"],
        choices=["mnist", "fashion_mnist", "overhead_mnist"],
        help="Datasets for Experiment 4 ablation study (default: all three)",
    )
    parser.add_argument(
        "--part", nargs="+", default=["A", "B"],
        choices=["A", "B"],
        help="Parts of Experiment 5 to run: A=qubit sweep, B=depth sweep (default: both)",
    )
    parser.add_argument(
        "--dataset_exp5", default="mnist",
        choices=["mnist", "fashion_mnist", "overhead_mnist"],
        help="Dataset for Experiment 5 scalability study (default: mnist)",
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run a quick smoke test to verify installation and model correctness",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Ignore all checkpoints and start experiments from scratch",
    )
    parser.add_argument(
        "--max_hours", type=float, default=47.0,
        help="Walltime budget in hours before saving and pausing for cluster re-submission (default: 47.0 for 48h queues)",
    )
    parser.add_argument(
        "--checkpoint_interval_batches", type=int, default=25,
        help="Frequency of intra-epoch batch checkpointing (default: 25 batches)",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory to store results and progress manifests (default: results)",
    )
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = results_dir / "ALL_COMPLETED"
    status_path   = results_dir / "master_progress.json"

    # If completely finished previously and resuming
    resume = not args.no_resume
    if resume and sentinel_path.exists():
        # Verify that all requested experiments are recorded
        tracker = MasterProgressTracker(status_path)
        all_req_done = all(tracker.is_completed(e) for e in args.exp)
        if all_req_done:
            print("\n" + "=" * 70)
            print("  🎉 ALL REQUESTED EXPERIMENTS ARE ALREADY COMPLETED!")
            print(f"  Completion marker: {sentinel_path}")
            print("  Exiting without redundant computation.")
            print("=" * 70 + "\n")
            sys.exit(0)

    # Initialize time budget manager for cluster queue safety (48h limit)
    time_budget_mgr = TimeBudgetManager(max_runtime_hours=args.max_hours)
    tracker = MasterProgressTracker(status_path)

    print("\n" + "=" * 70)
    print("  QC-CNN-Parallel Batch Execution Runner")
    print(f"  Target Experiments: {args.exp}")
    print(f"  Max Queue Budget  : {args.max_hours:.2f} hours (Queue limit: 48h)")
    print(f"  Batch Checkpoint  : Every {args.checkpoint_interval_batches} batches")
    print(f"  Resume enabled    : {resume}")
    print("=" * 70)

    try:
        # --- Experiment 1 ---
        if 1 in args.exp:
            if resume and tracker.is_completed(1):
                print("\n  ✓ EXPERIMENT 1 is already fully completed — skipping.")
            else:
                print("\n" + "=" * 60)
                print("  EXPERIMENT 1: PQC Selection Study (Tables 2 & 3)")
                print("=" * 60)
                from experiments.experiment1_circuit_selection import run_experiment1
                run_experiment1(
                    resume=resume,
                    time_budget_mgr=time_budget_mgr,
                    checkpoint_interval_batches=args.checkpoint_interval_batches,
                )
                tracker.mark_completed(1)

        # --- Experiment 2 ---
        if 2 in args.exp:
            if resume and tracker.is_completed(2):
                print("\n  ✓ EXPERIMENT 2 is already fully completed — skipping.")
            else:
                print("\n" + "=" * 60)
                print("  EXPERIMENT 2: Classification Benchmarks (Figs 6-8)")
                print("=" * 60)
                from experiments.experiment2_classification import run_experiment2
                run_experiment2(
                    datasets_to_run=args.dataset,
                    resume=resume,
                    time_budget_mgr=time_budget_mgr,
                    checkpoint_interval_batches=args.checkpoint_interval_batches,
                )
                tracker.mark_completed(2)

        # --- Experiment 3 ---
        if 3 in args.exp:
            if resume and tracker.is_completed(3):
                print("\n  ✓ EXPERIMENT 3 is already fully completed — skipping.")
            else:
                print("\n" + "=" * 60)
                print("  EXPERIMENT 3: Noise Robustness (Tables 5-8)")
                print("=" * 60)
                from experiments.experiment3_noise_robustness import run_experiment3
                run_experiment3(
                    resume=resume,
                    time_budget_mgr=time_budget_mgr,
                    checkpoint_interval_batches=args.checkpoint_interval_batches,
                )
                tracker.mark_completed(3)

        # --- Experiment 4 ---
        if 4 in args.exp:
            if resume and tracker.is_completed(4):
                print("\n  ✓ EXPERIMENT 4 is already fully completed — skipping.")
            else:
                print("\n" + "=" * 60)
                print("  EXPERIMENT 4: Ablation Study — Quantum vs Classical Branch")
                print("=" * 60)
                from experiments.experiment4_ablation_study import run_experiment4
                run_experiment4(
                    datasets_to_run=args.dataset_exp4,
                    resume=resume,
                    time_budget_mgr=time_budget_mgr,
                    checkpoint_interval_batches=args.checkpoint_interval_batches,
                )
                tracker.mark_completed(4)

        # --- Experiment 5 ---
        if 5 in args.exp:
            if resume and tracker.is_completed(5):
                print("\n  ✓ EXPERIMENT 5 is already fully completed — skipping.")
            else:
                print("\n" + "=" * 60)
                print("  EXPERIMENT 5: Scalability Study — Qubit Count & Circuit Depth")
                print("=" * 60)
                from experiments.experiment5_scalability_study import run_experiment5
                run_experiment5(
                    parts=args.part,
                    dataset_name=args.dataset_exp5,
                    resume=resume,
                    time_budget_mgr=time_budget_mgr,
                    checkpoint_interval_batches=args.checkpoint_interval_batches,
                )
                tracker.mark_completed(5)

        # All requested experiments successfully finished!
        with open(sentinel_path, "w") as f:
            f.write(f"Completed all experiments {args.exp} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print("\n" + "=" * 70)
        print("  🎉 ALL EXPERIMENTS FINISHED SUCCESSFULLY!")
        print(f"  Completion recorded at: {sentinel_path}")
        print(f"  Total wall-clock runtime: {time_budget_mgr.elapsed_hours:.2f} hours")
        print("=" * 70 + "\n")
        sys.exit(0)

    except TimeBudgetExceeded as e:
        print("\n" + "=" * 70)
        print("  ⏳ [WALLTIME BUDGET PAUSE]")
        print(f"  {e}")
        print(f"  Elapsed wall-clock time: {time_budget_mgr.elapsed_hours:.2f} hours.")
        print("  All model parameters, batch states, optimizer buffers, and RNG seeds")
        print("  have been safely written to disk via atomic checkpoints.")
        print("  Exiting with status code 42 for automatic cluster queue re-submission.")
        print("=" * 70 + "\n")
        sys.exit(42)


if __name__ == "__main__":
    main()
