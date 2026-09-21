"""
run_all.py
==========
Master entry point to run all five paper experiments sequentially.

Paper: "A Parallel Hybrid Quantum-Classical Convolutional Design Using
        Parameterized Quantum Circuits for Image Classification"
        Quantum Engineering (2026), article 6643049.

Usage:
  # Run Experiment 2 only (classification — fastest to validate correctness)
  python run_all.py --exp 2 --dataset mnist

  # Run Experiment 2 on all datasets
  python run_all.py --exp 2

  # Run all experiments in order
  python run_all.py --exp 1 2 3 4 5

  # Run Experiment 4 (ablation) on a single dataset
  python run_all.py --exp 4 --dataset_exp4 mnist

  # Run Experiment 5 (scalability) — Part A only
  python run_all.py --exp 5 --part A --dataset_exp5 mnist

  # Smoke test with tiny subset (verify forward/backward pass works)
  python run_all.py --smoke_test

Experiment map
--------------
  Experiment 1 → experiments/experiment1_circuit_selection.py
                 Reproduces Tables 2 and 3 (PQC selection study)

  Experiment 2 → experiments/experiment2_classification.py
                 Reproduces Figures 6, 7, 8 and Table 4 (main classification)

  Experiment 3 → experiments/experiment3_noise_robustness.py
                 Reproduces Tables 5, 6, 7, 8 (noise robustness)

  Experiment 4 → experiments/experiment4_ablation_study.py
                 Ablation study: isolates quantum vs classical branch contribution
                 across all 4 model variants and all 3 datasets

  Experiment 5 → experiments/experiment5_scalability_study.py
                 Scalability study: sweeps qubit count (Part A) and circuit
                 depth (Part B) to validate the paper's 4-qubit / depth-3 choice
"""

from __future__ import annotations

import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))


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
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    resume = not args.no_resume

    if 1 in args.exp:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 1: PQC Selection Study (Tables 2 & 3)")
        print("=" * 60)
        from experiments.experiment1_circuit_selection import run_experiment1
        run_experiment1(resume=resume)

    if 2 in args.exp:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 2: Classification Benchmarks (Figs 6-8)")
        print("=" * 60)
        from experiments.experiment2_classification import run_experiment2
        run_experiment2(datasets_to_run=args.dataset, resume=resume)

    if 3 in args.exp:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 3: Noise Robustness (Tables 5-8)")
        print("=" * 60)
        from experiments.experiment3_noise_robustness import run_experiment3
        run_experiment3(resume=resume)

    if 4 in args.exp:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 4: Ablation Study — Quantum vs Classical Branch")
        print("=" * 60)
        from experiments.experiment4_ablation_study import run_experiment4
        run_experiment4(datasets_to_run=args.dataset_exp4, resume=resume)

    if 5 in args.exp:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 5: Scalability Study — Qubit Count & Circuit Depth")
        print("=" * 60)
        from experiments.experiment5_scalability_study import run_experiment5
        run_experiment5(parts=args.part, dataset_name=args.dataset_exp5,
                        resume=resume)


if __name__ == "__main__":
    main()
