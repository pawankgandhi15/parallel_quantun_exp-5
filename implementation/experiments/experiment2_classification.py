"""
experiment2_classification.py
==============================
Experiment 2: Analysis of General Classification Performance
(Paper Section 4.3.2, Figures 6, 7, 8, Table 4)

Trains QC-CNN-Parallel (Circuit 11) against 2 baselines on all 3 paper
datasets and reproduces the accuracy/loss training curves.

Models compared:
  - ClassicalCNN   (LeNet-5 adapted, 464 conv params, Table 4)
  - QC-CNN-Parallel  (proposed, 136 conv params, Table 4)

Datasets (Table 1):
  - MNIST         : 10,000 train / 2,000 test (balanced subsampling)
  - Fashion-MNIST : 10,000 train / 2,000 test (balanced subsampling)
  - Overhead-MNIST:  8,519 train / 1,065 test (full dataset)

Hyper-params (Table 4):
  lr=0.01, seed=42, batch_size=32, epochs=50, optimizer=Adam

Usage:
  python experiments/experiment2_classification.py [--dataset mnist]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import torch
from pathlib import Path

from models       import QCCNNParallel, ClassicalCNN
from datasets     import get_dataloaders
from training     import train, set_seed
from utils        import plot_training_curves, plot_confusion_matrix


# ---------------------------------------------------------------------------
# Configuration (Table 4)
# ---------------------------------------------------------------------------
SEED       = 42
LR         = 0.01
BATCH_SIZE = 32
EPOCHS     = 70
RESULTS    = Path("results/experiment2")
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Paper reference values  (noise-free baseline from Tables 5–8)
# ---------------------------------------------------------------------------
PAPER_REF = {
    "proposed" : {"MNIST": 0.9005, "Fashion-MNIST": "Fig. 7", "Overhead-MNIST": "Fig. 8"},
    "classical": {"MNIST": 0.8935, "Fashion-MNIST": "Fig. 7", "Overhead-MNIST": "Fig. 8"},
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_experiment2(datasets_to_run=("mnist", "fashion_mnist", "overhead_mnist"),
                    resume: bool = True):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Resume: load previous progress if available
    progress_path = RESULTS / "checkpoint_progress.json"
    all_results = {}
    if resume and progress_path.exists():
        try:
            with open(progress_path, "r") as f:
                all_results = json.load(f)
            completed = sum(len(models) for models in all_results.values())
            if completed > 0:
                print(f"  ⟳ Loaded checkpoint: {completed} completed training run(s).")
        except Exception:
            all_results = {}

    for ds_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name.upper()}")
        print(f"{'='*60}")

        # Load dataset
        try:
            train_loader, test_loader = get_dataloaders(
                ds_name, data_root="data", batch_size=BATCH_SIZE, seed=SEED
            )
        except FileNotFoundError as e:
            print(f"  [SKIP] Could not load {ds_name}: {e}")
            print("  Please download the dataset and set data_root accordingly.")
            continue

        # Determine number of classes from first batch
        _, sample_labels = next(iter(train_loader))
        num_classes = int(sample_labels.max().item()) + 1
        print(f"  Detected {num_classes} classes.")

        if ds_name not in all_results:
            all_results[ds_name] = {}

        ds_histories = {}

        for model_name, model_cls in [
            ("classical_cnn",     ClassicalCNN),
            ("qc_cnn_parallel",   QCCNNParallel),
        ]:
            # Skip if already completed on resume
            if resume and model_name in all_results.get(ds_name, {}):
                print(f"\n  ✓ {model_name} on {ds_name} — already completed, skipping.")
                continue

            print(f"\n  Training: {model_name}")
            model = model_cls(num_classes=num_classes)

            # Print parameter counts (Table 4)
            if hasattr(model, "count_parameters"):
                counts = model.count_parameters()
                print(f"  Conv params: {counts['conv_total']}  |  Total: {counts['total']}")

            history, summary, trained_model = train(
                model        = model,
                train_loader = train_loader,
                test_loader  = test_loader,
                num_epochs   = EPOCHS,
                lr           = LR,
                seed         = SEED,
                device       = device,
                save_dir     = str(RESULTS),
                model_name   = model_name,
                dataset_name = ds_name,
                resume       = resume,
            )

            all_results[ds_name][model_name] = summary
            ds_histories[model_name] = history

            # Save progress incrementally (checkpoint)
            with open(progress_path, "w") as f:
                json.dump(all_results, f, indent=2)

        # --- Plot training curves (reproducing Figures 6–8) ---
        # Only plot if we have history data from this run
        if ds_histories:
            plot_training_curves(
                ds_histories,
                metric    = "acc",
                title     = f"Accuracy — {ds_name} (Figure 6/7/8a)",
                save_path = str(RESULTS / ds_name / "accuracy_curve.png"),
            )
            plot_training_curves(
                ds_histories,
                metric    = "loss",
                title     = f"Loss — {ds_name} (Figure 6/7/8b)",
                save_path = str(RESULTS / ds_name / "loss_curve.png"),
            )

    # Save all summaries
    with open(RESULTS / "all_summaries.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Clean up checkpoint — experiment completed
    if progress_path.exists():
        progress_path.unlink()
        print(f"\n  ✓ Experiment 2 checkpoint cleaned up (all runs complete).")

    # Print final comparison
    print("\n" + "=" * 60)
    print("  Experiment 2 — Final Accuracy Comparison")
    print("=" * 60)
    for ds, models in all_results.items():
        print(f"\n  Dataset: {ds}")
        for mname, summary in models.items():
            print(f"    {mname:<25} acc={summary['best_test_acc']:.4f}  "
                  f"F1={summary['final_test_f1']:.4f}")
        print(f"    Paper reference (Proposed / MNIST no-noise): 0.9005")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", nargs="+",
        default=["mnist", "fashion_mnist", "overhead_mnist"],
        choices=["mnist", "fashion_mnist", "overhead_mnist"],
        help="Datasets to run (default: all three paper datasets)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore checkpoints and start from scratch",
    )
    args = parser.parse_args()
    run_experiment2(datasets_to_run=args.dataset, resume=not args.no_resume)
