"""
run_benchmark.py (BaseBoosting)
--------------------------------
Reproduces BaseBoosting (Oloren AI, 2022) across all 22 TDC ADMET tasks.

Pipeline:
    SMILES → olorenchemengine BaseBoosting ensemble of 3 Random Forest learners:
        1. RF on Morgan counts fingerprints (descriptastorus "morgan3counts")
        2. RF on normalized RDKit 2D descriptors ("rdkit2dnormalized")
        3. RF on OlorenCheckpoint("default") — proprietary pre-trained GIN
           fingerprint, downloads from Google Cloud Storage at runtime

NETWORK WARNING:
    Learner 3 (OlorenCheckpoint) downloads weights from:
        storage.googleapis.com/oloren-public-data
    This may be blocked on restricted servers. The script will attempt all 3
    learners and gracefully fall back to 2 learners if the download fails.
    The deviation is logged and documented in the result JSON.

INSTALL WARNING:
    olorenchemengine uses a shell script installer that downloads from GitHub.
    If GitHub is blocked, install via pip directly:
        pip install olorenchemengine
    Or copy the wheel to the server manually — see README.

Usage:
    python run_benchmark.py
    python run_benchmark.py --runs 5
    python run_benchmark.py --task hia_hou

Outputs:
    ../data/          TDC datasets (auto-downloaded)
    ../logs/          per-task JSON result logs
    ../artifacts/     saved model + hyperparams.json per task
"""

import os
import json
import time
import warnings
import argparse
import traceback
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

from tdc.benchmark_group import admet_group
from sklearn.metrics import roc_auc_score, mean_absolute_error

import olorenchemengine as oce


def is_classification_task(y):
    unique_vals = np.unique(y)
    return len(unique_vals) <= 2 or set(unique_vals).issubset({0, 1})


def build_model(use_oloren_checkpoint=True):
    """
    Build the BaseBoosting model.
    If use_oloren_checkpoint=False, use only 2 learners (Morgan + RDKit2D).
    """
    learners = [
        oce.RandomForestModel(
            oce.DescriptastorusDescriptor("morgan3counts"),
            n_estimators=1000
        ),
        oce.RandomForestModel(
            oce.DescriptastorusDescriptor("rdkit2dnormalized"),
            n_estimators=1000
        ),
    ]

    if use_oloren_checkpoint:
        learners.append(
            oce.RandomForestModel(
                oce.OlorenCheckpoint("default"),
                n_estimators=1000
            )
        )

    return oce.BaseBoosting(learners)


def test_oloren_checkpoint():
    """
    Try to instantiate OlorenCheckpoint to see if download succeeds.
    Returns True if available, False if blocked/failed.
    """
    try:
        print("   Checking OlorenCheckpoint availability...")
        checkpoint = oce.OlorenCheckpoint("default")
        # Try converting a single test molecule
        test_result = checkpoint.convert(["CC(=O)O"])
        print("   OlorenCheckpoint: available ✓")
        return True
    except Exception as e:
        print(f"   OlorenCheckpoint: unavailable ({e})")
        print("   Falling back to 2-learner model (Morgan + RDKit2D only)")
        return False


def run_baseboosting(train, valid, test, task_name, logs_dir,
                     use_oloren_checkpoint=True):
    """
    Train BaseBoosting on train set, evaluate on valid, retrain on train+valid,
    predict on test.

    OCE's BaseBoosting handles classification vs regression automatically.
    Returns: preds, model, n_learners_used
    """
    is_clf = is_classification_task(train["Y"].values)

    # Build model
    model = build_model(use_oloren_checkpoint=use_oloren_checkpoint)
    n_learners = 3 if use_oloren_checkpoint else 2

    # Combine train + valid for final fit (following TDC protocol)
    train_valid_df = pd.concat([train, valid]).reset_index(drop=True)

    # Fit on train only first to get validation score
    model_val = build_model(use_oloren_checkpoint=use_oloren_checkpoint)
    model_val.fit(
        train["Drug"].tolist(),
        train["Y"].tolist()
    )
    val_preds_raw = model_val.predict(valid["Drug"].tolist())
    val_preds = np.array(val_preds_raw, dtype=np.float64).flatten()

    if is_clf:
        val_score = roc_auc_score(valid["Y"].values, val_preds)
        print(f"   Val score (AUROC): {val_score:.4f}")
    else:
        val_score = mean_absolute_error(valid["Y"].values, val_preds)
        print(f"   Val score (MAE): {val_score:.4f}")

    # Save val score to tuning log (no grid search — fixed architecture)
    with open(os.path.join(logs_dir, f"{task_name}_tuning.json"), "w") as f:
        json.dump({
            "task":                task_name,
            "model":               "BaseBoosting",
            "n_learners":          n_learners,
            "use_oloren_checkpoint": use_oloren_checkpoint,
            "val_score":           round(float(val_score), 4),
            "note": ("No hyperparameter search — BaseBoosting uses fixed "
                     "architecture with n_estimators=1000 per learner"),
        }, f, indent=2)

    # Retrain on train + valid combined
    final_model = build_model(use_oloren_checkpoint=use_oloren_checkpoint)
    final_model.fit(
        train_valid_df["Drug"].tolist(),
        train_valid_df["Y"].tolist()
    )

    preds_raw = final_model.predict(test["Drug"].tolist())
    preds = np.array(preds_raw, dtype=np.float64).flatten().tolist()

    return preds, final_model, n_learners


# ── Main benchmark loop ────────────────────────────────────────────────────────

def run_benchmark(n_runs=5, task_filter=None):
    data_dir      = "../data"
    logs_dir      = "../logs"
    artifacts_dir = "../artifacts"

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    group = admet_group(path=data_dir)
    benchmark_names = group.dataset_names

    if task_filter:
        if task_filter not in benchmark_names:
            raise ValueError(f"Unknown task: {task_filter}. Available: {benchmark_names}")
        benchmark_names = [task_filter]

    # Check OlorenCheckpoint availability once at the start
    use_oloren_checkpoint = test_oloren_checkpoint()
    n_learners_label = "3 (Morgan + RDKit2D + OlorenCheckpoint)" if use_oloren_checkpoint \
                       else "2 (Morgan + RDKit2D) — OlorenCheckpoint unavailable"

    all_results = {}

    print(f"\n{'='*60}")
    print(f"  Model:    BaseBoosting (Oloren AI)")
    print(f"  Learners: {n_learners_label}")
    print(f"  Tasks:    {len(benchmark_names)}")
    print(f"  Runs:     {n_runs} (seeds 1-{n_runs})")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for task_name in benchmark_names:
        print(f"── Task: {task_name}")

        try:
            benchmark        = group.get(task_name)
            test             = benchmark["test"]
            predictions_list = []
            timing_list      = []

            for run_idx in range(n_runs):
                seed = run_idx + 1

                train, valid = group.get_train_valid_split(
                    benchmark=task_name,
                    split_type="default",
                    seed=seed
                )

                t_start = time.time()

                preds, final_model, n_learners = run_baseboosting(
                    train, valid, test, task_name, logs_dir,
                    use_oloren_checkpoint=use_oloren_checkpoint
                )

                t_end = time.time()

                predictions_list.append({task_name: preds})
                timing_list.append({
                    "run":            seed,
                    "train_time_sec": round(t_end - t_start, 1),
                    "gpu_memory_mib": 0.0,
                    "n_learners":     n_learners,
                })

                print(f"   Seed {seed}/{n_runs} | time: {round(t_end - t_start, 1)}s")

                if run_idx == 0:
                    task_artifact_dir = os.path.join(artifacts_dir, task_name)
                    os.makedirs(task_artifact_dir, exist_ok=True)
                    oce.save(final_model,
                             os.path.join(task_artifact_dir, "model.oce"))
                    with open(os.path.join(task_artifact_dir, "hyperparams.json"), "w") as f:
                        json.dump({
                            "task":                  task_name,
                            "model":                 "BaseBoosting",
                            "n_learners":            n_learners,
                            "use_oloren_checkpoint": use_oloren_checkpoint,
                            "n_estimators_per_rf":   1000,
                            "seed":                  seed,
                            "timestamp":             datetime.now().isoformat(),
                        }, f, indent=2)

            # evaluate_many needs 5 runs; fall back for < 5
            single_result = group.evaluate({task_name: predictions_list[0][task_name]})
            metric_name   = list(single_result[task_name].keys())[0]

            if len(predictions_list) == 5:
                results    = group.evaluate_many(predictions_list)
                mean_score = results[task_name][0]
                std_score  = results[task_name][1]
            else:
                mean_score = list(single_result[task_name].values())[0]
                std_score  = 0.0

            times = [t["train_time_sec"] for t in timing_list]

            summary = {
                "task":                  task_name,
                "model":                 "BaseBoosting",
                "n_learners":            n_learners,
                "use_oloren_checkpoint": use_oloren_checkpoint,
                "metric":                metric_name,
                "score_mean":            round(float(mean_score), 4),
                "score_std":             round(float(std_score), 4),
                "train_time_sec_mean":   round(float(np.mean(times)), 1),
                "gpu_memory_mib_mean":   0.0,
                "runs":                  timing_list,
                "timestamp":             datetime.now().isoformat(),
            }
            all_results[task_name] = summary

            with open(os.path.join(logs_dir, f"{task_name}.json"), "w") as f:
                json.dump(summary, f, indent=2)

            print(f"   ✓ Mean: {mean_score:.4f} ± {std_score:.4f}  [{metric_name}]\n")

        except Exception as e:
            print(f"   ✗ FAILED: {e}\n")
            error_result = {
                "task":      task_name,
                "model":     "BaseBoosting",
                "error":     str(e),
                "traceback": traceback.format_exc(),
            }
            all_results[task_name] = error_result
            with open(os.path.join(logs_dir, f"{task_name}_error.json"), "w") as f:
                json.dump(error_result, f, indent=2)

    combined_path = os.path.join(logs_dir, "_all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✓ Done. Results saved to: {combined_path}")
    return all_results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BaseBoosting (Oloren AI) TDC ADMET Benchmarking Script"
    )
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per task (default: 5)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task only (e.g. --task hia_hou)")
    args = parser.parse_args()
    run_benchmark(n_runs=args.runs, task_filter=args.task)