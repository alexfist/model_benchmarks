"""
run_benchmark.py
----------------
Runs MiniMol across all 22 TDC ADMET tasks and logs results.

Usage:
    python run_benchmark.py
    python run_benchmark.py --runs 5
    python run_benchmark.py --task hia_hou   # run a single task

Outputs:
    ../data/          TDC datasets (auto-downloaded)
    ../logs/          per-task JSON result logs
    ../artifacts/     saved downstream model files (model.pkl per task)
"""

import os
import json
import time
import argparse
import traceback
import numpy as np
import joblib
from datetime import datetime

from tdc.benchmark_group import admet_group
from minimol import Minimol
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ── GPU memory tracking ────────────────────────────────────────────────────────

def get_gpu_memory_mib():
    """Returns current GPU memory usage in MiB. Returns 0 if no GPU available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 1024 / 1024
    except Exception:
        pass
    return 0.0


# ── MiniMol predictor ──────────────────────────────────────────────────────────

def run_minimol(model, train, valid, test):
    """
    Stage 1: Generate MiniMol fingerprints for all splits.
    Stage 2: Train a Random Forest on train+valid, predict on test.
    Returns predictions and the trained classifier (for saving to artifacts).
    """
    # Stage 1: Generate fingerprints for all splits
    train_fps = model(train["Drug"].tolist()).numpy()
    valid_fps = model(valid["Drug"].tolist()).numpy()
    test_fps  = model(test["Drug"].tolist()).numpy()

    # Stage 2: Combine train and valid for final training
    X = np.vstack([train_fps, valid_fps])
    y = np.concatenate([train["Y"].values, valid["Y"].values])

    # Detect classification vs regression by label values
    unique_vals = np.unique(y)
    is_classification = len(unique_vals) <= 2 or set(unique_vals).issubset({0, 1})

    # ── Hyperparameter grid search on validation set ───────────────────────────
    from sklearn.metrics import roc_auc_score, mean_absolute_error

    param_grid = [
        {"n_estimators": 50,  "max_depth": None},
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 200, "max_depth": 10},
    ]

    tuning_logs = []
    best_score  = None
    best_params = None
    best_clf    = None

    for params in param_grid:
        if is_classification:
            candidate = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42, n_jobs=-1
            )
            candidate.fit(train_fps, train["Y"].values)
            val_preds = candidate.predict_proba(valid_fps)[:, 1]
            val_score = roc_auc_score(valid["Y"].values, val_preds)
            higher_is_better = True
        else:
            candidate = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42, n_jobs=-1
            )
            candidate.fit(train_fps, train["Y"].values)
            val_preds = candidate.predict(valid_fps)
            val_score = mean_absolute_error(valid["Y"].values, val_preds)
            higher_is_better = False

        tuning_logs.append({
            "n_estimators": params["n_estimators"],
            "max_depth":    params["max_depth"] if params["max_depth"] else "None",
            "val_score":    round(float(val_score), 4),
        })

        is_better = (
            best_score is None or
            (higher_is_better and val_score > best_score) or
            (not higher_is_better and val_score < best_score)
        )
        if is_better:
            best_score  = val_score
            best_params = params
            best_clf    = candidate

    # Save tuning logs to ../logs/<task_name>_tuning.json
    tuning_log_path = os.path.join(logs_dir, f"{task_name}_tuning.json")
    with open(tuning_log_path, "w") as f:
        json.dump({
            "task":        task_name,
            "model":       "MiniMol",
            "tuning_runs": tuning_logs,
            "best_params": {
                "n_estimators": best_params["n_estimators"],
                "max_depth":    best_params["max_depth"] if best_params["max_depth"] else "None",
            },
            "best_val_score": round(float(best_score), 4),
        }, f, indent=2)

    print(f"   Tuning complete. Best params: {best_params} | val score: {best_score:.4f}")

    # Retrain best model on train+valid combined for final test evaluation
    X = np.vstack([train_fps, valid_fps])
    y = np.concatenate([train["Y"].values, valid["Y"].values])

    if is_classification:
        clf = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=42, n_jobs=-1
        )
        clf.fit(X, y)
        preds = clf.predict_proba(test_fps)[:, 1]
    else:
        clf = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=42, n_jobs=-1
        )
        clf.fit(X, y)
        preds = clf.predict(test_fps)

    return preds, clf


# ── Main benchmark loop ────────────────────────────────────────────────────────

def run_benchmark(n_runs=3, task_filter=None):
    """
    Runs MiniMol across all 22 ADMET tasks, n_runs times each.
    Saves per-task JSON logs and artifact model files.
    """
    # Paths relative to this script (which lives in code/)
    data_dir      = "../data"
    logs_dir      = "../logs"
    artifacts_dir = "../artifacts"

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Load TDC benchmark group
    group = admet_group(path=data_dir)
    benchmark_names = group.dataset_names

    # Filter to a single task if requested
    if task_filter:
        if task_filter not in benchmark_names:
            raise ValueError(f"Unknown task: {task_filter}. Available: {benchmark_names}")
        benchmark_names = [task_filter]

    # Load MiniMol once — reused across all tasks
    print("Loading MiniMol model...")
    minimol_model = Minimol()
    print("MiniMol loaded.\n")

    all_results = {}

    print(f"{'='*60}")
    print(f"  Model:   MiniMol")
    print(f"  Tasks:   {len(benchmark_names)}")
    print(f"  Runs:    {n_runs} per task")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for task_name in benchmark_names:
        print(f"── Task: {task_name}")
        task_results = []

        try:
            benchmark = group.get(task_name)
            test = benchmark["test"]

            for run_idx in range(n_runs):
                # Fresh train/valid split each run using a different seed
                train, valid = group.get_train_valid_split(
                    benchmark=task_name,
                    split_type="default",
                    seed=run_idx * 42
                )

                gpu_before = get_gpu_memory_mib()
                t_start = time.time()

                preds, clf = run_minimol(minimol_model, train, valid, test)

                t_end = time.time()
                gpu_after = get_gpu_memory_mib()

                # Evaluate using TDC's built-in evaluator
                predictions = {task_name: preds}
                result = group.evaluate(predictions, benchmark=task_name)
                metric_name = list(result[task_name].keys())[0]
                score = result[task_name][metric_name]

                run_result = {
                    "run":             run_idx,
                    "metric":          metric_name,
                    "score":           round(float(score), 4),
                    "train_time_sec":  round(t_end - t_start, 1),
                    "gpu_memory_mib":  round(gpu_after - gpu_before, 1),
                }
                task_results.append(run_result)
                print(f"   Run {run_idx+1}/{n_runs} | {metric_name}: {score:.4f} | "
                      f"time: {run_result['train_time_sec']}s | "
                      f"GPU: {run_result['gpu_memory_mib']} MiB")

                # Save model artifact for the first run only
                if run_idx == 0:
                    task_artifact_dir = os.path.join(artifacts_dir, task_name)
                    os.makedirs(task_artifact_dir, exist_ok=True)
                    joblib.dump(clf, os.path.join(task_artifact_dir, "model.pkl"))

            # Aggregate across runs
            scores = [r["score"] for r in task_results]
            times  = [r["train_time_sec"] for r in task_results]
            gpus   = [r["gpu_memory_mib"] for r in task_results]

            summary = {
                "task":                  task_name,
                "model":                 "MiniMol",
                "metric":                task_results[0]["metric"],
                "score_mean":            round(float(np.mean(scores)), 4),
                "score_std":             round(float(np.std(scores)), 4),
                "train_time_sec_mean":   round(float(np.mean(times)), 1),
                "gpu_memory_mib_mean":   round(float(np.mean(gpus)), 1),
                "runs":                  task_results,
                "timestamp":             datetime.now().isoformat(),
            }
            all_results[task_name] = summary

            # Save per-task log immediately so progress isn't lost on crash
            log_path = os.path.join(logs_dir, f"{task_name}.json")
            with open(log_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"   ✓ Mean: {summary['score_mean']} ± {summary['score_std']}\n")

        except Exception as e:
            print(f"   ✗ FAILED: {e}\n")
            error_result = {
                "task":      task_name,
                "model":     "MiniMol",
                "error":     str(e),
                "traceback": traceback.format_exc(),
            }
            all_results[task_name] = error_result

            # Save error log so we know what failed
            log_path = os.path.join(logs_dir, f"{task_name}_error.json")
            with open(log_path, "w") as f:
                json.dump(error_result, f, indent=2)

    # Save combined results file
    combined_path = os.path.join(logs_dir, "_all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✓ Done. All results saved to: {combined_path}")
    return all_results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMol TDC ADMET Benchmarking Script")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per task (default: 3)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task only (e.g. --task hia_hou)")
    args = parser.parse_args()

    run_benchmark(n_runs=args.runs, task_filter=args.task)