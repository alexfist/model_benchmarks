"""
run_benchmark.py (MiniMol)
--------------------------
Runs MiniMol across all 22 TDC ADMET tasks.

Usage:
    python run_benchmark.py
    python run_benchmark.py --runs 5
    python run_benchmark.py --task hia_hou

Outputs:
    ../data/          TDC datasets (auto-downloaded)
    ../logs/          per-task JSON result logs + tuning logs
    ../artifacts/     saved model.pkl + hyperparams.json per task
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
from sklearn.metrics import roc_auc_score, mean_absolute_error


# ── GPU memory tracking ────────────────────────────────────────────────────────

def get_gpu_memory_mib():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 1024 / 1024
    except Exception:
        pass
    return 0.0


# ── MiniMol predictor with tuning ─────────────────────────────────────────────

def run_minimol(model, train, valid, test, task_name, logs_dir):
    """
    Stage 1: Generate MiniMol fingerprints.
    Stage 2: Grid search Random Forest hyperparams on validation set.
    Stage 3: Retrain best model on train+valid, predict on test.
    Returns: preds, best_clf, best_params
    """
    train_fps = model(train["Drug"].tolist()).numpy()
    valid_fps = model(valid["Drug"].tolist()).numpy()
    test_fps  = model(test["Drug"].tolist()).numpy()

    y_train = train["Y"].values
    y_valid = valid["Y"].values

    unique_vals = np.unique(y_train)
    is_clf = len(unique_vals) <= 2 or set(unique_vals).issubset({0, 1})

    param_grid = [
        {"n_estimators": 50,  "max_depth": None},
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 200, "max_depth": 10},
    ]

    tuning_logs  = []
    best_score   = None
    best_params  = None
    best_clf     = None
    higher_is_better = is_clf

    for params in param_grid:
        combo_label = f"n_estimators={params['n_estimators']}, max_depth={params['max_depth']}"
        try:
            if is_clf:
                candidate = RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    random_state=42, n_jobs=-1
                )
                candidate.fit(train_fps, y_train)
                val_preds = candidate.predict_proba(valid_fps)[:, 1]
                val_score = roc_auc_score(y_valid, val_preds)
            else:
                candidate = RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    random_state=42, n_jobs=-1
                )
                candidate.fit(train_fps, y_train)
                val_preds = candidate.predict(valid_fps)
                val_score = mean_absolute_error(y_valid, val_preds)

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

        except Exception as e:
            tuning_logs.append({"combo": combo_label, "error": str(e)})

    # Save tuning log
    tuning_log_path = os.path.join(logs_dir, f"{task_name}_tuning.json")
    with open(tuning_log_path, "w") as f:
        json.dump({
            "task":           task_name,
            "model":          "MiniMol",
            "tuning_runs":    tuning_logs,
            "best_params":    {k: (v if v else "None") for k, v in best_params.items()},
            "best_val_score": round(float(best_score), 4),
        }, f, indent=2)

    print(f"   Tuning done. Best: {best_params} | val: {best_score:.4f}")

    # Retrain on train + valid combined
    X = np.vstack([train_fps, valid_fps])
    y = np.concatenate([y_train, y_valid])

    if is_clf:
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

    return preds, clf, best_params


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

    print("Loading MiniMol model...")
    minimol_model = Minimol()
    print("MiniMol loaded.\n")

    all_results = {}

    print(f"{'='*60}")
    print(f"  Model:   MiniMol")
    print(f"  Tasks:   {len(benchmark_names)}")
    print(f"  Runs:    {n_runs} (seeds 1-{n_runs})")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for task_name in benchmark_names:
        print(f"── Task: {task_name}")

        try:
            benchmark    = group.get(task_name)
            test         = benchmark["test"]
            predictions_list = []
            timing_list      = []
            best_params_final = None

            for run_idx in range(n_runs):
                seed = run_idx + 1  # TDC standard: seeds 1,2,3,4,5

                train, valid = group.get_train_valid_split(
                    benchmark=task_name,
                    split_type="default",
                    seed=seed
                )

                gpu_before = get_gpu_memory_mib()
                t_start    = time.time()

                preds, clf, best_params = run_minimol(
                    minimol_model, train, valid, test, task_name, logs_dir
                )

                t_end      = time.time()
                gpu_after  = get_gpu_memory_mib()

                predictions_list.append({task_name: preds})
                timing_list.append({
                    "run":            seed,
                    "train_time_sec": round(t_end - t_start, 1),
                    "gpu_memory_mib": round(gpu_after - gpu_before, 1),
                })

                print(f"   Seed {seed}/{n_runs} | "
                      f"time: {round(t_end - t_start, 1)}s | "
                      f"GPU: {round(gpu_after - gpu_before, 1)} MiB")

                # Save artifact from first run
                if run_idx == 0:
                    best_params_final = best_params
                    task_artifact_dir = os.path.join(artifacts_dir, task_name)
                    os.makedirs(task_artifact_dir, exist_ok=True)
                    joblib.dump(clf, os.path.join(task_artifact_dir, "model.pkl"))
                    with open(os.path.join(task_artifact_dir, "hyperparams.json"), "w") as f:
                        json.dump({
                            "task":      task_name,
                            "model":     "MiniMol",
                            "params":    {k: (v if v else "None") for k, v in best_params.items()},
                            "seed":      seed,
                            "timestamp": datetime.now().isoformat(),
                        }, f, indent=2)

            # Evaluate all runs together using TDC's official evaluate_many()
            results     = group.evaluate_many(predictions_list)
            mean_score  = results[task_name][0]
            std_score   = results[task_name][1]

            # Get metric name
            single_result = group.evaluate({task_name: predictions_list[0][task_name]})
            metric_name   = list(single_result[task_name].keys())[0]

            times = [t["train_time_sec"] for t in timing_list]
            gpus  = [t["gpu_memory_mib"] for t in timing_list]

            summary = {
                "task":                task_name,
                "model":               "MiniMol",
                "metric":              metric_name,
                "score_mean":          round(float(mean_score), 4),
                "score_std":           round(float(std_score), 4),
                "train_time_sec_mean": round(float(np.mean(times)), 1),
                "gpu_memory_mib_mean": round(float(np.mean(gpus)), 1),
                "runs":                timing_list,
                "timestamp":           datetime.now().isoformat(),
            }
            all_results[task_name] = summary

            log_path = os.path.join(logs_dir, f"{task_name}.json")
            with open(log_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"   ✓ Mean: {mean_score:.4f} ± {std_score:.4f}\n")

        except Exception as e:
            print(f"   ✗ FAILED: {e}\n")
            error_result = {
                "task":      task_name,
                "model":     "MiniMol",
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
    parser = argparse.ArgumentParser(description="MiniMol TDC ADMET Benchmarking Script")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per task (default: 5)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task only (e.g. --task hia_hou)")
    args = parser.parse_args()
    run_benchmark(n_runs=args.runs, task_filter=args.task)