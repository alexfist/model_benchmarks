"""
run_benchmark.py
----------------
Runs MapLight + GNN across all 22 TDC ADMET tasks.

Pipeline:
    ECFP + Avalon + ErG + RDKit 2D descriptors + GIN embeddings → CatBoost

Usage:
    python run_benchmark.py
    python run_benchmark.py --runs 3
    python run_benchmark.py --task hia_hou

Outputs:
    ../data/          TDC datasets (auto-downloaded)
    ../logs/          per-task JSON result logs + tuning logs
    ../artifacts/     saved CatBoost model per task
"""

import os
import json
import time
import argparse
import traceback
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from tdc.benchmark_group import admet_group
from molfeat.trans import MoleculeTransformer
from molfeat.trans.pretrained import PretrainedDGLTransformer
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error


# ── GPU memory tracking ────────────────────────────────────────────────────────

def get_gpu_memory_mib():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


# ── Task type detection ────────────────────────────────────────────────────────

def is_classification_task(y):
    unique_vals = np.unique(y)
    return len(unique_vals) <= 2 or set(unique_vals).issubset({0, 1})


# ── Featurization ──────────────────────────────────────────────────────────────

def build_transformers():
    """Build all feature transformers. Called once and reused across tasks."""
    print("Loading feature transformers...")
    transformers = {
        "ecfp":   MoleculeTransformer("ecfp:4"),
        "avalon": MoleculeTransformer("avalon"),
        "erg":    MoleculeTransformer("erg"),
        "desc2d": MoleculeTransformer("desc2D"),
        "gin":    PretrainedDGLTransformer(kind="gin_supervised_masking", dtype=float),
    }
    print("Transformers loaded.\n")
    return transformers


def featurize(smiles_list, transformers):
    """Concatenate all fingerprints and embeddings into one feature matrix."""
    features = []
    for name, trans in transformers.items():
        try:
            feat = np.array(trans(smiles_list))
            # Replace NaN/Inf with 0
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(feat)
        except Exception as e:
            print(f"  Warning: {name} featurizer failed: {e} — skipping")
    return np.hstack(features)


# ── MapLight + GNN runner with tuning ─────────────────────────────────────────

def run_maplight_gnn(transformers, train, valid, test, task_name, logs_dir):
    """
    Generates concatenated features, runs CatBoost hyperparameter tuning
    on validation set, retrains best model on train+valid, predicts on test.
    """
    is_clf = is_classification_task(train["Y"].values)

    train_smiles = train["Drug"].tolist()
    valid_smiles = valid["Drug"].tolist()
    test_smiles  = test["Drug"].tolist()

    X_train = featurize(train_smiles, transformers)
    X_valid = featurize(valid_smiles, transformers)
    X_test  = featurize(test_smiles,  transformers)

    y_train = train["Y"].values
    y_valid = valid["Y"].values

    # Hyperparameter grid
    param_grid = [
        {"iterations": 500,  "learning_rate": 0.1},
        {"iterations": 1000, "learning_rate": 0.1},
        {"iterations": 1000, "learning_rate": 0.03},
        {"iterations": 2000, "learning_rate": 0.03},
    ]

    tuning_logs  = []
    best_score   = None
    best_params  = None
    higher_is_better = is_clf

    for params in param_grid:
        combo_label = f"iter={params['iterations']}, lr={params['learning_rate']}"
        try:
            if is_clf:
                model = CatBoostClassifier(
                    iterations=params["iterations"],
                    learning_rate=params["learning_rate"],
                    random_seed=42, verbose=0
                )
                model.fit(X_train, y_train)
                val_preds = model.predict_proba(X_valid)[:, 1]
                val_score = roc_auc_score(y_valid, val_preds)
            else:
                model = CatBoostRegressor(
                    iterations=params["iterations"],
                    learning_rate=params["learning_rate"],
                    random_seed=42, verbose=0
                )
                model.fit(X_train, y_train)
                val_preds = model.predict(X_valid)
                val_score = mean_absolute_error(y_valid, val_preds)

            tuning_logs.append({
                "iterations":    params["iterations"],
                "learning_rate": params["learning_rate"],
                "combo":         combo_label,
                "val_score":     round(float(val_score), 4),
            })
            print(f"     {combo_label}: {val_score:.4f}")

            is_better = (
                best_score is None or
                (higher_is_better and val_score > best_score) or
                (not higher_is_better and val_score < best_score)
            )
            if is_better:
                best_score  = val_score
                best_params = params

        except Exception as e:
            print(f"     {combo_label} failed: {e}")
            tuning_logs.append({
                "iterations":    params["iterations"],
                "learning_rate": params["learning_rate"],
                "combo":         combo_label,
                "error":         str(e),
            })

    # Save tuning log
    tuning_log_path = os.path.join(logs_dir, f"{task_name}_tuning.json")
    with open(tuning_log_path, "w") as f:
        json.dump({
            "task":           task_name,
            "model":          "MapLight_GNN",
            "tuning_runs":    tuning_logs,
            "best_params":    best_params,
            "best_val_score": round(float(best_score), 4) if best_score else None,
        }, f, indent=2)

    print(f"   Best params: {best_params} | val score: {best_score:.4f}")

    # Retrain on train + valid combined
    X_all = np.vstack([X_train, X_valid])
    y_all = np.concatenate([y_train, y_valid])

    if is_clf:
        final_model = CatBoostClassifier(
            iterations=best_params["iterations"],
            learning_rate=best_params["learning_rate"],
            random_seed=42, verbose=0
        )
        final_model.fit(X_all, y_all)
        preds = final_model.predict_proba(X_test)[:, 1]
    else:
        final_model = CatBoostRegressor(
            iterations=best_params["iterations"],
            learning_rate=best_params["learning_rate"],
            random_seed=42, verbose=0
        )
        final_model.fit(X_all, y_all)
        preds = final_model.predict(X_test)

    return preds, final_model


# ── Main benchmark loop ────────────────────────────────────────────────────────

def run_benchmark(n_runs=3, task_filter=None):
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

    # Load transformers once — reused across all tasks
    transformers = build_transformers()

    all_results = {}

    print(f"{'='*60}")
    print(f"  Model:   MapLight + GNN")
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
                train, valid = group.get_train_valid_split(
                    benchmark=task_name,
                    split_type="default",
                    seed=run_idx * 42
                )

                gpu_before = get_gpu_memory_mib()
                t_start = time.time()

                preds, final_model = run_maplight_gnn(
                    transformers, train, valid, test, task_name, logs_dir
                )

                t_end = time.time()
                gpu_after = get_gpu_memory_mib()

                predictions = {task_name: preds}
                result = group.evaluate(predictions, benchmark=task_name)
                metric_name = list(result[task_name].keys())[0]
                score = result[task_name][metric_name]

                run_result = {
                    "run":            run_idx,
                    "metric":         metric_name,
                    "score":          round(float(score), 4),
                    "train_time_sec": round(t_end - t_start, 1),
                    "gpu_memory_mib": round(gpu_after - gpu_before, 1),
                }
                task_results.append(run_result)
                print(f"   Run {run_idx+1}/{n_runs} | {metric_name}: {score:.4f} | "
                      f"time: {run_result['train_time_sec']}s | "
                      f"GPU: {run_result['gpu_memory_mib']} MiB")

                # Save model artifact from first run
                if run_idx == 0:
                    task_artifact_dir = os.path.join(artifacts_dir, task_name)
                    os.makedirs(task_artifact_dir, exist_ok=True)
                    final_model.save_model(os.path.join(task_artifact_dir, "model.cbm"))

            scores = [r["score"] for r in task_results]
            times  = [r["train_time_sec"] for r in task_results]
            gpus   = [r["gpu_memory_mib"] for r in task_results]

            summary = {
                "task":                task_name,
                "model":               "MapLight_GNN",
                "metric":              task_results[0]["metric"],
                "score_mean":          round(float(np.mean(scores)), 4),
                "score_std":           round(float(np.std(scores)), 4),
                "train_time_sec_mean": round(float(np.mean(times)), 1),
                "gpu_memory_mib_mean": round(float(np.mean(gpus)), 1),
                "runs":                task_results,
                "timestamp":           datetime.now().isoformat(),
            }
            all_results[task_name] = summary

            log_path = os.path.join(logs_dir, f"{task_name}.json")
            with open(log_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"   ✓ Mean: {summary['score_mean']} ± {summary['score_std']}\n")

        except Exception as e:
            print(f"   ✗ FAILED: {e}\n")
            error_result = {
                "task":      task_name,
                "model":     "MapLight_GNN",
                "error":     str(e),
                "traceback": traceback.format_exc(),
            }
            all_results[task_name] = error_result
            log_path = os.path.join(logs_dir, f"{task_name}_error.json")
            with open(log_path, "w") as f:
                json.dump(error_result, f, indent=2)

    combined_path = os.path.join(logs_dir, "_all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✓ Done. All results saved to: {combined_path}")
    return all_results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MapLight+GNN TDC ADMET Benchmarking Script")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per task (default: 3)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task only (e.g. --task hia_hou)")
    args = parser.parse_args()

    run_benchmark(n_runs=args.runs, task_filter=args.task)