"""
run_benchmark.py (MapLight + GNN)
----------------------------------
Runs MapLight+GNN across all 22 TDC ADMET tasks.
Pipeline: ECFP + Avalon + ErG + RDKit 2D + GIN embeddings → CatBoost

Usage:
    python run_benchmark.py
    python run_benchmark.py --runs 5
    python run_benchmark.py --task hia_hou

Outputs:
    ../data/          TDC datasets (auto-downloaded)
    ../logs/          per-task JSON result logs + tuning logs
    ../artifacts/     saved model.cbm + hyperparams.json per task
"""

import os
import json
import time
import argparse
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

os.environ["LD_LIBRARY_PATH"] = (
    "/opt/miniforge3/envs/minimol_env/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

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


def is_classification_task(y):
    unique_vals = np.unique(y)
    return len(unique_vals) <= 2 or set(unique_vals).issubset({0, 1})


# ── Featurization ──────────────────────────────────────────────────────────────

def build_transformers():
    print("Loading feature transformers...")
    transformers = {
        "ecfp":   MoleculeTransformer("ecfp"),
        "avalon": MoleculeTransformer("avalon"),
        "erg":    MoleculeTransformer("erg"),
        "desc2d": MoleculeTransformer("desc2D"),
        "gin":    PretrainedDGLTransformer(kind="gin_supervised_masking", dtype=float),
    }
    print("Transformers loaded.\n")
    return transformers


def featurize(smiles_list, transformers, allowed_names=None):
    """Featurize molecules. If allowed_names given, only use those transformers.
    Returns (feature_matrix, list_of_used_names)."""
    features   = []
    used_names = []
    n = len(smiles_list)
    for name, trans in transformers.items():
        if allowed_names is not None and name not in allowed_names:
            continue
        try:
            feat = np.array(trans(smiles_list))
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            if feat.shape[0] != n:
                print(f"  Warning: {name} returned {feat.shape[0]} rows for {n} molecules — skipping")
                continue
            features.append(feat)
            used_names.append(name)
        except Exception as e:
            print(f"  Warning: {name} failed: {e} — skipping")
    return np.hstack(features), used_names


def get_global_used_names(transformers, full_data):
    """Determine which featurizers work on the full dataset.
    Called once per task before the seed loop."""
    print("   Checking featurizers on full dataset...")
    _, used_names = featurize(full_data["Drug"].tolist(), transformers)
    print(f"   Valid featurizers: {used_names}")
    return used_names


# ── MapLight + GNN runner ─────────────────────────────────────────────────────

def run_maplight_gnn(transformers, train, valid, test, task_name, logs_dir, global_used_names):
    """
    Uses globally determined featurizers (consistent across all seeds),
    tunes CatBoost on validation set, retrains on train+valid, predicts on test.
    Returns: preds, final_model, best_params
    """
    is_clf           = is_classification_task(train["Y"].values)
    higher_is_better = is_clf

    # Featurize using globally valid featurizers — consistent across all seeds
    X_train, _ = featurize(train["Drug"].tolist(), transformers, allowed_names=global_used_names)
    X_valid, _ = featurize(valid["Drug"].tolist(), transformers, allowed_names=global_used_names)
    X_test, _  = featurize(test["Drug"].tolist(),  transformers, allowed_names=global_used_names)

    y_train = train["Y"].values
    y_valid = valid["Y"].values

    param_grid = [
        {"iterations": 500,  "learning_rate": 0.1},
        {"iterations": 1000, "learning_rate": 0.1},
        {"iterations": 1000, "learning_rate": 0.03},
        {"iterations": 2000, "learning_rate": 0.03},
    ]

    tuning_logs = []
    best_score  = None
    best_params = None

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
            tuning_logs.append({"combo": combo_label, "error": str(e)})

    with open(os.path.join(logs_dir, f"{task_name}_tuning.json"), "w") as f:
        json.dump({
            "task":           task_name,
            "model":          "MapLight_GNN",
            "tuning_runs":    tuning_logs,
            "best_params":    best_params,
            "best_val_score": round(float(best_score), 4) if best_score else None,
        }, f, indent=2)

    if best_params is None:
        raise RuntimeError(f"All tuning combos failed for {task_name}")

    print(f"   Best: {best_params} | val: {best_score:.4f}")

    # Retrain on train + valid combined
    # Safe to vstack since both use same global_used_names
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

    return preds, final_model, best_params


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

    transformers = build_transformers()

    all_results = {}

    print(f"{'='*60}")
    print(f"  Model:   MapLight + GNN")
    print(f"  Tasks:   {len(benchmark_names)}")
    print(f"  Runs:    {n_runs} (seeds 1-{n_runs})")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for task_name in benchmark_names:
        print(f"── Task: {task_name}")

        try:
            benchmark = group.get(task_name)
            test      = benchmark["test"]
            full_data = benchmark["train_val"]

            # Determine valid featurizers once for this task
            global_used_names = get_global_used_names(transformers, full_data)

            predictions_list = []
            timing_list      = []

            for run_idx in range(n_runs):
                seed = run_idx + 1

                train, valid = group.get_train_valid_split(
                    benchmark=task_name,
                    split_type="default",
                    seed=seed
                )

                gpu_before = get_gpu_memory_mib()
                t_start    = time.time()

                preds, final_model, best_params = run_maplight_gnn(
                    transformers, train, valid, test,
                    task_name, logs_dir, global_used_names
                )

                t_end     = time.time()
                gpu_after = get_gpu_memory_mib()

                predictions_list.append({task_name: preds})
                timing_list.append({
                    "run":            seed,
                    "train_time_sec": round(t_end - t_start, 1),
                    "gpu_memory_mib": round(gpu_after - gpu_before, 1),
                })

                print(f"   Seed {seed}/{n_runs} | "
                      f"time: {round(t_end - t_start, 1)}s | "
                      f"GPU: {round(gpu_after - gpu_before, 1)} MiB")

                if run_idx == 0:
                    task_artifact_dir = os.path.join(artifacts_dir, task_name)
                    os.makedirs(task_artifact_dir, exist_ok=True)
                    final_model.save_model(os.path.join(task_artifact_dir, "model.cbm"))
                    with open(os.path.join(task_artifact_dir, "hyperparams.json"), "w") as f:
                        json.dump({
                            "task":      task_name,
                            "model":     "MapLight_GNN",
                            "params":    best_params,
                            "seed":      seed,
                            "timestamp": datetime.now().isoformat(),
                        }, f, indent=2)

            results    = group.evaluate_many(predictions_list)
            mean_score = results[task_name][0]
            std_score  = results[task_name][1]

            single_result = group.evaluate({task_name: predictions_list[0][task_name]})
            metric_name   = list(single_result[task_name].keys())[0]

            times = [t["train_time_sec"] for t in timing_list]
            gpus  = [t["gpu_memory_mib"] for t in timing_list]

            summary = {
                "task":                task_name,
                "model":               "MapLight_GNN",
                "metric":              metric_name,
                "score_mean":          round(float(mean_score), 4),
                "score_std":           round(float(std_score), 4),
                "train_time_sec_mean": round(float(np.mean(times)), 1),
                "gpu_memory_mib_mean": round(float(np.mean(gpus)), 1),
                "runs":                timing_list,
                "timestamp":           datetime.now().isoformat(),
            }
            all_results[task_name] = summary

            with open(os.path.join(logs_dir, f"{task_name}.json"), "w") as f:
                json.dump(summary, f, indent=2)

            print(f"   ✓ Mean: {mean_score:.4f} ± {std_score:.4f}\n")

        except Exception as e:
            print(f"   ✗ FAILED: {e}\n")
            error_result = {
                "task":      task_name,
                "model":     "MapLight_GNN",
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
    parser = argparse.ArgumentParser(description="MapLight+GNN TDC ADMET Benchmarking Script")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per task (default: 5)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task only (e.g. --task hia_hou)")
    args = parser.parse_args()
    run_benchmark(n_runs=args.runs, task_filter=args.task)