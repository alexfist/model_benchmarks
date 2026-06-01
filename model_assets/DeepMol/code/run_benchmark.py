"""
run_benchmark.py (DeepMol)
--------------------------
Runs DeepMol (AutoML) across all 22 TDC ADMET tasks.

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
import pandas as pd
import joblib
from datetime import datetime

from tdc.benchmark_group import admet_group
from deepmol.datasets.datasets import SmilesDataset
from deepmol.compound_featurization import MorganFingerprint
from deepmol.models import SklearnModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error
import xgboost as xgb


# ── GPU memory tracking ────────────────────────────────────────────────────────

def get_gpu_memory_mib():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_dataset(smiles, labels = None):
    return SmilesDataset(smiles=smiles, y=labels)

def featurize_dataset(dataset, featurizer):
    return featurizer.featurize(dataset)
def is_classification_task(y):
    unique_vals = np.unique(y)
    return len(unique_vals) <= 2 or set(unique_vals).issubset({0, 1})


# ── DeepMol AutoML runner ─────────────────────────────────────────────────────

def run_deepmol(train, valid, test, task_name, logs_dir):
    """
    Tries multiple featurizer + model combos on validation set.
    Returns: preds, best_model, best_combo_label
    """
    is_clf = is_classification_task(train["Y"].values)

    featurizers = {
        "morgan_2048": MorganFingerprint(radius=2, size=2048),
        "morgan_1024": MorganFingerprint(radius=2, size=1024),
    }

    if is_clf:
        models = {
            "rf_100": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "rf_200": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "xgb":    xgb.XGBClassifier(n_estimators=100, random_state=42,
                                         use_label_encoder=False, eval_metric="logloss"),
        }
    else:
        models = {
            "rf_100": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "rf_200": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "xgb":    xgb.XGBRegressor(n_estimators=100, random_state=42),
        }

    tuning_logs    = []
    best_score     = None
    best_combo     = None
    best_model     = None
    best_feat_name = None
    higher_is_better = is_clf

    train_dataset = make_dataset(train["Drug"].tolist(), train["Y"].tolist())
    valid_dataset = make_dataset(valid["Drug"].tolist(), valid["Y"].tolist())

    for feat_name, featurizer in featurizers.items():
        try:
            train_feat = featurize_dataset(train_dataset, featurizer)
            valid_feat = featurize_dataset(valid_dataset, featurizer)
            if train_feat is None or valid_feat is None:
                print(f"    Featurizer {feat_name} failed during featurization.")
        except Exception as e:
            print(f"     Featurizer {feat_name} failed: {e}")
            continue

        for model_name, clf in models.items():
            combo_label = f"{feat_name}+{model_name}"
            try:
                dm_model = SklearnModel(
                    model=clf,
                    task="classification" if is_clf else "regression"
                )
                dm_model.fit(train_dataset)
                val_preds = dm_model.predict(valid_dataset)

                val_score = (roc_auc_score(valid["Y"].values, val_preds) if is_clf
                             else mean_absolute_error(valid["Y"].values, val_preds))

                tuning_logs.append({
                    "featurizer": feat_name,
                    "model":      model_name,
                    "combo":      combo_label,
                    "val_score":  round(float(val_score), 4),
                })
                print(f"     {combo_label}: {val_score:.4f}")

                is_better = (
                    best_score is None or
                    (higher_is_better and val_score > best_score) or
                    (not higher_is_better and val_score < best_score)
                )
                if is_better:
                    best_score     = val_score
                    best_combo     = combo_label
                    best_model     = dm_model
                    best_feat_name = feat_name

            except Exception as e:
                print(f"     {combo_label} failed: {e}")
                tuning_logs.append({"combo": combo_label, "error": str(e)})

    # Save tuning log
    tuning_log_path = os.path.join(logs_dir, f"{task_name}_tuning.json")
    with open(tuning_log_path, "w") as f:
        json.dump({
            "task":           task_name,
            "model":          "DeepMol",
            "tuning_runs":    tuning_logs,
            "best_combo":     best_combo,
            "best_val_score": round(float(best_score), 4) if best_score else None,
        }, f, indent=2)

    print(f"   Best: {best_combo} | val: {best_score:.4f}")

    # Retrain on train + valid combined
    train_valid_df      = pd.concat([train, valid]).reset_index(drop=True)
    train_valid_dataset = make_dataset(train_valid_df["Drug"].tolist(), train_valid_df["Y"].tolist())
    test_dataset        = make_dataset(test["Drug"].tolist(), test["Y"].tolist())

    train_valid_dataset = featurize_dataset(train_valid_dataset, featurizers[best_feat_name])
    test_dataset        = featurize_dataset(test_dataset, featurizers[best_feat_name])
   

    best_model.fit(train_valid_dataset)
    preds = best_model.predict(test_dataset)
    preds = np.array(preds).flatten

    return preds, best_model, best_combo


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

    all_results = {}

    print(f"{'='*60}")
    print(f"  Model:   DeepMol (AutoML)")
    print(f"  Tasks:   {len(benchmark_names)}")
    print(f"  Runs:    {n_runs} (seeds 1-{n_runs})")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

                gpu_before = get_gpu_memory_mib()
                t_start    = time.time()

                preds, best_model, best_combo = run_deepmol(
                    train, valid, test, task_name, logs_dir
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
                    joblib.dump(best_model, os.path.join(task_artifact_dir, "model.pkl"))
                    with open(os.path.join(task_artifact_dir, "hyperparams.json"), "w") as f:
                        json.dump({
                            "task":       task_name,
                            "model":      "DeepMol",
                            "best_combo": best_combo,
                            "seed":       seed,
                            "timestamp":  datetime.now().isoformat(),
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
                "model":               "DeepMol",
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
                "task": task_name, "model": "DeepMol",
                "error": str(e), "traceback": traceback.format_exc(),
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
    parser = argparse.ArgumentParser(description="DeepMol TDC ADMET Benchmarking Script")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per task (default: 5)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task only (e.g. --task hia_hou)")
    args = parser.parse_args()
    run_benchmark(n_runs=args.runs, task_filter=args.task)