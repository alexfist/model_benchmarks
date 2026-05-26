"""
run_benchmark.py
----------------
Runs DeepMol (AutoML) across all 22 TDC ADMET tasks and logs results.

Usage:
    python run_benchmark.py
    python run_benchmark.py --runs 3
    python run_benchmark.py --task hia_hou

Outputs:
    ../data/          TDC datasets (auto-downloaded)
    ../logs/          per-task JSON result logs + tuning logs
    ../artifacts/     saved best pipeline per task
"""

import os
import json
import time
import argparse
import traceback
import tempfile
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from tdc.benchmark_group import admet_group
from deepmol.loaders import CSVLoader
from deepmol.feature_engineering import MorganFingerprint, RDKitDescriptors
from deepmol.models import SklearnModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
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

def df_to_deepmol_dataset(df, smiles_field="Drug", label_field="Y"):
    """Converts a TDC dataframe to a DeepMol dataset via temp CSV."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        tmp_path = f.name
    loader = CSVLoader(
        dataset_path=tmp_path,
        smiles_field=smiles_field,
        labels_fields=[label_field]
    )
    dataset = loader.create_dataset()
    os.unlink(tmp_path)
    return dataset


def is_classification_task(y):
    unique_vals = np.unique(y)
    return len(unique_vals) <= 2 or set(unique_vals).issubset({0, 1})


# ── AutoML pipeline search ────────────────────────────────────────────────────

def run_deepmol_automl(train, valid, test, task_name, logs_dir):
    """
    Tries multiple featurizer + model combinations on the validation set,
    picks the best pipeline, retrains on train+valid, predicts on test.
    Logs all tuning results to ../logs/<task_name>_tuning.json
    """
    is_clf = is_classification_task(train["Y"].values)

    # Define pipeline combinations to try
    featurizers = {
        "morgan_2048": MorganFingerprint(radius=2, size=2048),
        "morgan_1024": MorganFingerprint(radius=2, size=1024),
        "rdkit_desc":  RDKitDescriptors(),
    }

    if is_clf:
        models = {
            "rf_100":   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "rf_200":   RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "xgb":      xgb.XGBClassifier(n_estimators=100, random_state=42,
                                           use_label_encoder=False, eval_metric="logloss"),
        }
    else:
        models = {
            "rf_100":   RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "rf_200":   RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "xgb":      xgb.XGBRegressor(n_estimators=100, random_state=42),
        }

    tuning_logs  = []
    best_score   = None
    best_combo   = None
    best_model   = None
    best_feat_name = None

    train_dataset = df_to_deepmol_dataset(train)
    valid_dataset = df_to_deepmol_dataset(valid)

    for feat_name, featurizer in featurizers.items():
        # Featurize train and valid
        try:
            featurizer.featurize(train_dataset)
            featurizer.featurize(valid_dataset)
        except Exception as e:
            print(f"     Featurizer {feat_name} failed: {e}")
            continue

        for model_name, clf in models.items():
            combo_label = f"{feat_name} + {model_name}"
            try:
                dm_model = SklearnModel(
                    model=clf,
                    mode="classification" if is_clf else "regression"
                )
                dm_model.fit(train_dataset)
                val_preds = dm_model.predict(valid_dataset)

                if is_clf:
                    val_score = roc_auc_score(valid["Y"].values, val_preds)
                    higher_is_better = True
                else:
                    val_score = mean_absolute_error(valid["Y"].values, val_preds)
                    higher_is_better = False

                tuning_logs.append({
                    "featurizer":  feat_name,
                    "model":       model_name,
                    "combo":       combo_label,
                    "val_score":   round(float(val_score), 4),
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
                tuning_logs.append({
                    "featurizer": feat_name,
                    "model":      model_name,
                    "combo":      combo_label,
                    "error":      str(e),
                })

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

    print(f"   Best pipeline: {best_combo} | val score: {best_score:.4f}")

    # Retrain best pipeline on train + valid combined
    train_valid_df = pd.concat([train, valid]).reset_index(drop=True)
    train_valid_dataset = df_to_deepmol_dataset(train_valid_df)
    test_dataset        = df_to_deepmol_dataset(test)

    featurizers[best_feat_name].featurize(train_valid_dataset)
    featurizers[best_feat_name].featurize(test_dataset)

    best_model.fit(train_valid_dataset)
    preds = best_model.predict(test_dataset)

    return preds, best_model


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

    all_results = {}

    print(f"{'='*60}")
    print(f"  Model:   DeepMol (AutoML)")
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

                preds, best_model = run_deepmol_automl(
                    train, valid, test, task_name, logs_dir
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

                # Save best model artifact from first run
                if run_idx == 0:
                    task_artifact_dir = os.path.join(artifacts_dir, task_name)
                    os.makedirs(task_artifact_dir, exist_ok=True)
                    joblib.dump(best_model, os.path.join(task_artifact_dir, "model.pkl"))

            scores = [r["score"] for r in task_results]
            times  = [r["train_time_sec"] for r in task_results]
            gpus   = [r["gpu_memory_mib"] for r in task_results]

            summary = {
                "task":                task_name,
                "model":               "DeepMol",
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
                "model":     "DeepMol",
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
    parser = argparse.ArgumentParser(description="DeepMol TDC ADMET Benchmarking Script")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per task (default: 3)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task only (e.g. --task hia_hou)")
    args = parser.parse_args()

    run_benchmark(n_runs=args.runs, task_filter=args.task)