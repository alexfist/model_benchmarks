"""
run_benchmark.py (AttrMasking)
------------------------------
Runs AttrMasking (DGL_GIN_AttrMasking via DeepPurpose) across all 22 TDC ADMET tasks.

Usage:
    python run_benchmark.py
    python run_benchmark.py --runs 5
    python run_benchmark.py --task hia_hou

Outputs:
    ../data/          TDC datasets (auto-downloaded)
    ../logs/          per-task JSON result logs + tuning logs
    ../artifacts/     saved model + hyperparams.json per task
"""
import dgl.data.utils
_original_download = dgl.data.utils.download
def _patched_download(url, path=None, overwrite=False, **kwargs):
    return _original_download(url,path=path, overwrite=overwrite, **kwargs)
dgl.data.utils.download = _patched_download

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import time
import argparse
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

from tdc.benchmark_group import admet_group
from DeepPurpose import utils, CompoundPred
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


# ── AttrMasking runner with tuning ────────────────────────────────────────────

def run_attrmasking(train, valid, test, task_name, logs_dir):
    """
    Fine-tunes DGL_GIN_AttrMasking with hyperparameter search on validation set.
    Returns: preds, final_model, best_params
    """
    drug_encoding    = "DGL_GIN_AttrMasking"
    is_clf           = is_classification_task(train["Y"].values)
    higher_is_better = is_clf

    param_grid = [
    {"lr": 1e-3, "epochs": 20},
    {"lr": 1e-4, "epochs": 30},
    ]

    tuning_logs = []
    best_score  = None
    best_params = None

    train_dp = utils.data_process(
        X_drug=train["Drug"].tolist(),
        y=train["Y"].tolist(),
        drug_encoding=drug_encoding,
        split_method="no_split"
    )
    valid_dp = utils.data_process(
        X_drug=valid["Drug"].tolist(),
        y=valid["Y"].tolist(),
        drug_encoding=drug_encoding,
        split_method="no_split"
    )

    for params in param_grid:
        combo_label = f"lr={params['lr']}, epochs={params['epochs']}"
        try:
            config = utils.generate_config(
                drug_encoding=drug_encoding,
                train_epoch=params["epochs"],
                LR=params["lr"],
                batch_size=32,
            )
            model = CompoundPred.model_initialize(**config)
            model.train(train_dp, valid_dp, valid_dp)
            val_preds = model.predict(valid_dp)

            val_score = (roc_auc_score(valid["Y"].values, val_preds) if is_clf
                         else mean_absolute_error(valid["Y"].values, val_preds))

            tuning_logs.append({
                "lr": params["lr"], "epochs": params["epochs"],
                "combo": combo_label, "val_score": round(float(val_score), 4),
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

    # Save tuning log
    with open(os.path.join(logs_dir, f"{task_name}_tuning.json"), "w") as f:
        json.dump({
            "task": task_name, "model": "AttrMasking",
            "tuning_runs": tuning_logs, "best_params": best_params,
            "best_val_score": round(float(best_score), 4) if best_score else None,
        }, f, indent=2)

    print(f"   Best: lr={best_params['lr']}, epochs={best_params['epochs']} | val: {best_score:.4f}")

    # Retrain on train + valid combined
    train_valid_df = pd.concat([train, valid]).reset_index(drop=True)
    train_valid_dp = utils.data_process(
        X_drug=train_valid_df["Drug"].tolist(),
        y=train_valid_df["Y"].tolist(),
        drug_encoding=drug_encoding,
        split_method="no_split"
    )
    test_dp = utils.data_process(
        X_drug=test["Drug"].tolist(),
        y=test["Y"].tolist(),
        drug_encoding=drug_encoding,
        split_method="no_split"
    )

    config = utils.generate_config(
        drug_encoding=drug_encoding,
        train_epoch=best_params["epochs"],
        LR=best_params["lr"],
        batch_size=32
    )
    final_model = CompoundPred.model_initialize(**config)
    final_model.train(train_valid_dp, test_dp, test_dp)
    preds = final_model.predict(test_dp)

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

    all_results = {}

    print(f"{'='*60}")
    print(f"  Model:   AttrMasking (DGL_GIN_AttrMasking)")
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

                preds, final_model, best_params = run_attrmasking(
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
                    final_model.save_model(os.path.join(task_artifact_dir, "model"))
                    with open(os.path.join(task_artifact_dir, "hyperparams.json"), "w") as f:
                        json.dump({
                            "task": task_name, "model": "AttrMasking",
                            "params": best_params, "seed": seed,
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
                "model":               "AttrMasking",
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
                "task": task_name, "model": "AttrMasking",
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
    parser = argparse.ArgumentParser(description="AttrMasking TDC ADMET Benchmarking Script")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per task (default: 5)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task only (e.g. --task hia_hou)")
    args = parser.parse_args()
    run_benchmark(n_runs=args.runs, task_filter=args.task)
