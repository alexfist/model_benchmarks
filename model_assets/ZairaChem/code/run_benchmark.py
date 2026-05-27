"""
run_benchmark.py (ZairaChem)
-----------------------------
Runs ZairaChem across TDC ADMET classification tasks.
Regression tasks are skipped (ZairaChem v1 classification only).

Usage:
    python run_benchmark.py
    python run_benchmark.py --runs 5
    python run_benchmark.py --task hia_hou

Outputs:
    ../data/          TDC datasets saved as CSVs
    ../logs/          per-task JSON result logs
    ../artifacts/     saved ZairaChem model folders per task
"""

import os
import json
import time
import shutil
import argparse
import tempfile
import traceback
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime

from tdc.benchmark_group import admet_group


# ── Regression tasks to skip ───────────────────────────────────────────────────

REGRESSION_TASKS = {
    "caco2_wang", "lipophilicity_astrazeneca", "solubility_aqsoldb",
    "ppbr_az", "vdss_lombardo", "half_life_obach",
    "clearance_hepatocyte_az", "clearance_microsome_az",
}


# ── GPU memory tracking ────────────────────────────────────────────────────────

def get_gpu_memory_mib():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


# ── ZairaChem runner ───────────────────────────────────────────────────────────

def run_zairachem(train, valid, test, task_name, artifacts_dir):
    train_combined = pd.concat([train, valid]).reset_index(drop=True)
    train_combined = train_combined.rename(columns={"Drug": "smiles", "Y": "activity"})
    test_renamed   = test.rename(columns={"Drug": "smiles", "Y": "activity"})

    tmp_dir    = tempfile.mkdtemp()
    train_path = os.path.join(tmp_dir, "train.csv")
    test_path  = os.path.join(tmp_dir, "test.csv")
    model_dir  = os.path.join(tmp_dir, "model")
    pred_dir   = os.path.join(tmp_dir, "predictions")

    try:
        train_combined.to_csv(train_path, index=False)
        test_renamed.to_csv(test_path, index=False)

        fit_result = subprocess.run(
            ["zairachem", "fit", "-i", train_path, "-m", model_dir],
            capture_output=True, text=True
        )
        if fit_result.returncode != 0:
            raise RuntimeError(f"zairachem fit failed:\n{fit_result.stderr}")

        pred_result = subprocess.run(
            ["zairachem", "predict", "-i", test_path, "-m", model_dir, "-o", pred_dir],
            capture_output=True, text=True
        )
        if pred_result.returncode != 0:
            raise RuntimeError(f"zairachem predict failed:\n{pred_result.stderr}")

        pred_file = os.path.join(pred_dir, "predictions.csv")
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Predictions file not found: {pred_file}")

        preds_df = pd.read_csv(pred_file)
        prob_col = None
        for col in preds_df.columns:
            if any(k in col.lower() for k in ["prob", "pred", "score"]):
                prob_col = col
                break
        if prob_col is None:
            prob_col = preds_df.columns[-1]

        preds = preds_df[prob_col].values

        # Save model artifact
        task_artifact_dir = os.path.join(artifacts_dir, task_name)
        if os.path.exists(task_artifact_dir):
            shutil.rmtree(task_artifact_dir)
        shutil.copytree(model_dir, task_artifact_dir)

        # Save hyperparams (ZairaChem is fully automated — no manual params)
        with open(os.path.join(task_artifact_dir, "hyperparams.json"), "w") as f:
            json.dump({
                "task":      task_name,
                "model":     "ZairaChem",
                "params":    "automated — ZairaChem manages its own hyperparameter search",
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        return preds

    finally:
        shutil.rmtree(tmp_dir)


# ── Main benchmark loop ────────────────────────────────────────────────────────

def run_benchmark(n_runs=5, task_filter=None):
    data_dir      = "../data"
    logs_dir      = "../logs"
    artifacts_dir = "../artifacts"

    os.makedirs(data_dir, exist_ok=True)
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
    print(f"  Model:   ZairaChem")
    print(f"  Tasks:   {len(benchmark_names)} total ({len(REGRESSION_TASKS)} regression skipped)")
    print(f"  Runs:    {n_runs} (seeds 1-{n_runs})")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for task_name in benchmark_names:
        print(f"── Task: {task_name}")

        if task_name in REGRESSION_TASKS:
            print(f"   ⚠ Skipped — ZairaChem v1 supports classification only\n")
            all_results[task_name] = {
                "task": task_name, "model": "ZairaChem",
                "status": "skipped",
                "reason": "ZairaChem v1 does not support regression tasks",
            }
            with open(os.path.join(logs_dir, f"{task_name}.json"), "w") as f:
                json.dump(all_results[task_name], f, indent=2)
            continue

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

                preds = run_zairachem(train, valid, test, task_name, artifacts_dir)

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

            results    = group.evaluate_many(predictions_list)
            mean_score = results[task_name][0]
            std_score  = results[task_name][1]

            single_result = group.evaluate({task_name: predictions_list[0][task_name]})
            metric_name   = list(single_result[task_name].keys())[0]

            times = [t["train_time_sec"] for t in timing_list]
            gpus  = [t["gpu_memory_mib"] for t in timing_list]

            summary = {
                "task":                task_name,
                "model":               "ZairaChem",
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
                "task": task_name, "model": "ZairaChem",
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
    parser = argparse.ArgumentParser(description="ZairaChem TDC ADMET Benchmarking Script")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per task (default: 5)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task only (e.g. --task hia_hou)")
    args = parser.parse_args()
    run_benchmark(n_runs=args.runs, task_filter=args.task)