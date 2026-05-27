"""
run_benchmark.py
----------------
Runs ZairaChem across TDC ADMET classification tasks.

NOTE: ZairaChem v1 supports classification tasks only.
Regression tasks are skipped and logged with a note.

Usage:
    python run_benchmark.py
    python run_benchmark.py --runs 3
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


# ── Task classification ────────────────────────────────────────────────────────

# ZairaChem v1 supports classification only
# These are the regression tasks in TDC ADMET — they will be skipped
REGRESSION_TASKS = {
    "caco2_wang",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "ppbr_az",
    "vdss_lombardo",
    "half_life_obach",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
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
    """
    Runs ZairaChem fit + predict via CLI.
    Returns predictions array and path to saved model folder.
    """
    train_combined = pd.concat([train, valid]).reset_index(drop=True)
    train_combined = train_combined.rename(columns={"Drug": "smiles", "Y": "activity"})
    test_renamed   = test.rename(columns={"Drug": "smiles", "Y": "activity"})

    tmp_dir   = tempfile.mkdtemp()
    train_path = os.path.join(tmp_dir, "train.csv")
    test_path  = os.path.join(tmp_dir, "test.csv")
    model_dir  = os.path.join(tmp_dir, "model")
    pred_dir   = os.path.join(tmp_dir, "predictions")

    try:
        train_combined.to_csv(train_path, index=False)
        test_renamed.to_csv(test_path, index=False)

        # Fit
        fit_result = subprocess.run(
            ["zairachem", "fit", "-i", train_path, "-m", model_dir],
            capture_output=True, text=True
        )
        if fit_result.returncode != 0:
            raise RuntimeError(f"zairachem fit failed:\n{fit_result.stderr}")

        # Predict
        pred_result = subprocess.run(
            ["zairachem", "predict", "-i", test_path, "-m", model_dir, "-o", pred_dir],
            capture_output=True, text=True
        )
        if pred_result.returncode != 0:
            raise RuntimeError(f"zairachem predict failed:\n{pred_result.stderr}")

        # Read predictions
        pred_file = os.path.join(pred_dir, "predictions.csv")
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Predictions file not found: {pred_file}")

        preds_df = pd.read_csv(pred_file)

        # ZairaChem outputs probability column — find it
        prob_col = None
        for col in preds_df.columns:
            if "prob" in col.lower() or "pred" in col.lower() or "score" in col.lower():
                prob_col = col
                break
        if prob_col is None:
            prob_col = preds_df.columns[-1]  # fallback to last column

        preds = preds_df[prob_col].values

        # Save model artifacts
        task_artifact_dir = os.path.join(artifacts_dir, task_name)
        if os.path.exists(task_artifact_dir):
            shutil.rmtree(task_artifact_dir)
        shutil.copytree(model_dir, task_artifact_dir)

        return preds

    finally:
        shutil.rmtree(tmp_dir)


# ── Main benchmark loop ────────────────────────────────────────────────────────

def run_benchmark(n_runs=3, task_filter=None):
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
    print(f"  Tasks:   {len(benchmark_names)} total")
    print(f"  Skipping regression tasks: {len(REGRESSION_TASKS)}")
    print(f"  Runs:    {n_runs} per task")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for task_name in benchmark_names:
        print(f"── Task: {task_name}")

        # Skip regression tasks
        if task_name in REGRESSION_TASKS:
            print(f"   ⚠ Skipped — ZairaChem v1 supports classification only\n")
            all_results[task_name] = {
                "task":   task_name,
                "model":  "ZairaChem",
                "status": "skipped",
                "reason": "ZairaChem v1 does not support regression tasks",
            }
            log_path = os.path.join(logs_dir, f"{task_name}.json")
            with open(log_path, "w") as f:
                json.dump(all_results[task_name], f, indent=2)
            continue

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

                preds = run_zairachem(train, valid, test, task_name, artifacts_dir)

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

            scores = [r["score"] for r in task_results]
            times  = [r["train_time_sec"] for r in task_results]
            gpus   = [r["gpu_memory_mib"] for r in task_results]

            summary = {
                "task":                task_name,
                "model":               "ZairaChem",
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
                "model":     "ZairaChem",
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
    parser = argparse.ArgumentParser(description="ZairaChem TDC ADMET Benchmarking Script")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per task (default: 3)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task only (e.g. --task hia_hou)")
    args = parser.parse_args()

    run_benchmark(n_runs=args.runs, task_filter=args.task)