"""
run_benchmark.py
----------------
Runs a model across all 22 TDC ADMET tasks and logs results.

Usage:
    python run_benchmark.py --model minimol
    python run_benchmark.py --model deepmol
    python run_benchmark.py --model all

Each run saves:
    - logs/<model_name>/<task_name>.json   (per-task results)
    - artifacts/<model_name>/              (saved model files)
"""

import os
import json
import time
import argparse
import traceback
import numpy as np
from datetime import datetime

# ── TDC ────────────────────────────────────────────────────────────────────────
from tdc.benchmark_group import admet_group

# ── GPU memory tracking ────────────────────────────────────────────────────────
def get_gpu_memory_mib():
    """Returns current GPU memory usage in MiB. Returns 0 if no GPU available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        # Apple Silicon MPS
        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 1024 / 1024
    except Exception:
        pass
    return 0.0

# ── Model runners ──────────────────────────────────────────────────────────────

def run_minimol(train, valid, test):
    """Two-stage pipeline: MiniMol fingerprints + RandomForest classifier/regressor."""
    from minimol import Minimol
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    model = Minimol()

    train_fps = model(train["Drug"].tolist()).numpy()
    valid_fps = model(valid["Drug"].tolist()).numpy()
    test_fps  = model(test["Drug"].tolist()).numpy()

    y_train = train["Y"].values
    y_valid = valid["Y"].values

    # Detect task type
    unique_vals = np.unique(y_train)
    is_classification = len(unique_vals) <= 2 or set(unique_vals).issubset({0, 1})

    if is_classification:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(np.vstack([train_fps, valid_fps]), np.concatenate([y_train, y_valid]))
        preds = clf.predict_proba(test_fps)[:, 1]
    else:
        clf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(np.vstack([train_fps, valid_fps]), np.concatenate([y_train, y_valid]))
        preds = clf.predict(test_fps)

    return preds


def run_attrmasking(train, valid, test):
    """AttrMasking: pretrained GNN with attribute masking. Requires DeepChem."""
    try:
        import deepchem as dc
        from deepchem.models import AttentiveFPModel

        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        X_train = featurizer.featurize(train["Drug"].tolist())
        X_valid = featurizer.featurize(valid["Drug"].tolist())
        X_test  = featurizer.featurize(test["Drug"].tolist())

        y_train = train["Y"].values
        unique_vals = np.unique(y_train)
        is_classification = len(unique_vals) <= 2

        if is_classification:
            dataset_train = dc.data.NumpyDataset(X_train, y_train.reshape(-1,1))
            dataset_test  = dc.data.NumpyDataset(X_test,  test["Y"].values.reshape(-1,1))
            model = AttentiveFPModel(mode="classification", n_tasks=1, batch_size=32)
        else:
            dataset_train = dc.data.NumpyDataset(X_train, y_train.reshape(-1,1))
            dataset_test  = dc.data.NumpyDataset(X_test,  test["Y"].values.reshape(-1,1))
            model = AttentiveFPModel(mode="regression", n_tasks=1, batch_size=32)

        model.fit(dataset_train, nb_epoch=30)
        preds = model.predict(dataset_test).flatten()
        return preds

    except ImportError:
        raise ImportError("AttrMasking requires deepchem. Install with: pip install deepchem")


# Map model names to their runner functions
MODEL_RUNNERS = {
    "minimol":    run_minimol,
    "attrmasking": run_attrmasking,
    # Add deepmol, zaira, maplight runners here when ready
}


# ── Main benchmark loop ────────────────────────────────────────────────────────

def run_benchmark(model_name, n_runs=3, output_dir="./logs"):
    """
    Runs a model across all 22 ADMET tasks, n_runs times each.
    Saves per-task JSON logs and a combined results file.
    """
    if model_name not in MODEL_RUNNERS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_RUNNERS.keys())}")

    runner = MODEL_RUNNERS[model_name]
    log_dir = os.path.join(output_dir, model_name)
    os.makedirs(log_dir, exist_ok=True)

    group = admet_group(path="./data")
    benchmark_names = group.dataset_names  # all 22 task names

    all_results = {}

    print(f"\n{'='*60}")
    print(f"  Running: {model_name.upper()}")
    print(f"  Tasks:   {len(benchmark_names)}")
    print(f"  Runs:    {n_runs} per task")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for task_name in benchmark_names:
        print(f"── Task: {task_name}")
        task_results = []

        try:
            benchmark = group.get(task_name)
            train_val, test = benchmark["train_val"], benchmark["test"]

            for run_idx in range(n_runs):
                # Fresh train/valid split each run
                train, valid = group.get_train_valid_split(
                    benchmark=task_name,
                    split_type="default",
                    seed=run_idx * 42
                )

                gpu_before = get_gpu_memory_mib()
                t_start = time.time()

                preds = runner(train, valid, test)

                t_end = time.time()
                gpu_after = get_gpu_memory_mib()

                # Evaluate using TDC's built-in evaluator
                predictions = {task_name: preds}
                result = group.evaluate(predictions, benchmark=task_name)
                metric_name = list(result[task_name].keys())[0]
                score = result[task_name][metric_name]

                run_result = {
                    "run": run_idx,
                    "metric": metric_name,
                    "score": round(float(score), 4),
                    "train_time_sec": round(t_end - t_start, 1),
                    "gpu_memory_mib": round(gpu_after - gpu_before, 1),
                }
                task_results.append(run_result)
                print(f"   Run {run_idx+1}/{n_runs} | {metric_name}: {score:.4f} | "
                      f"time: {run_result['train_time_sec']}s | "
                      f"GPU: {run_result['gpu_memory_mib']} MiB")

            # Aggregate across runs
            scores = [r["score"] for r in task_results]
            times  = [r["train_time_sec"] for r in task_results]
            gpus   = [r["gpu_memory_mib"] for r in task_results]

            summary = {
                "task": task_name,
                "model": model_name,
                "metric": task_results[0]["metric"],
                "score_mean": round(float(np.mean(scores)), 4),
                "score_std":  round(float(np.std(scores)), 4),
                "train_time_sec_mean": round(float(np.mean(times)), 1),
                "gpu_memory_mib_mean": round(float(np.mean(gpus)), 1),
                "runs": task_results,
                "timestamp": datetime.now().isoformat(),
            }
            all_results[task_name] = summary

            # Save per-task log immediately (so progress isn't lost on crash)
            log_path = os.path.join(log_dir, f"{task_name}.json")
            with open(log_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"   ✓ Mean: {summary['score_mean']} ± {summary['score_std']}\n")

        except Exception as e:
            print(f"   ✗ FAILED: {e}\n")
            all_results[task_name] = {
                "task": task_name,
                "model": model_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    # Save combined results file
    combined_path = os.path.join(log_dir, "_all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Done. Results saved to: {combined_path}")
    return all_results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TDC ADMET Benchmarking Script")
    parser.add_argument("--model", type=str, default="minimol",
                        help=f"Model to run. Options: {list(MODEL_RUNNERS.keys()) + ['all']}")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per task (default: 3)")
    parser.add_argument("--output", type=str, default="./logs",
                        help="Directory to save logs (default: ./logs)")
    args = parser.parse_args()

    if args.model == "all":
        for model_name in MODEL_RUNNERS:
            run_benchmark(model_name, n_runs=args.runs, output_dir=args.output)
    else:
        run_benchmark(args.model, n_runs=args.runs, output_dir=args.output)