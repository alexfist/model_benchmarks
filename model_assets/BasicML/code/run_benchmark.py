"""
run_benchmark.py (Basic ML)
----------------------------
Reproduces "Accountable Prediction of Drug ADMET Properties with Molecular
Descriptors" (Boral et al. 2022) across all 22 TDC ADMET tasks.

Pipeline:
    SMILES → 31 RDKit 2D descriptors → model selection from 10 sklearn/XGBoost
    algorithms → RandomizedSearchCV tuning on validation set → retrain on
    train+valid → predict on test.

Key detail from the paper: each task gets its own best model chosen from
{LinearRegression, LogisticRegression, KNN, DecisionTree, RandomForest,
ExtraTrees, Bagging, AdaBoost, GradientBoosting, XGBoost}.

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
import warnings
import argparse
import traceback
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from tdc.benchmark_group import admet_group
from sklearn.metrics import roc_auc_score, mean_absolute_error

# ── Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier

# ── Regressors
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
)
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ── The 31 RDKit descriptors from the original paper (Appendix B) ────────────
# From Chem.Descriptors and Chem.Lipinski modules as described in the paper.
DESCRIPTOR_NAMES = [
    # Molecular weight / composition
    "MolWt", "HeavyAtomMolWt", "ExactMolWt",
    "NumValenceElectrons", "NumRadicalElectrons",
    # Partial charges
    "MaxPartialCharge", "MinPartialCharge",
    "MaxAbsPartialCharge", "MinAbsPartialCharge",
    # Morgan density (topological)
    "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
    # Lipinski / drug-likeness
    "NumHDonors", "NumHAcceptors", "MolLogP", "MolMR",
    # Topological / shape
    "TPSA", "LabuteASA", "BalabanJ", "BertzCT",
    "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    # Connectivity
    "Chi0", "Chi0n", "Chi1", "Chi1n",
    # Ring / structural
    "NumAromaticRings", "NumRotatableBonds", "RingCount",
]


def compute_descriptors(smiles_list):
    """Compute 31 RDKit descriptors for a list of SMILES strings.
    Returns a numpy array (n_molecules, 31). NaN/Inf → 0."""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([0.0] * len(DESCRIPTOR_NAMES))
            continue
        row = []
        for name in DESCRIPTOR_NAMES:
            try:
                # Most are on Descriptors, a few on Lipinski
                if hasattr(Descriptors, name):
                    val = getattr(Descriptors, name)(mol)
                elif hasattr(Lipinski, name):
                    val = getattr(Lipinski, name)(mol)
                else:
                    val = 0.0
                val = float(val) if val is not None else 0.0
                if not np.isfinite(val):
                    val = 0.0
            except Exception:
                val = 0.0
            row.append(val)
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def is_classification_task(y):
    unique_vals = np.unique(y)
    return len(unique_vals) <= 2 or set(unique_vals).issubset({0, 1})


# ── Model pools ───────────────────────────────────────────────────────────────

def get_clf_candidates():
    """10 classifiers with lightweight RandomizedSearchCV param grids."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1, 10, 100]}
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]}
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
        ),
        "Bagging": (
            BaggingClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [10, 20, 50], "max_samples": [0.5, 0.75, 1.0]}
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=42),
            {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="logloss",
                          verbosity=0, use_label_encoder=False),
            {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1],
             "max_depth": [3, 5, 7]}
        ),
    }


def get_reg_candidates():
    """10 regressors with lightweight RandomizedSearchCV param grids."""
    return {
        "LinearRegression": (
            LinearRegression(),
            {}   # no hyperparams to tune
        ),
        "KNN": (
            KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]}
        ),
        "DecisionTree": (
            DecisionTreeRegressor(random_state=42),
            {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(random_state=42, n_jobs=-1),
            {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
        ),
        "Bagging": (
            BaggingRegressor(random_state=42, n_jobs=-1),
            {"n_estimators": [10, 20, 50], "max_samples": [0.5, 0.75, 1.0]}
        ),
        "AdaBoost": (
            AdaBoostRegressor(random_state=42),
            {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]}
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        ),
        "XGBoost": (
            XGBRegressor(random_state=42, verbosity=0),
            {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1],
             "max_depth": [3, 5, 7]}
        ),
    }


# ── BasicML runner ────────────────────────────────────────────────────────────

def run_basic_ml(train, valid, test, task_name, logs_dir):
    """
    1. Compute 31 RDKit descriptors for train/valid/test.
    2. Try all candidate models on validation set (with RandomizedSearchCV tuning).
    3. Pick best model.
    4. Retrain best model on train+valid, predict on test.
    Returns: preds, best_model, best_model_name, best_params
    """
    is_clf = is_classification_task(train["Y"].values)
    higher_is_better = is_clf

    X_train = compute_descriptors(train["Drug"].tolist())
    X_valid = compute_descriptors(valid["Drug"].tolist())
    X_test  = compute_descriptors(test["Drug"].tolist())

    y_train = train["Y"].values
    y_valid = valid["Y"].values

    candidates = get_clf_candidates() if is_clf else get_reg_candidates()

    tuning_logs  = []
    best_score   = None
    best_name    = None
    best_params  = None

    for model_name, (estimator, param_grid) in candidates.items():
        try:
            if param_grid:
                search = RandomizedSearchCV(
                    estimator, param_grid,
                    n_iter=min(8, len(param_grid) * 3),
                    cv=3,
                    scoring="roc_auc" if is_clf else "neg_mean_absolute_error",
                    random_state=42, n_jobs=-1, refit=True
                )
                search.fit(X_train, y_train)
                tuned = search.best_estimator_
                params = search.best_params_
            else:
                tuned = estimator
                tuned.fit(X_train, y_train)
                params = {}

            if is_clf:
                if hasattr(tuned, "predict_proba"):
                    val_preds = tuned.predict_proba(X_valid)[:, 1]
                else:
                    val_preds = tuned.decision_function(X_valid)
                val_score = roc_auc_score(y_valid, val_preds)
            else:
                val_preds = tuned.predict(X_valid)
                val_score = mean_absolute_error(y_valid, val_preds)

            tuning_logs.append({
                "model":     model_name,
                "params":    params,
                "val_score": round(float(val_score), 4),
            })
            print(f"     {model_name}: {val_score:.4f}")

            is_better = (
                best_score is None or
                (higher_is_better and val_score > best_score) or
                (not higher_is_better and val_score < best_score)
            )
            if is_better:
                best_score  = val_score
                best_name   = model_name
                best_params = params

        except Exception as e:
            print(f"     {model_name} failed: {e}")
            tuning_logs.append({"model": model_name, "error": str(e)})

    with open(os.path.join(logs_dir, f"{task_name}_tuning.json"), "w") as f:
        json.dump({
            "task":           task_name,
            "model":          "BasicML",
            "tuning_runs":    tuning_logs,
            "best_model":     best_name,
            "best_params":    best_params,
            "best_val_score": round(float(best_score), 4) if best_score else None,
        }, f, indent=2)

    if best_name is None:
        raise RuntimeError(f"All models failed for {task_name}")

    print(f"   Best: {best_name} {best_params} | val: {best_score:.4f}")

    # Retrain best model on train + valid combined
    candidates_fresh = get_clf_candidates() if is_clf else get_reg_candidates()
    final_estimator, _ = candidates_fresh[best_name]

    # Apply best hyperparams to fresh estimator
    if best_params:
        final_estimator.set_params(**best_params)

    X_all = np.vstack([X_train, X_valid])
    y_all = np.concatenate([y_train, y_valid])
    final_estimator.fit(X_all, y_all)

    if is_clf:
        if hasattr(final_estimator, "predict_proba"):
            preds = np.array(final_estimator.predict_proba(X_test)[:, 1], dtype = np.float64).flatten().tolist()
        else:
            preds = np.array(final_estimator.decision_function(X_test), dtype = np.float64).flatten().tolist()
    else:
        preds = final_estimator.predict(X_test)

    return preds, final_estimator, best_name, best_params


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
    print(f"  Model:       Basic ML")
    print(f"  Features:    31 RDKit 2D descriptors")
    print(f"  Algorithms:  10 sklearn/XGBoost models per task")
    print(f"  Tasks:       {len(benchmark_names)}")
    print(f"  Runs:        {n_runs} (seeds 1-{n_runs})")
    print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for task_name in benchmark_names:
        print(f"── Task: {task_name}")

        try:
            benchmark        = group.get(task_name)
            test             = benchmark["test"]
            predictions_list = []
            timing_list      = []
            best_model_name_final = None

            for run_idx in range(n_runs):
                seed = run_idx + 1

                train, valid = group.get_train_valid_split(
                    benchmark=task_name,
                    split_type="default",
                    seed=seed
                )

                t_start = time.time()

                preds, final_model, best_name, best_params = run_basic_ml(
                    train, valid, test, task_name, logs_dir
                )

                t_end = time.time()

                predictions_list.append({task_name: preds})
                timing_list.append({
                    "run":            seed,
                    "train_time_sec": round(t_end - t_start, 1),
                    "best_model":     best_name,
                })

                print(f"   Seed {seed}/{n_runs} | "
                      f"time: {round(t_end - t_start, 1)}s | "
                      f"best: {best_name}")

                if run_idx == 0:
                    best_model_name_final = best_name
                    task_artifact_dir = os.path.join(artifacts_dir, task_name)
                    os.makedirs(task_artifact_dir, exist_ok=True)
                    joblib.dump(final_model,
                                os.path.join(task_artifact_dir, "model.pkl"))
                    with open(os.path.join(task_artifact_dir, "hyperparams.json"), "w") as f:
                        json.dump({
                            "task":       task_name,
                            "model":      "BasicML",
                            "algorithm":  best_name,
                            "params":     best_params,
                            "seed":       seed,
                            "timestamp":  datetime.now().isoformat(),
                        }, f, indent=2)

           # Get metric name from single evaluate
            single_result = group.evaluate({task_name: predictions_list[0][task_name]})
            metric_name   = list(single_result[task_name].keys())[0]

            # evaluate_many requires exactly 5 runs — fall back for testing
            if len(predictions_list) == 5:
                results    = group.evaluate_many(predictions_list)
                mean_score = results[task_name][0]
                std_score  = results[task_name][1]
            else:
                mean_score = list(single_result[task_name].values())[0]
                std_score  = 0.0

            times = [t["train_time_sec"] for t in timing_list]

            summary = {
                "task":                task_name,
                "model":               "BasicML",
                "algorithm":           best_model_name_final,
                "metric":              metric_name,
                "score_mean":          round(float(mean_score), 4),
                "score_std":           round(float(std_score), 4),
                "train_time_sec_mean": round(float(np.mean(times)), 1),
                "gpu_memory_mib_mean": 0.0,   # CPU only
                "runs":                timing_list,
                "timestamp":           datetime.now().isoformat(),
            }
            all_results[task_name] = summary

            with open(os.path.join(logs_dir, f"{task_name}.json"), "w") as f:
                json.dump(summary, f, indent=2)

            print(f"   ✓ Mean: {mean_score:.4f} ± {std_score:.4f}  "
                  f"[{metric_name}]  best algo: {best_model_name_final}\n")

        except Exception as e:
            print(f"   ✗ FAILED: {e}\n")
            error_result = {
                "task":      task_name,
                "model":     "BasicML",
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
    parser = argparse.ArgumentParser(description="BasicML TDC ADMET Benchmarking Script")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per task (default: 5)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single task only (e.g. --task hia_hou)")
    args = parser.parse_args()
    run_benchmark(n_runs=args.runs, task_filter=args.task)