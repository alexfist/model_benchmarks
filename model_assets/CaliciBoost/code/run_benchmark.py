"""
run_benchmark.py (CaliciBoost)
-------------------------------
Reproduces CaliciBoost (Huong Van Le et al., 2025) for the TDC caco2_wang task.

NOTE: CaliciBoost only covers caco2_wang — it was developed specifically for
Caco-2 permeability prediction and does not cover all 22 ADMET tasks.

Pipeline:
    SMILES → PaDEL descriptors (2D + 3D, 1875 features) → feature selection
    via permutation importance → XGBoost Regressor with Bayesian-optimized
    hyperparameters → predict on test.

Key paper details:
    - Best model: XGBoost Regressor on top-ranked PaDEL descriptors
    - PaDEL includes both 2D and 3D molecular features (1875 total)
    - Feature selection reduces dimensionality before final model fit
    - TDC leaderboard score: MAE = 0.256 ± 0.006 (rank 1)
    - Critical assessment score: MAE = 0.271 ± 0.002 (still rank 1)

Usage:
    python run_benchmark.py
    python run_benchmark.py --runs 5

Outputs:
    ../data/          TDC datasets (auto-downloaded)
    ../logs/          per-task JSON result logs + tuning logs
    ../artifacts/     saved model.pkl + hyperparams.json
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

from tdc.benchmark_group import admet_group
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

try:
    from padelpy import from_smiles
    PADEL_AVAILABLE = True
except ImportError:
    PADEL_AVAILABLE = False

# Task CaliciBoost covers
TASK_NAME = "caco2_wang"

# XGBoost hyperparameters from the paper (Bayesian-optimized)
# Paper reports best model as XGBoost on top-ranked PaDEL features
BEST_PARAMS = {
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "verbosity":        0,
}

# Number of top features to select (from paper's feature selection step)
N_TOP_FEATURES = 200


def compute_padel_descriptors(smiles_list):
    """
    Compute PaDEL 2D + 3D descriptors for a list of SMILES using batch mode.
    Writes all SMILES to a temp file and calls PaDEL once — much faster than
    calling from_smiles() per molecule.

    Returns (feature_matrix, feature_names).
    NaN/Inf values are filled with 0.
    """
    import tempfile
    from padelpy import padeldescriptor

    if not PADEL_AVAILABLE:
        raise RuntimeError(
            "padelpy not installed. Run: pip install padelpy\n"
            "PaDEL-Descriptor Java software also required — see README."
        )

    print(f"   Computing PaDEL descriptors for {len(smiles_list)} molecules (batch mode)...")

    # Write SMILES to a temp file — PaDEL reads this in one Java call
    with tempfile.NamedTemporaryFile(mode="w", suffix=".smi",
                                     delete=False) as smi_file:
        smi_path = smi_file.name
        for i, smi in enumerate(smiles_list):
            smi_file.write(f"{smi}\tmol{i}\n")

    out_csv = smi_path.replace(".smi", "_padel.csv")

    try:
        padeldescriptor(
            mol_dir=smi_path,
            d_file=out_csv,
            d_2d=True,
            d_3d=True,
            fingerprints=False,
            convert3d=True,
            retainorder=True,
            threads=8,
        )

        df = pd.read_csv(out_csv)

        # Drop the Name column
        if "Name" in df.columns:
            df = df.drop(columns=["Name"])

        feature_names = list(df.columns)
        X = df.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"   Done — {X.shape[0]} molecules × {X.shape[1]} descriptors")
        return X, feature_names

    finally:
        # Clean up temp files
        for f in [smi_path, out_csv]:
            try:
                os.remove(f)
            except Exception:
                pass


def select_top_features(X_train, y_train, feature_names, n_top=N_TOP_FEATURES):
    """
    Select top features using permutation importance on a quick RF model.
    Returns (selected_indices, selected_names).
    """
    from sklearn.ensemble import RandomForestRegressor

    print(f"   Selecting top {n_top} features via permutation importance...")
    quick_model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    quick_model.fit(X_train, y_train)

    result = permutation_importance(
        quick_model, X_train, y_train,
        n_repeats=5, random_state=42, n_jobs=-1
    )
    importances = result.importances_mean

    top_indices = np.argsort(importances)[::-1][:n_top]
    top_names   = [feature_names[i] for i in top_indices]
    print(f"   Selected {len(top_indices)} features")
    return top_indices, top_names


def run_caliciboost(train, valid, test, logs_dir):
    """
    Full CaliciBoost pipeline:
    1. Compute PaDEL descriptors
    2. Select top features via permutation importance
    3. Tune XGBoost hyperparameters (light grid around paper's best params)
    4. Retrain on train+valid, predict on test
    """
    # ── Featurize ─────────────────────────────────────────────────────────────
    X_train, feature_names = compute_padel_descriptors(train["Drug"].tolist())
    X_valid, _             = compute_padel_descriptors(valid["Drug"].tolist())
    X_test,  _             = compute_padel_descriptors(test["Drug"].tolist())

    y_train = train["Y"].values
    y_valid = valid["Y"].values

    # ── Feature selection ──────────────────────────────────────────────────────
    top_idx, top_names = select_top_features(
        X_train, y_train, feature_names, n_top=N_TOP_FEATURES
    )
    X_train_sel = X_train[:, top_idx]
    X_valid_sel = X_valid[:, top_idx]
    X_test_sel  = X_test[:, top_idx]

    # ── Light hyperparameter grid around paper's best params ──────────────────
    param_grid = [
        {"n_estimators": 300,  "learning_rate": 0.05, "max_depth": 5},
        {"n_estimators": 500,  "learning_rate": 0.05, "max_depth": 6},
        {"n_estimators": 500,  "learning_rate": 0.1,  "max_depth": 6},
        {"n_estimators": 1000, "learning_rate": 0.03, "max_depth": 6},
    ]

    tuning_logs = []
    best_score  = None
    best_params = None

    for params in param_grid:
        combo = {**BEST_PARAMS, **params}
        combo_label = f"n={params['n_estimators']}, lr={params['learning_rate']}, depth={params['max_depth']}"
        try:
            model = XGBRegressor(**combo)
            model.fit(X_train_sel, y_train)
            val_preds = model.predict(X_valid_sel)
            val_score = mean_absolute_error(y_valid, val_preds)

            tuning_logs.append({
                "combo":     combo_label,
                "params":    params,
                "val_score": round(float(val_score), 4),
            })
            print(f"     {combo_label}: MAE={val_score:.4f}")

            if best_score is None or val_score < best_score:
                best_score  = val_score
                best_params = combo

        except Exception as e:
            print(f"     {combo_label} failed: {e}")
            tuning_logs.append({"combo": combo_label, "error": str(e)})

    with open(os.path.join(logs_dir, f"{TASK_NAME}_tuning.json"), "w") as f:
        json.dump({
            "task":           TASK_NAME,
            "model":          "CaliciBoost",
            "n_features_in":  len(feature_names) if feature_names else 0,
            "n_features_sel": N_TOP_FEATURES,
            "tuning_runs":    tuning_logs,
            "best_params":    best_params,
            "best_val_score": round(float(best_score), 4) if best_score else None,
        }, f, indent=2)

    if best_params is None:
        raise RuntimeError("All tuning combos failed")

    print(f"   Best: MAE={best_score:.4f}")

    # ── Retrain on train + valid ───────────────────────────────────────────────
    X_all = np.vstack([X_train_sel, X_valid_sel])
    y_all = np.concatenate([y_train, y_valid])

    final_model = XGBRegressor(**best_params)
    final_model.fit(X_all, y_all)
    preds = np.array(final_model.predict(X_test_sel),
                     dtype=np.float64).flatten().tolist()

    return preds, final_model, best_params, top_names


# ── Main benchmark loop ────────────────────────────────────────────────────────

def run_benchmark(n_runs=5):
    data_dir      = "../data"
    logs_dir      = "../logs"
    artifacts_dir = "../artifacts"

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    if not PADEL_AVAILABLE:
        print("ERROR: padelpy not installed.")
        print("  pip install padelpy")
        print("  Also requires Java + PaDEL-Descriptor — see README.")
        return {}

    group = admet_group(path=data_dir)

    print(f"{'='*60}")
    print(f"  Model:   CaliciBoost")
    print(f"  Task:    {TASK_NAME} (Caco-2 only)")
    print(f"  Runs:    {n_runs} (seeds 1-{n_runs})")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    print(f"── Task: {TASK_NAME}")

    try:
        benchmark        = group.get(TASK_NAME)
        test             = benchmark["test"]
        predictions_list = []
        timing_list      = []

        for run_idx in range(n_runs):
            seed = run_idx + 1

            train, valid = group.get_train_valid_split(
                benchmark=TASK_NAME,
                split_type="default",
                seed=seed
            )

            t_start = time.time()

            preds, final_model, best_params, top_names = run_caliciboost(
                train, valid, test, logs_dir
            )

            t_end = time.time()

            predictions_list.append({TASK_NAME: preds})
            timing_list.append({
                "run":            seed,
                "train_time_sec": round(t_end - t_start, 1),
                "gpu_memory_mib": 0.0,
            })

            print(f"   Seed {seed}/{n_runs} | time: {round(t_end - t_start, 1)}s")

            if run_idx == 0:
                task_artifact_dir = os.path.join(artifacts_dir, TASK_NAME)
                os.makedirs(task_artifact_dir, exist_ok=True)
                joblib.dump(final_model,
                            os.path.join(task_artifact_dir, "model.pkl"))
                with open(os.path.join(task_artifact_dir, "hyperparams.json"), "w") as f:
                    json.dump({
                        "task":              TASK_NAME,
                        "model":             "CaliciBoost",
                        "params":            best_params,
                        "top_features":      top_names[:20],  # save top 20 names
                        "n_features_used":   N_TOP_FEATURES,
                        "seed":              seed,
                        "timestamp":         datetime.now().isoformat(),
                    }, f, indent=2)

        # evaluate_many needs 5 runs; fall back for < 5
        single_result = group.evaluate({TASK_NAME: predictions_list[0][TASK_NAME]})
        metric_name   = list(single_result[TASK_NAME].keys())[0]

        if len(predictions_list) == 5:
            results    = group.evaluate_many(predictions_list)
            mean_score = results[TASK_NAME][0]
            std_score  = results[TASK_NAME][1]
        else:
            mean_score = list(single_result[TASK_NAME].values())[0]
            std_score  = 0.0

        times = [t["train_time_sec"] for t in timing_list]

        summary = {
            "task":                TASK_NAME,
            "model":               "CaliciBoost",
            "metric":              metric_name,
            "score_mean":          round(float(mean_score), 4),
            "score_std":           round(float(std_score), 4),
            "train_time_sec_mean": round(float(np.mean(times)), 1),
            "gpu_memory_mib_mean": 0.0,
            "runs":                timing_list,
            "timestamp":           datetime.now().isoformat(),
        }

        with open(os.path.join(logs_dir, f"{TASK_NAME}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        combined_path = os.path.join(logs_dir, "_all_results.json")
        with open(combined_path, "w") as f:
            json.dump({TASK_NAME: summary}, f, indent=2)

        print(f"   ✓ Mean: {mean_score:.4f} ± {std_score:.4f}  [{metric_name}]\n")
        print(f"✓ Done. Results saved to: {combined_path}")
        return {TASK_NAME: summary}

    except Exception as e:
        print(f"   ✗ FAILED: {e}\n")
        error_result = {
            "task":      TASK_NAME,
            "model":     "CaliciBoost",
            "error":     str(e),
            "traceback": traceback.format_exc(),
        }
        with open(os.path.join(logs_dir, f"{TASK_NAME}_error.json"), "w") as f:
            json.dump(error_result, f, indent=2)
        return {TASK_NAME: error_result}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CaliciBoost TDC Caco-2 Benchmarking Script")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs (default: 5)")
    args = parser.parse_args()
    run_benchmark(n_runs=args.runs)