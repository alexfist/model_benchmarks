"""
generate_summary.py
-------------------
Produces a benchmark results table matching the ZairaChem paper format.
One row per task, showing the best model and its score.

Output: results/benchmark_comparison.csv

Usage:
    python generate_summary.py
    python generate_summary.py --logs ./model_assets --output ./results
"""

import os
import json
import argparse
import pandas as pd
from glob import glob

MODELS = ["MiniMol", "DeepMol", "MapLight_GNN", "AttrMasking", "ZairaChem"]

TASKS = [
    "caco2_wang",
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "bbb_martins",
    "ppbr_az",
    "vdss_lombardo",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp2c9_veith",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "half_life_obach",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "ld50_zhu",
    "herg",
    "ames",
    "dili",
]

LOWER_IS_BETTER = {
    "caco2_wang",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "ppbr_az",
    "half_life_obach",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "ld50_zhu",
}

TASK_TYPE = {
    "caco2_wang":                    "regression",
    "hia_hou":                       "binary",
    "pgp_broccatelli":               "binary",
    "bioavailability_ma":            "binary",
    "lipophilicity_astrazeneca":     "regression",
    "solubility_aqsoldb":            "regression",
    "bbb_martins":                   "binary",
    "ppbr_az":                       "regression",
    "vdss_lombardo":                 "regression",
    "cyp2d6_veith":                  "binary",
    "cyp3a4_veith":                  "binary",
    "cyp2c9_veith":                  "binary",
    "cyp2c9_substrate_carbonmangels":"binary",
    "cyp2d6_substrate_carbonmangels":"binary",
    "cyp3a4_substrate_carbonmangels":"binary",
    "half_life_obach":               "regression",
    "clearance_hepatocyte_az":       "regression",
    "clearance_microsome_az":        "regression",
    "ld50_zhu":                      "regression",
    "herg":                          "binary",
    "ames":                          "binary",
    "dili":                          "binary",
}


def load_all_results(logs_dir):
    records = {}
    for model_name in MODELS:
        model_log_dir = os.path.join(logs_dir, model_name, "logs")
        if not os.path.exists(model_log_dir):
            continue

        log_files = glob(os.path.join(model_log_dir, "*.json"))
        log_files = [f for f in log_files
                     if not os.path.basename(f).startswith("_")
                     and "_tuning" not in os.path.basename(f)]

        for log_file in log_files:
            with open(log_file) as f:
                result = json.load(f)

            if "error" in result or result.get("status") == "skipped":
                continue

            task = result.get("task")
            if task not in records:
                records[task] = {}

            records[task][model_name] = {
                "score_mean": result.get("score_mean"),
                "score_std":  result.get("score_std"),
                "metric":     result.get("metric", ""),
            }

    return records


def generate_comparison_table(records, output_dir):
    rows = []

    for task in TASKS:
        task_records = records.get(task, {})
        task_type = TASK_TYPE.get(task, "unknown")
        lower_is_better = task in LOWER_IS_BETTER

        # Get metric name from any available model
        metric = "-"
        for model in MODELS:
            if model in task_records and task_records[model]["metric"]:
                metric = task_records[model]["metric"]
                break

        # Find best model
        best_score = None
        best_model = None
        best_std   = None

        for model in MODELS:
            if model not in task_records:
                continue
            mean = task_records[model]["score_mean"]
            std  = task_records[model]["score_std"]
            if mean is None:
                continue

            is_better = (
                best_score is None or
                (lower_is_better and mean < best_score) or
                (not lower_is_better and mean > best_score)
            )
            if is_better:
                best_score = mean
                best_model = model
                best_std   = std

        rows.append({
            "benchmark":          task,
            "model":              best_model if best_model else "-",
            "task_type":          task_type,
            "leaderboard_metric": metric,
            "score":              round(best_score, 3) if best_score is not None else "-",
            "std":                round(best_std, 3) if best_std is not None else "-",
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "benchmark_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"✓ Saved: {out_path}")

    # Print model win summary
    wins = df[df["model"] != "-"]["model"].value_counts()
    print(f"\nBest model per task summary ({len(df)} tasks):")
    for model, count in wins.items():
        print(f"  {model}: {count} tasks")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs",   type=str, default="./model_assets")
    parser.add_argument("--output", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading logs...")
    records = load_all_results(args.logs)
    print(f"Loaded results for {len(records)} tasks.")

    print("\nGenerating benchmark_comparison.csv...")
    df = generate_comparison_table(records, args.output)

    print("\nFull table:")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.to_string(index=False))