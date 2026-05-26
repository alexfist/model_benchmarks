"""
generate_summary.py
-------------------
Reads all model logs and produces two deliverable CSV reports:

1. results/summary_by_model.csv   — per-model scores across all 22 tasks
2. results/top3_by_task.csv       — top 3 models per task with compute stats

Usage:
    python generate_summary.py
    python generate_summary.py --logs ./logs --output ./results
"""

import os
import json
import argparse
import pandas as pd
from glob import glob

MODELS = ["MiniMol", "DeepMol", "MapLight_GNN", "ZairaChem", "AttrMasking"]

# TDC ADMET tasks in order
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
    "cyp2c19_veith",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp1a2_veith",
    "cyp2c9_veith",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "half_life_obach",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "herg",
    "ames",
    "dili",
]


def load_all_results(logs_dir):
    """Load all per-task JSON logs for all models. Returns a flat list of result dicts."""
    records = []

    for model_name in MODELS:
        model_log_dir = os.path.join(logs_dir, model_name)
        if not os.path.exists(model_log_dir):
            print(f"  ⚠ No logs found for {model_name} — skipping")
            continue

        log_files = glob(os.path.join(model_log_dir, "*.json"))
        log_files = [f for f in log_files if not os.path.basename(f).startswith("_")]

        for log_file in log_files:
            with open(log_file) as f:
                result = json.load(f)

            if "error" in result:
                print(f"  ✗ {model_name} / {result.get('task', '?')} failed: {result['error']}")
                continue

            records.append({
                "model":              model_name,
                "task":               result["task"],
                "metric":             result.get("metric", ""),
                "score_mean":         result.get("score_mean"),
                "score_std":          result.get("score_std"),
                "train_time_sec":     result.get("train_time_sec_mean"),
                "gpu_memory_mib":     result.get("gpu_memory_mib_mean"),
            })

    return pd.DataFrame(records)


def generate_summary_by_model(df, output_dir):
    """
    Table 1: rows = tasks, columns = models
    Cell value: "mean ± std"
    """
    rows = []
    for task in TASKS:
        task_df = df[df["task"] == task]
        row = {"task": task}
        for model in MODELS:
            model_row = task_df[task_df["model"] == model]
            if model_row.empty:
                row[model] = "-"
            else:
                mean = model_row.iloc[0]["score_mean"]
                std  = model_row.iloc[0]["score_std"]
                metric = model_row.iloc[0]["metric"]
                row[f"{model} ({metric})"] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "summary_by_model.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"  ✓ Saved: {out_path}")
    return summary_df


def generate_top3_by_task(df, output_dir):
    """
    Table 2: for each task, the top 3 models ranked by score.
    Includes model name, metric, score, training time, GPU memory.
    """
    # For classification tasks higher is better (AUROC),
    # for regression tasks lower is better (MAE). 
    # We detect this by metric name.
    regression_metrics = {"mae", "rmse", "mse", "spearman"}

    rows = []
    for task in TASKS:
        task_df = df[df["task"] == task].copy()
        if task_df.empty:
            continue

        metric = task_df.iloc[0]["metric"].lower() if not task_df.empty else ""
        ascending = any(m in metric for m in regression_metrics)

        task_df = task_df.sort_values("score_mean", ascending=ascending).reset_index(drop=True)

        for rank, (_, row) in enumerate(task_df.head(3).iterrows(), start=1):
            rows.append({
                "task":             row["task"],
                "rank":             rank,
                "model":            row["model"],
                "metric":           row["metric"],
                "score":            f"{row['score_mean']:.4f} ± {row['score_std']:.4f}",
                "train_time_min":   f"{row['train_time_sec'] / 60:.1f}" if row["train_time_sec"] else "-",
                "gpu_memory_mib":   f"{row['gpu_memory_mib']:.0f}" if row["gpu_memory_mib"] else "-",
            })

    top3_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "top3_by_task.csv")
    top3_df.to_csv(out_path, index=False)
    print(f"  ✓ Saved: {out_path}")
    return top3_df


def print_preview(df, n=10):
    """Print first n rows for quick sanity check."""
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.head(n).to_string(index=False))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark summary reports")
    parser.add_argument("--logs",   type=str, default="./logs",    help="Directory containing model logs")
    parser.add_argument("--output", type=str, default="./results", help="Directory to save summary CSVs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("\nLoading logs...")
    df = load_all_results(args.logs)

    if df.empty:
        print("\n⚠ No results found. Run run_benchmark.py first.")
        exit(1)

    print(f"\nLoaded {len(df)} results across {df['model'].nunique()} models and {df['task'].nunique()} tasks.")

    print("\nGenerating summary_by_model.csv...")
    summary_df = generate_summary_by_model(df, args.output)
    print_preview(summary_df)

    print("\nGenerating top3_by_task.csv...")
    top3_df = generate_top3_by_task(df, args.output)
    print_preview(top3_df)

    print("\n✓ All reports generated successfully.")