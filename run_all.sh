#!/bin/bash
# =============================================================================
# run_all.sh
# ----------
# Runs all 5 TDC ADMET benchmark models sequentially.
# Each model is run in its own conda environment.
#
# Usage:
#   bash run_all.sh                  # run all models, all tasks
#   bash run_all.sh --task hia_hou   # run all models on a single task
#   bash run_all.sh --runs 5         # run all models with 5 seeds per task
#   bash run_all.sh --model minimol  # run a single model only
#
# Requirements:
#   - All conda environments must be set up first (see each model's README)
#   - Run check_install.py for each model before running this script
# =============================================================================

set -e  # exit immediately on error

# ── Parse arguments ───────────────────────────────────────────────────────────

TASK=""
RUNS=5
MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)   TASK="$2";  shift 2 ;;
        --runs)   RUNS="$2";  shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Build optional flags string
EXTRA_FLAGS="--runs $RUNS"
if [ -n "$TASK" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --task $TASK"
fi

# ── Helper functions ──────────────────────────────────────────────────────────

run_model() {
    local model_name=$1
    local env_name=$2
    local script_path=$3

    echo ""
    echo "============================================================"
    echo "  Running: $model_name"
    echo "  Environment: $env_name"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    # Activate environment and run script
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"

    python "$script_path" $EXTRA_FLAGS

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ $model_name completed successfully"
    else
        echo "  ✗ $model_name failed with exit code $exit_code"
        echo "  Check logs in model_benchmarks/$model_name/logs/ for details"
    fi

    conda deactivate
    return $exit_code
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  TDC ADMET Full Benchmark Run"
echo "  Models: MiniMol, DeepMol, AttrMasking, ZairaChem, MapLight+GNN"
echo "  Runs per task: $RUNS"
if [ -n "$TASK" ]; then
    echo "  Task filter: $TASK"
fi
if [ -n "$MODEL" ]; then
    echo "  Model filter: $MODEL"
fi
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Track which models passed and failed
PASSED=()
FAILED=()

run_if_selected() {
    local model_name=$1
    local env_name=$2
    local script_path=$3

    # Skip if --model flag is set and doesn't match
    if [ -n "$MODEL" ] && [ "$MODEL" != "$model_name" ] && [ "$MODEL" != "$(echo $model_name | tr '[:upper:]' '[:lower:]')" ]; then
        return 0
    fi

    if run_model "$model_name" "$env_name" "$script_path"; then
        PASSED+=("$model_name")
    else
        FAILED+=("$model_name")
    fi
}

# ── Run each model ────────────────────────────────────────────────────────────

run_if_selected \
    "MiniMol" \
    "minimol_env" \
    "model_benchmarks/MiniMol/code/run_benchmark.py"

run_if_selected \
    "DeepMol" \
    "deepmol_env" \
    "model_benchmarks/DeepMol/code/run_benchmark.py"

run_if_selected \
    "AttrMasking" \
    "attrmasking_env" \
    "model_benchmarks/AttrMasking/code/run_benchmark.py"

run_if_selected \
    "ZairaChem" \
    "zairachem" \
    "model_benchmarks/ZairaChem/code/run_benchmark.py"

run_if_selected \
    "MapLight_GNN" \
    "maplight_env" \
    "model_benchmarks/MapLight_GNN/code/run_benchmark.py"

# ── Generate summary reports ──────────────────────────────────────────────────

if [ -z "$MODEL" ]; then
    echo ""
    echo "============================================================"
    echo "  Generating summary reports..."
    echo "============================================================"

    # Use minimol_env for the summary script (only needs pandas/numpy)
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate minimol_env
    python results/generate_summary.py
    conda deactivate

    echo "  ✓ Summary reports saved to results/"
fi

# ── Final report ──────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Run Complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Passed: ${#PASSED[@]} — ${PASSED[*]}"
echo "  Failed: ${#FAILED[@]} — ${FAILED[*]}"
echo "============================================================"

# Exit with error if any model failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi