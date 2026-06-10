# TDC ADMET Benchmarking Project

Internal benchmarking of models from the TDC (Therapeutics Data Commons) leaderboard across all ADMET and Toxicity tasks.

---

## Project Overview

This project reproduces and evaluates five models from the TDC leaderboard across 22 ADMET benchmark tasks. For each model, we collect performance metrics, training time, and GPU memory usage, then summarize the results to identify which models perform best on which tasks.

**Models evaluated:**
- MiniMol ✅
- DeepMol (AutoML) ✅
- MapLight + GNN ✅
- AttrMasking ✅
- ZairaChem ❌ — could not run due to network restrictions (see results/README.md)

**Benchmark groups:**
- ADMET Benchmark Group (22 tasks)

---

## Repository Structure

```
model_benchmarks/
│
├── README.md                        # This file
├── results/
│   ├── README.md                    # Deviations from original pipelines + reproducibility notes
│   ├── generate_summary.py          # Compiles logs into 3 deliverable CSV tables
│   ├── summary_by_model.csv         # (generated) per-model scores across all tasks
│   ├── top3_by_task.csv             # (generated) top 3 models per task
│   └── hyperparams_summary.csv      # (generated) best hyperparameters per model per task
│
└── model_assets/
    ├── MiniMol/
    │   ├── README.md
    │   ├── data/
    │   ├── code/
    │   │   ├── run_benchmark.py
    │   │   ├── check_install.py
    │   │   └── requirements.txt
    │   ├── artifacts/
    │   └── logs/
    ├── DeepMol/         (same structure)
    ├── MapLight_GNN/    (same structure)
    ├── AttrMasking/     (same structure)
    └── ZairaChem/       (same structure — not run due to network restrictions)
```

---

## Environment Setup

Each model requires its own conda environment due to conflicting dependencies. Environments must be set up manually — the `run_all.sh` script is provided as a reference but may not work on all systems due to conda activation differences.

### Create all environments

```bash
conda create -n minimol_env python=3.11 -y
conda create -n deepmol_env python=3.10 -y
conda create -n attrmasking_env python=3.8 -y
conda create -n maplight_env python=3.10 -y
```

### Install dependencies

Follow the `requirements.txt` in each model's `code/` folder — the install order and special flags matter. See each model's README for exact instructions.

**Important:** Do not use `pip install -r requirements.txt` directly for all models — some require conda installs, specific wheel URLs, or `--no-deps` flags that are documented in comments inside each requirements.txt.

---

## How to Run

### Step 1 — Verify installation

Run the check script for each model before running the full benchmark:

```bash
conda activate minimol_env
python model_assets/MiniMol/code/check_install.py

conda activate deepmol_env
python model_assets/DeepMol/code/check_install.py

conda activate attrmasking_env
python model_assets/AttrMasking/code/check_install.py

conda activate maplight_env
python model_assets/MapLight_GNN/code/check_install.py
```

### Step 2 — Run benchmarks

Run each model separately in its own environment. Use `nohup` for long-running models:

```bash
# MiniMol (~2-3 hours)
conda activate minimol_env
cd model_assets/MiniMol/code
nohup python run_benchmark.py > ~/model_benchmarks/minimol.log 2>&1 &
disown

# DeepMol (~3-4 hours)
conda activate deepmol_env
cd model_assets/DeepMol/code
nohup python run_benchmark.py > ~/model_benchmarks/deepmol.log 2>&1 &
disown

# MapLight+GNN (~2-3 hours)
conda activate maplight_env
cd model_assets/MapLight_GNN/code
nohup python run_benchmark.py > ~/model_benchmarks/maplight.log 2>&1 &
disown

# AttrMasking (~18-24 hours — run overnight)
conda activate attrmasking_env
cd model_assets/AttrMasking/code
nohup python run_benchmark.py > ~/model_benchmarks/attrmasking.log 2>&1 &
disown
```

To run a single task for testing:
```bash
python run_benchmark.py --task hia_hou
```

### Step 3 — Monitor progress

```bash
# Check completed tasks per model
ls model_assets/MiniMol/logs/ | grep -v tuning | grep "\.json" | wc -l
ls model_assets/DeepMol/logs/ | grep -v tuning | grep "\.json" | wc -l
ls model_assets/MapLight_GNN/logs/ | grep -v tuning | grep "\.json" | wc -l
ls model_assets/AttrMasking/logs/ | grep -v tuning | grep "\.json" | wc -l

# Follow live output
tail -f ~/model_benchmarks/attrmasking.log
```

### Step 4 — Generate summary reports

Once all models are done:

```bash
conda activate minimol_env
python results/generate_summary.py
```

This produces three CSV tables in `results/`:
- `summary_by_model.csv` — scores for each model across all 22 tasks
- `top3_by_task.csv` — top 3 models per task with time and GPU memory
- `hyperparams_summary.csv` — best hyperparameters and all combos tried per model per task

---

## Hardware Used

| | Details |
|---|---|
| OS | Linux (Ubuntu) |
| GPU | NVIDIA H100 x8 |
| CUDA Version | 12.8 |
| CPU | Intel Xeon Platinum 8558P (192 cores) |
| RAM | 2TB |

---

## TDC Protocol

All tasks follow TDC standard protocol:
- Scaffold split via `split_type="default"`
- Seeds [1, 2, 3, 4, 5]
- `group.evaluate_many()` for official scoring
- Mean ± std reported across 5 seeds

---

## References

- [TDC Platform](https://tdcommons.ai)
- [TDC Leaderboard](https://tdcommons.ai/benchmark/admet_group/overview/)
- [MiniMol GitHub](https://github.com/graphcore-research/minimol)
- [DeepMol GitHub](https://github.com/BioSystemsUM/DeepMol)
- [ZairaChem GitHub](https://github.com/ersilia-os/zaira-chem)
- [MapLight GitHub](https://github.com/maplightrx/MapLight-TDC)
- [AttrMasking Paper](https://arxiv.org/abs/1905.12265)