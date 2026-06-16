# TDC ADMET Benchmarking Project

Internal benchmarking of models from the TDC (Therapeutics Data Commons) ADMET leaderboard. The goal is to reproduce top leaderboard results across 22 standardized drug property prediction tasks, identify which models perform best per task, and provide a clean reproducible benchmark suite for downstream use in ADMET property prediction of new drug candidates.

**Repository:** https://github.com/alexfist/minimol_benchmark  
**Server:** H100 GPU cluster (192-core Intel Xeon, 2TB RAM, CUDA 12.8)

---

## Models

| Model | Status | Tasks | Notes |
|---|---|---|---|
| MiniMol | ✅ Complete | 22/22 | Data leakage flagged — scores may be inflated |
| DeepMol | ✅ Complete | 22/22 | |
| MapLight_GNN | ✅ Complete | 22/22 | Most trustworthy baseline |
| AttrMasking | ✅ Complete | 22/22 | |
| AttentiveFP | ✅ Complete | 22/22 | |
| BasicML | ✅ Complete | 22/22 | |
| CaliciBoost | ✅ Complete | 1/22 | Caco-2 only by design |
| BaseBoosting | ❌ Blocked | 0/22 | olorenchemengine requires GitHub access |
| ZairaChem | ❌ Blocked | 0/22 | Ersilia Hub requires GitHub + DockerHub access |

---

## Repository Structure

```
model_benchmarks/
├── README.md                       # This file
├── results/
│   ├── README.md                   # Deviations, reproducibility notes, blocked models
│   ├── generate_summary.py         # Compiles logs into CSV and Excel outputs
│   ├── benchmark_comparison.csv    # Best model per task
│   ├── benchmark_results.csv       # All models × all tasks
│   ├── benchmark_results.xlsx      # Excel workbook with one sheet per model
│   └── model_csvs/                 # One CSV per model
│       ├── MiniMol.csv
│       ├── DeepMol.csv
│       └── ...
└── model_assets/
    ├── MiniMol/
    │   ├── README.md               # Model description, deviations, setup instructions
    │   ├── data/                   # TDC datasets (auto-downloaded or symlinked)
    │   ├── code/
    │   │   ├── run_benchmark.py    # Main benchmarking script
    │   │   ├── check_install.py    # Installation verification
    │   │   └── requirements.txt    # Dependencies + setup instructions
    │   ├── artifacts/              # Saved models + hyperparams per task
    │   └── logs/                   # Per-task result JSONs + tuning logs
    ├── DeepMol/        (same structure)
    ├── MapLight_GNN/   (same structure)
    ├── AttrMasking/    (same structure)
    ├── AttentiveFP/    (same structure)
    ├── BasicML/        (same structure)
    ├── CaliciBoost/    (same structure — caco2_wang only)
    ├── BaseBoosting/   (code only — could not run)
    └── ZairaChem/      (code only — could not run)
```

---

## How to Run a Model

Every model follows the same three-step process. See each model's `README.md` for the exact conda environment name and any model-specific notes.

### Step 1 — Set up the environment

Each model has its own conda environment due to conflicting dependencies. Follow the setup instructions in `model_assets/<ModelName>/code/requirements.txt`.

```bash
conda activate <model_env>
```

### Step 2 — Verify installation

```bash
cd model_assets/<ModelName>/code
python check_install.py
```

All checks should pass before running the full benchmark. If any fail, the model's README has troubleshooting steps.

### Step 3 — Run the benchmark

```bash
# Full benchmark (run in background — some models take 24+ hours)
nohup python run_benchmark.py > ~/<model_name>.log 2>&1 &
disown

# Single task for testing
python run_benchmark.py --task hia_hou --runs 1

# Check progress
tail -f ~/<model_name>.log
ls ../logs/*.json | grep -v tuning | grep -v error | wc -l
```

---

## Data

TDC datasets are downloaded automatically on first run to each model's `data/` folder. Since `dataverse.harvard.edu` is blocked on the company server, subsequent models symlink to an already-downloaded copy:

```bash
cd model_assets/<ModelName>
ln -s ../MiniMol/data data
```

---

## Generating Results

Once one or more models have finished, generate the summary outputs:

```bash
cd results
python generate_summary.py --logs ../model_assets --output .
```

This produces:
- `benchmark_comparison.csv` — best model per task
- `benchmark_results.csv` — all models × all tasks in one flat table
- `model_csvs/<model>.csv` — one CSV per model
- `benchmark_results.xlsx` — Excel workbook with one sheet per model

---

## TDC Protocol

All models follow the standard TDC evaluation protocol:
- Scaffold split (`split_type="default"`)
- 5 seeds (seeds 1–5)
- `group.evaluate_many()` for official scoring
- Mean ± std reported across 5 seeds

---

## Hardware

| | |
|---|---|
| GPU | NVIDIA H100 × 8 |
| CPU | Intel Xeon Platinum 8558P (192 cores) |
| RAM | 2TB |
| CUDA | 12.8 |
| OS | Ubuntu 24 |

---

## References

- [TDC Platform](https://tdcommons.ai)
- [TDC ADMET Leaderboard](https://tdcommons.ai/benchmark/admet_group/overview/)
- [MiniMol](https://github.com/graphcore-research/minimol)
- [DeepMol](https://github.com/BioSystemsUM/DeepMol)
- [MapLight](https://github.com/maplightrx/MapLight-TDC)
- [AttrMasking / AttentiveFP (DeepPurpose)](https://github.com/kexinhuang12345/DeepPurpose)
- [CaliciBoost](https://github.com/Calici/CaliciBoost)
- [ZairaChem](https://github.com/ersilia-os/zaira-chem)
- [Oloren ChemEngine](https://github.com/Oloren-AI/olorenchemengine)