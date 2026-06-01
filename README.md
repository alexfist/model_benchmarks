# TDC ADMET Benchmarking Project

Internal benchmarking of models from the TDC (Therapeutics Data Commons) leaderboard across all ADMET and Toxicity tasks.

---

## Project Overview

This project reproduces and evaluates five models from the TDC leaderboard across 22 ADMET benchmark tasks. For each model, we collect performance metrics, training time, and GPU memory usage, then summarize the results to identify which models perform best on which tasks.

**Models evaluated:**
- MiniMol
- DeepMol (AutoML)
- MapLight + GNN
- ZairaChem
- AttrMasking

**Benchmark group:**
- ADMET Benchmark Group (22 tasks)

---

## Repository Structure

```
model_benchmarks/
│
├── README.md                        # This file
│
├── run_all.sh                       # Bash script to run all models AFTER environment intialization
│
├── results/
│   ├── summary_by_model.csv         # Per-model performance across all tasks
│   └── top3_by_task.csv             # Top 3 models per task with metrics
│
└── model_assets/
    ├── MiniMol/
    │   ├── README.md                # Model-specific documentation
    │   ├── data/                    # Standardized datasets for each task
    │   ├── code/                    # Training & inference scripts + requirements.txt
    │   ├── artifacts/               # Saved model files (model.pkl or model.pt)
    │   └── logs/                    # Tuning logs & test set evaluation reports
    │
    ├── DeepMol/
    │   └── ...
    ├── MapLight_GNN/
    │   └── ...
    ├── ZairaChem/
    │   └── ...
    └── AttrMasking/
        └── ...
```

---

## How to Reproduce Results

### 1. Environment Setup

Each model has its own `requirements.txt` inside its `code/` folder. You MUST create a separate conda environment per model to avoid dependency conflicts.

If you are in mainland China or want a faster PyPI mirror, configure pip to use the Tsinghua source first:

```bash
# Windows
setup_pip_tsinghua.bat

# Or manually
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Then create and activate the environment, and install dependencies:

```bash
conda create -n <model_name>_env python=3.10
conda activate <model_name>_env
pip install -r model_assets/<model_name>/code/requirements.txt
```

### 2. Running a Model

If you are using Windows CMD or PowerShell, run the benchmark with the Windows launcher instead of the bash script:

```bat
run_all.bat
```

The batch file resolves Conda automatically and uses the same environment names as the bash runner.


Each model folder contains a `run_benchmark.py` script that trains and evaluates the model through all 22 ADMET tasks individually. Note: for the ZairaChem model, regression tasks are left out and only classification tasks are included.

For Linux/macOS users, the existing `run_all.sh` workflow is still supported.

```bash
cd model_assets/<model_name>/code
python run_benchmark.py
```

Results are saved to `../logs/` and `../artifacts/`. These include some basic toying with the hyperparameters

### 3. Generating Summary Reports

Once all models have been run, generate the summary tables:

```bash
python results/generate_summary.py
```

This produces:
- `results/summary_by_model.csv` — each model's scores across all tasks
- `results/top3_by_task.csv` — top 3 models per task with compute stats

---

## Results Summary

### Per-Model Performance

See `results/summary_by_model.csv` for full results.

Example format:

| Task | MiniMol | DeepMol | MapLight+GNN | ZairaChem | AttrMasking |
|------|---------|---------|--------------|-----------|-------------|
| HIA_Hou | 0.993 | - | - | - | - |
| ... | ... | ... | ... | ... | ... |

### Top 3 Models Per Task

See `results/top3_by_task.csv` for full results.

Example format:

| Task | Rank | Model | Metric | Train Time | GPU Memory |
|------|------|-------|--------|------------|------------|
| HIA_Hou | 1 | MiniMol | 0.993 | 31 min | 437 MiB |
| ... | ... | ... | ... | ... | ... |

---

## Hardware & Environment

| | Details |
|---|---|
| OS | N/A |
| GPU | N/A |
| CUDA Version | N/A |
| Python Version | N/A |
| Run Date | N/A |

---

## Notes

- All tasks use TDC's default scaffold split for fair comparison
- Each task is run 5 times with random seeds 1,2,3,4,5 as outlined in the TDC requirements; mean ± std is reported afterwards
- Training time and GPU memory are measured per task per model
- Models with known data leakage issues (e.g. MiniMol) are noted in their individual README
- If your device has no GPU, then it is recommended to modify the AttrMasking model such that its parameter grid is 2x2 instead of 4x4. All other models should run fine on CPU only.
- Additionally, if you have a slower computer, it is recommended to run each model benchmark individually instead of running the bash script.
---

## References

- [TDC Platform](https://tdcommons.ai)
- [TDC Leaderboard](https://tdcommons.ai/benchmark/admet_group/overview/)
- [MiniMol GitHub](https://github.com/graphcore-research/minimol)
- [DeepMol GitHub](https://github.com/BioSystemsUM/DeepMol)
- [ZairaChem GitHub](https://github.com/ersilia-os/zaira-chem)
- [MapLight GitHub](https://github.com/maplightrx/MapLight-TDC)
- [AttrMasking Paper](https://arxiv.org/abs/1905.12265)
