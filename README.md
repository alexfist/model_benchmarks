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

**Benchmark groups:**
- ADMET Benchmark Group (22 tasks)
- Toxicity Group

---

## Repository Structure

```
minimol_benchmark/
│
├── README.md                        # This file
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

Each model has its own `requirements.txt` inside its `code/` folder. It is recommended to create a separate conda environment per model to avoid dependency conflicts.

```bash
conda create -n <model_name>_env python=3.10
conda activate <model_name>_env
pip install -r model_assets/<model_name>/code/requirements.txt
```

### 2. Running a Model

Each model folder contains a `run_benchmark.py` script that loops through all TDC ADMET tasks automatically.

```bash
cd model_assets/<model_name>/code
python run_benchmark.py
```

Results are saved to `../logs/` and `../artifacts/`.

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
| OS | Linux (Ubuntu 22.04) |
| GPU | |
| CUDA Version | |
| Python Version | |
| Run Date | |

---

## Notes

- All tasks use TDC's default scaffold split for fair comparison
- Each task is run 3 times with different random seeds; mean ± std is reported
- Training time and GPU memory are measured per task per model
- Models with known data leakage issues (e.g. MiniMol) are noted in their individual README

---

## References

- [TDC Platform](https://tdcommons.ai)
- [TDC Leaderboard](https://tdcommons.ai/benchmark/admet_group/overview/)
- [MiniMol GitHub](https://github.com/graphcore-research/minimol)
- [DeepMol GitHub](https://github.com/BioSystemsUM/DeepMol)
- [ZairaChem GitHub](https://github.com/ersilia-os/zaira-chem)
- [MapLight GitHub](https://github.com/maplightrx/MapLight-TDC)
- [AttrMasking Paper](https://arxiv.org/abs/1905.12265)
