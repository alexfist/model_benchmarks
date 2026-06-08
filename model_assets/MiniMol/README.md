# MiniMol

## Overview

MiniMol is a 10-million-parameter molecular fingerprinting model developed by Graphcore Research, pre-trained on over 3,300 biological and quantum tasks covering approximately 6 million molecules.

It uses a two-stage pipeline:
1. **Stage 1 (MiniMol)** — converts a molecule's SMILES string into a 512-dimensional fingerprint vector using a Graph Neural Network
2. **Stage 2 (downstream predictor)** — a lightweight model (e.g. Random Forest) trained on those fingerprints to predict a specific ADMET property

The same MiniMol encoder (Stage 1) is reused across all 22 ADMET tasks. Only the downstream predictor (Stage 2) changes per task.

**Paper:** [MiniMol: A Parameter-Efficient Foundation Model for Molecular Learning](https://arxiv.org/abs/2404.14986)
**GitHub:** https://github.com/graphcore-research/minimol

---

## Known Issues

- MiniMol has been flagged for potential data leakage in a 2026 assessment — some molecules in its pre-training data may be too similar to TDC test set molecules, which could inflate benchmark scores, though this is addressed in the paper linked above
- Results should be interpreted with this caveat in mind
- Not suitable for biologics (antibodies, proteins) — small molecules only

---

## Environment Setup

```bash
conda create -n minimol_env python=3.11
conda activate minimol_env

# Install PyTorch geometric packages with correct CUDA wheel
pip install torch==2.9.0
pip install torch-scatter torch-sparse torch-cluster -f https://pytorch-geometric.com/whl/torch-2.9.0+cu126.html #MAKE SURE TO INSTALL THIS VERSION SPECIFICALLY
pip install torch-geometric

# Install remaining dependencies
pip install -r code/requirements.txt
```

---

## Folder Structure

```
MiniMol/
├── README.md               # This file
├── data/                   # TDC datasets downloaded at runtime (auto-populated)
├── code/
│   ├── run_benchmark.py    # Main benchmarking script
│   └── requirements.txt    # Dependencies
├── artifacts/              # Saved downstream model files per task (auto-populated)
└── logs/                   # Per-task JSON result logs (auto-populated)
```

---

## How to Run

Make sure you are in the `minimol_env` conda environment, then:

```bash
cd model_benchmarks/MiniMol/code
python run_benchmark.py --model minimol
```

This will:
1. Download all 22 TDC ADMET datasets into `../data/`
2. Generate MiniMol fingerprints for each task
3. Train a Random Forest predictor on top
4. Run each task 3 times with different random seeds
5. Save per-task logs to `../logs/`
6. Save downstream model files to `../artifacts/`

---

## Data

All datasets are sourced from the TDC ADMET Benchmark Group and downloaded automatically at runtime via the `PyTDC` library. No manual data preparation is needed.

All tasks use TDC's default **scaffold split** — molecules are split by chemical scaffold to ensure the model is tested on molecules that are different from the molecules in the training set.

---

## Results

See `logs/_all_results.json` for full results after running.

Example expected output for `hia_hou` (Human Intestinal Absorption):

| Metric | Score | Train Time | GPU Memory |
|--------|-------|------------|------------|
| AUROC  | 0.993 ± 0.002 | ~31 min | ~437 MiB |

---

## How MiniMol Works

MiniMol takes a SMILES string (e.g. `CC(=O)Oc1ccccc1C(=O)O` for aspirin) and processes it as a 2D molecular graph where atoms are nodes and bonds are edges. A Graph Neural Network passes messages between neighboring atoms to build up a representation of the whole molecule, producing a 512-number fingerprint that captures its chemical structure and properties.

This fingerprint is then combined with the training data from an ADMET task, and then passed to a task-specific predictor (Random Forest for classification/regression).

The key advantage is that the expensive pre-training of MiniMol only needs to happen once — the fingerprints it produces transfer well across all ADMET tasks without retraining the core model.