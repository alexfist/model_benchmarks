# MapLight + GNN

## Overview

MapLight + GNN is a molecular property prediction model developed by MapLight that combines **traditional molecular fingerprints** with **GNN-derived embeddings** fed into a **CatBoost gradient boosted decision tree** model.

It is one of the most important models in this benchmark because a 2026 critical assessment of TDC leaderboard models found that <b>MapLight and MapLight+GNN are among only 3 models that passed all reproducibility checks</b> — meaning their results are genuinely trustworthy, unlike several other models that exhibited data leakage or non-reproducible environments.

---

## How It Works

MapLight + GNN concatenates multiple feature types into one wide feature vector, then feeds that into CatBoost:

```
SMILES string
      ↓
Generate 4 types of fingerprints/descriptors simultaneously:
  - ECFP (Morgan fingerprints) — circular atom environment encoding
  - Avalon fingerprints — substructure-based fingerprints
  - ErG fingerprints — extended reduced graph, captures pharmacophore features
  - RDKit 2D descriptors — 200 physicochemical properties (MW, logP, etc.)
      ↓
Also generate GIN supervised masking embeddings (300-dim)
  via molfeat from the DGL-LifeSci pretrained model zoo
      ↓
Concatenate all features → 2,573-dimensional feature vector
      ↓
CatBoost gradient boosted decision tree
  (5-seed ensemble, averaged predictions)
      ↓
ADMET property prediction
```

Total feature vector: 2,573 dimensions
- ECFP: ~1024 bits
- Avalon: ~512 bits
- ErG: ~315 bits
- RDKit 2D: 200 descriptors
- GIN embeddings: 300 dimensions

---

## What Makes This Approach Different

### Why CatBoost instead of a neural network?

CatBoost is a gradient boosted decision tree — a fundamentally different type of model from GNNs or neural networks. Rather than learning through backpropagation, it builds an ensemble of decision trees sequentially, each one correcting the errors of the previous. For tabular/fingerprint data this often outperforms neural networks because:
- No need to tune learning rates, batch sizes, or epochs
- Less prone to overfitting on small datasets
- Handles mixed feature types naturally
- More robust to irrelevant features

### Why combine so many fingerprints?

Each fingerprint type captures different aspects of molecular structure:
- **ECFP** — local atomic environments (what atoms are near each other)
- **Avalon** — substructure patterns (which chemical groups are present)
- **ErG** — pharmacophore features (shape and interaction points)
- **RDKit descriptors** — global physicochemical properties

No single fingerprint captures everything. By concatenating them all, the model has a much richer view of each molecule. The paper found that ECFP + Avalon + ErG was the best 3-way combination from an exhaustive search.

### What do the GIN embeddings add?

The GIN supervised masking embeddings (from molfeat/DGL-LifeSci) are pre-trained graph neural network features — similar conceptually to the AttrMasking model but used here as a fixed feature extractor (frozen, not fine-tuned) concatenated alongside the traditional fingerprints. This gives the model both hand-crafted chemical knowledge and learned structural representations.

---

## Known Issues

- Not runnable on Google Colab due to molfeat GIN dependency issues
- Requires mamba (not conda) for environment setup
- Python 3.10 required

---

## Environment Setup

```bash
# Use mamba (faster, more reliable than conda for this)
mamba create -n maplight_env python=3.10 -y
mamba activate maplight_env

# Install JupyterLab (used in original submission notebooks)
mamba install jupyterlab -y

# Clone the MapLight repo
git clone https://github.com/maplightrx/MapLight-TDC.git

# Install dependencies
pip install catboost
pip install molfeat[all]
pip install rdkit
pip install PyTDC
pip install numpy pandas scikit-learn joblib

# Install remaining dependencies
pip install -r code/requirements.txt
```

---

## Folder Structure

```
MapLight_GNN/
├── README.md               # This file
├── data/                   # TDC datasets downloaded at runtime (auto-populated)
├── code/
│   ├── run_benchmark.py    # Main benchmarking script
│   ├── check_install.py    # Installation verification script
│   └── requirements.txt    # Dependencies
├── artifacts/              # Saved CatBoost model per task (auto-populated)
└── logs/                   # Per-task JSON result logs and tuning logs (auto-populated)
```

---

## How to Run

```bash
mamba activate maplight_env
cd model_benchmarks/MapLight_GNN/code

# Verify installation first
python check_install.py

# Run full benchmark across all 22 tasks
python run_benchmark.py

# Run a single task
python run_benchmark.py --task hia_hou
```

---

## Data

All datasets from TDC ADMET Benchmark Group, downloaded automatically via `PyTDC`. All tasks use TDC's default **scaffold split**. Supports both classification and regression tasks.

---

## Tuning

CatBoost is notably robust with default hyperparameters. The original MapLight paper used default parameters with only a single modification to `random_strength`. The benchmark script tunes:
- `iterations` (number of trees): [500, 1000, 2000]
- `learning_rate`: [0.03, 0.1]

All tuning results logged to `logs/<task_name>_tuning.json`.

---

## Reproducibility Note

A 2026 critical assessment of TDC leaderboard models found that MapLight and MapLight+GNN passed all reproducibility checks, making them among the most trustworthy models on the leaderboard. This contrasts with MiniMol and several others which showed data leakage issues.

---

## References

- [MapLight TDC GitHub](https://github.com/maplightrx/MapLight-TDC)
- [MapLight Paper (arXiv:2310.00174)](https://arxiv.org/abs/2310.00174)
- [molfeat documentation](https://molfeat.datamol.io)
- [CatBoost documentation](https://catboost.ai)
- [Critical Assessment of TDC Models (2026)](https://www.researchgate.net/publication/401376507)