# BaseBoosting

## Overview

BaseBoosting reproduces the Oloren AI submission to the TDC ADMET leaderboard
from "A Unified System for Molecular Property Predictions: Oloren ChemEngine
and its Applications" (Huang et al., 2022).

It uses the **olorenchemengine (OCE)** library's `BaseBoosting` class — a
gradient-boosted ensemble of Random Forest models, each using a different
molecular representation.

---

## Pipeline

```
SMILES
    ↓
BaseBoosting ensemble of 3 Random Forest learners:
    ├── RF on Morgan counts fingerprints  (morgan3counts, ~2048-dim)
    ├── RF on normalized RDKit 2D descriptors (rdkit2dnormalized, ~200-dim)
    └── RF on OlorenCheckpoint("default")  ← pre-trained GIN fingerprint
            (downloads from Google Cloud Storage — may be blocked)
    ↓
Gradient boosting: each learner corrects errors of the previous
    ↓
Prediction (classification or regression, auto-detected)
```

---

## What is BaseBoosting?

Unlike standard ensembles (which average predictions), `BaseBoosting` trains
learners sequentially — each one fits the residuals of the previous. The key
innovation in OCE is that each "weak learner" can use a completely different
molecular representation, so the ensemble captures both structural (Morgan
fingerprints) and physicochemical (RDKit 2D) and learned (GIN embedding)
information in one model.

## What is OlorenCheckpoint("default")?

Oloren's proprietary pre-trained GIN (Graph Isomorphism Network) fingerprint,
trained using contrastive learning on PubChem. Similar concept to AttrMasking
(pre-trained GNN as a feature extractor) but Oloren's own implementation with
their own pre-training data and strategy. Downloads from Google Cloud Storage
on first use.

---

## Network Restrictions

Two potential blockers on restricted servers:

**1. OCE installation** — the recommended install is a shell script that pulls
from GitHub. If GitHub is blocked, use pip directly:
```bash
pip install olorenchemengine
```

**2. OlorenCheckpoint weights** — downloaded from `storage.googleapis.com` at
runtime. The script automatically falls back to 2 learners if this fails, and
logs the deviation in the result JSON.

---

## Deviations from Original Pipeline

- Original submission code unavailable (GitHub repo down as of June 2026)
- Pipeline inferred from OCE documentation and paper
- If `OlorenCheckpoint` download fails, we run with 2 learners (Morgan + RDKit2D)
  instead of 3 — this is documented in each task's result JSON
- No hyperparameter tuning added — original uses fixed n_estimators=1000 per RF

---

## Environment Setup

```bash
conda create -n baseboosting_env python=3.8 -y
conda activate baseboosting_env
conda install -c conda-forge rdkit -y

# Install OCE
pip install olorenchemengine -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install PyTorch Geometric (required for GNN components)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install -r code/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## Folder Structure

```
BaseBoosting/
├── README.md
├── data/               # TDC datasets (symlink to shared data dir)
├── code/
│   ├── run_benchmark.py    # Main benchmarking script
│   ├── check_install.py    # Installation verification
│   └── requirements.txt    # Dependencies
├── artifacts/          # Saved model.oce + hyperparams.json per task
└── logs/               # Per-task JSON result logs
```

---

## How to Run

```bash
conda activate baseboosting_env
cd model_assets/BaseBoosting

# Symlink data
ln -s ../MapLight_GNN/data data

cd code

# Check install (OlorenCheckpoint check is optional/informational)
python check_install.py

# Single task test first
python run_benchmark.py --task hia_hou --runs 1

# Full benchmark (~6-12 hours — RF with 1000 trees × 3 learners × 5 seeds × 22 tasks)
nohup python run_benchmark.py > ~/model_benchmarks/baseboosting.log 2>&1 &
disown
```

---

## Expected Runtime

Each task: 3 RF models × 1000 trees × 5 seeds = 15,000 trees trained. On a
large CPU server with parallelism (n_jobs=-1 inside sklearn RF), expect:
- Small tasks (500-700 molecules): ~2-5 min per seed
- Large tasks (12,000+ molecules, CYP): ~10-20 min per seed

Total estimate: **6-12 hours** for all 22 tasks.

---

## References

- [Oloren ChemEngine Paper (Huang et al., 2022)](https://chemrxiv.org/engage/chemrxiv/article-details/635da3ed6e0d367d91d92362)
- [OCE Documentation](https://docs.oloren.ai)
- [OCE PyPI](https://pypi.org/project/olorenchemengine/)
- [TDC Leaderboard submission](https://github.com/Oloren-AI/OCE-TDC) (currently down)