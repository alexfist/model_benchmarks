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
    ↓
Gradient boosting: each learner corrects residuals of the previous
    ↓
Prediction (classification or regression, auto-detected)
```

---

## What is BaseBoosting?

Unlike standard ensembles (which average predictions), `BaseBoosting` trains
learners sequentially — each one fits the residuals of the previous. The key
innovation in OCE is that each "weak learner" can use a completely different
molecular representation, so the ensemble captures structural (Morgan
fingerprints), physicochemical (RDKit 2D), and learned (GIN embedding)
information in one model.

## What is OlorenCheckpoint("default")?

Oloren's proprietary pre-trained GIN (Graph Isomorphism Network) fingerprint,
trained using contrastive learning on PubChem. Similar concept to AttrMasking
(pre-trained GNN as a feature extractor) but Oloren's own implementation with
their own pre-training data and strategy. Downloads from Google Cloud Storage
on first use — if this fails on a restricted network, the script automatically
falls back to a 2-learner model (Morgan + RDKit2D only) and logs the deviation.

---

## Deviations from Original Pipeline

- Original submission code (`Oloren-AI/OCE-TDC`) was unavailable at time of
  implementation — pipeline reconstructed from OCE documentation and the
  Oloren ChemEngine paper
- No hyperparameter tuning added — original uses fixed n_estimators=1000 per RF
- If `OlorenCheckpoint` download fails at runtime, falls back to 2 learners
  (Morgan + RDKit2D) — logged per-task in the result JSON

---

## Environment Setup

OCE installation requires GitHub access for some methods, but the package is
also available directly on PyPI, which works even when GitHub is blocked.

```bash
conda create -n baseboosting_env python=3.8 -y
conda activate baseboosting_env
conda install -c conda-forge rdkit -y

# Install OCE from PyPI (works even when GitHub is blocked)
pip install olorenchemengine

# Remaining dependencies
pip install PyTDC scikit-learn descriptastorus joblib
```

If `pip install olorenchemengine` fails to build, try installing build
dependencies first (`pip install --upgrade pip setuptools wheel`) and retry.

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

# Symlink data (avoids re-downloading from blocked dataverse.harvard.edu)
ln -s ../MapLight_GNN/data data

cd code

# Verify installation (OlorenCheckpoint check is informational — falls back if blocked)
python check_install.py

# Single task test first
python run_benchmark.py --task hia_hou --runs 1

# Full benchmark (run in background — several hours)
nohup python run_benchmark.py > ~/model_benchmarks/baseboosting.log 2>&1 &
disown
```

---

## Expected Runtime

Each task: 3 RF models × 1000 trees × 5 seeds = 15,000 trees trained.
- Small tasks (500-700 molecules): ~2-5 min per seed
- Large tasks (12,000+ molecules, CYP): ~10-20 min per seed

Total estimate: **6-12 hours** for all 22 tasks.

---

## References

- [Oloren ChemEngine Paper (Huang et al., 2022)](https://chemrxiv.org/engage/chemrxiv/article-details/635da3ed6e0d367d91d92362)
- [OCE Documentation](https://docs.oloren.ai)
- [OCE PyPI](https://pypi.org/project/olorenchemengine/)
- [TDC Leaderboard submission](https://github.com/Oloren-AI/OCE-TDC)