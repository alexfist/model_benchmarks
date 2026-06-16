# BaseBoosting

## ❌ Status: Could Not Run

**BaseBoosting could not be executed on the company server due to network restrictions.**

`olorenchemengine` (OCE) is not available as a prebuilt wheel on PyPI or
mirrored package indexes. Installation requires cloning from GitHub
(`github.com/Oloren-AI/olorenchemengine`), which is blocked by the company
server firewall. All three install methods were attempted and failed:

```
pip install olorenchemengine                          # not on PyPI as wheel
pip install olorenchemengine -i <Tsinghua mirror>     # not on mirror
pip install git+https://github.com/Oloren-AI/...     # GitHub HTTPS blocked
```

This is the same class of network restriction that blocked ZairaChem. The
benchmark code is fully implemented and documented — BaseBoosting can be run
on any machine with unrestricted internet access to GitHub.

---

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
            (downloads from Google Cloud Storage — likely also blocked)
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
on first use.

---

## Network Restrictions (Confirmed Blockers)

**1. OCE installation — BLOCKED.** `olorenchemengine` requires cloning from
GitHub to build. The company server blocks all GitHub HTTPS traffic. Neither
PyPI, the Tsinghua mirror, nor direct GitHub clone worked. This is a hard
blocker — the benchmark cannot run without OCE.

**2. OlorenCheckpoint weights — likely also blocked.** Even if OCE could be
installed, `OlorenCheckpoint("default")` downloads a pre-trained GIN from
`storage.googleapis.com`, which is also likely blocked. The script includes
a graceful 2-learner fallback for this case, but it is untested.

---

## Deviations from Original Pipeline

- Original submission code unavailable (GitHub repo `Oloren-AI/OCE-TDC` down as of June 2026)
- Pipeline inferred from OCE documentation and paper
- Could not run — blocked by company server network restrictions (see above)
- If `OlorenCheckpoint` download fails, script falls back to 2 learners (Morgan + RDKit2D)
- No hyperparameter tuning added — original uses fixed n_estimators=1000 per RF

---

## Environment Setup (for machines with unrestricted internet)

```bash
conda create -n baseboosting_env python=3.8 -y
conda activate baseboosting_env
conda install -c conda-forge rdkit -y

# Install OCE (requires GitHub access)
pip install git+https://github.com/Oloren-AI/olorenchemengine.git

# Install PyTorch Geometric (required for GNN components)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install PyTDC numpy pandas scikit-learn joblib
```

---

## Folder Structure

```
BaseBoosting/
├── README.md
├── data/               # TDC datasets (symlink to shared data dir)
├── code/
│   ├── run_benchmark.py    # Main benchmarking script (implemented, not run)
│   ├── check_install.py    # Installation verification
│   └── requirements.txt    # Dependencies
├── artifacts/          # Empty — could not run
└── logs/               # Empty — could not run
```

---

## How to Run (on unrestricted machine)

```bash
conda activate baseboosting_env
cd model_assets/BaseBoosting
ln -s ../MapLight_GNN/data data
cd code

python check_install.py
python run_benchmark.py --task hia_hou --runs 1  # single task test
nohup python run_benchmark.py > ~/model_benchmarks/baseboosting.log 2>&1 &
disown
```

---

## Expected Runtime (if run)

Each task: 3 RF models × 1000 trees × 5 seeds = 15,000 trees trained.
- Small tasks (500-700 molecules): ~2-5 min per seed
- Large tasks (12,000+ molecules, CYP): ~10-20 min per seed

Total estimate: **6-12 hours** for all 22 tasks.

---

## References

- [Oloren ChemEngine Paper (Huang et al., 2022)](https://chemrxiv.org/engage/chemrxiv/article-details/635da3ed6e0d367d91d92362)
- [OCE Documentation](https://docs.oloren.ai)
- [OCE PyPI](https://pypi.org/project/olorenchemengine/)
- [TDC Leaderboard submission](https://github.com/Oloren-AI/OCE-TDC) (currently down)