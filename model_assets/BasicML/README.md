# Basic ML

## Overview

Basic ML reproduces "Accountable Prediction of Drug ADMET Properties with
Molecular Descriptors" (Boral et al., 2022). The core claim of the paper is
that **31 hand-crafted 2D molecular descriptors + simple sklearn models can
match deep learning performance** on many ADMET tasks — and on some tasks
(especially excretion) they actually win.

No GPU required. No pre-trained weights. No graph neural networks. Just
RDKit descriptors and sklearn.

---

## Pipeline

```
SMILES
    ↓
31 RDKit 2D descriptors (Chem.Descriptors + Chem.Lipinski)
    ↓
Try 10 algorithms on validation set (with RandomizedSearchCV tuning):
    LogisticRegression / LinearRegression
    KNN
    DecisionTree
    RandomForest
    ExtraTrees
    Bagging
    AdaBoost
    GradientBoosting
    XGBoost
    ↓
Pick best model per task
    ↓
Retrain on train + valid combined
    ↓
Predict on test
```

---

## The 31 Descriptors

From `rdkit.Chem.Descriptors` and `rdkit.Chem.Lipinski`:

| Category | Descriptors |
|---|---|
| Molecular weight | MolWt, HeavyAtomMolWt, ExactMolWt |
| Composition | NumValenceElectrons, NumRadicalElectrons |
| Partial charges | MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge, MinAbsPartialCharge |
| Morgan density | FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3 |
| Lipinski / drug-likeness | NumHDonors, NumHAcceptors, MolLogP, MolMR |
| Topological / shape | TPSA, LabuteASA, BalabanJ, BertzCT, HallKierAlpha, Kappa1, Kappa2, Kappa3 |
| Connectivity | Chi0, Chi0n, Chi1, Chi1n |
| Ring / structural | NumAromaticRings, NumRotatableBonds, RingCount |

These are all **global descriptors** — one number per molecule, not per atom.
They describe the molecule's overall physical and chemical character rather
than its local structural patterns.

---

## Deviations from Original Pipeline

- Original does model selection on a random split first, then re-selects on scaffold split
- **Our implementation** does model selection directly on the TDC scaffold split validation set (consistent with all other benchmarks in this project)
- Original uses default parameters for initial candidate screening, then tunes top-2 models
- **Our implementation** runs `RandomizedSearchCV` (8 iterations, 3-fold CV) for all models simultaneously — equivalent coverage, cleaner code
- Original fixed best model per task based on prior analysis; **we re-select per seed** to be fully reproducible without requiring the paper's pre-selected pkl files

---

## Environment Setup

This is the simplest environment of all models — no GPU, no DGL, no torch.

```bash
conda create -n basicml_env python=3.10 -y
conda activate basicml_env

# RDKit is the only tricky one — use conda-forge
conda install -c conda-forge rdkit -y

# Everything else via pip (use Tsinghua mirror in China)
pip install -r code/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## Folder Structure

```
BasicML/
├── README.md
├── data/           # TDC datasets (auto-populated at runtime)
├── code/
│   ├── run_benchmark.py    # Main benchmarking script
│   ├── check_install.py    # Installation verification
│   └── requirements.txt    # Dependencies
├── artifacts/      # Saved best model.pkl + hyperparams.json per task
└── logs/           # Per-task JSON result logs and tuning logs
```

---

## How to Run

```bash
conda activate basicml_env
cd model_assets/BasicML/code

# Verify installation (fast — no downloads needed)
python check_install.py

# Run full benchmark (~1-2 hours on CPU — much faster than GNN models)
nohup python run_benchmark.py > ~/model_benchmarks/basicml.log 2>&1 &
disown

# Single task test
python run_benchmark.py --task hia_hou
python run_benchmark.py --task hia_hou --runs 1
```

---

## Expected Performance

The paper reports that the 31-descriptor + ML approach does particularly well on:
- **Excretion tasks** (half_life_obach, clearance_hepatocyte_az) — ranks 1st
- **VDss** (vdss_lombardo) — ranks 1st
- **Caco-2** (caco2_wang) — ranks 2nd at time of submission

It does poorly on:
- **Metabolism tasks** (CYP inhibition/substrate) — descriptors don't capture local electronic patterns well enough
- **Classification tasks** generally — fingerprints tend to outperform global descriptors for activity prediction

---

## References

- [Boral et al. 2022 — Accountable Prediction of Drug ADMET Properties with Molecular Descriptors](https://www.biorxiv.org/content/10.1101/2022.06.29.115436v1)
- [GitHub — NilavoBoral/Therapeutics-Data-Commons](https://github.com/NilavoBoral/Therapeutics-Data-Commons)
- [RDKit Descriptors Documentation](https://www.rdkit.org/docs/GettingStartedInPython.html)