# CaliciBoost

## Overview

CaliciBoost reproduces "CaliciBoost: Performance-Driven Evaluation of Molecular
Representations for Caco-2 Permeability Prediction" (Huong Van Le et al., 2025).

**Important:** CaliciBoost only covers **caco2_wang** — it was developed
specifically for Caco-2 permeability and does not benchmark all 22 ADMET tasks.

It holds **rank 1** on the TDC Caco-2 leaderboard (MAE = 0.256 ± 0.006) and is
one of only three models that passed the 2026 critical assessment reproducibility
checks (alongside MapLight and MapLight+GNN).

---

## Pipeline

```
SMILES
    ↓
PaDEL descriptors (1875 features: 1444 2D + 431 3D)
    ↓
Feature selection via permutation importance (top 200 features)
    ↓
XGBoost Regressor (Bayesian-optimized hyperparameters)
    ↓
Prediction on test set
```

---

## Why PaDEL?

PaDEL-Descriptor computes over 1875 molecular descriptors including both 2D
(topological, constitutional, electronic) and 3D (geometric, surface area)
features. The paper found that including 3D descriptors reduced MAE by ~16%
compared to 2D alone. This is richer than RDKit's ~200 descriptors or Morgan
fingerprints, which is why CaliciBoost outperforms most other models on Caco-2.

The key insight from the paper: **PaDEL + feature selection + XGBoost** beats
more complex models including GNNs on this small dataset (906 molecules).

---

## Key Dependency: Java

PaDEL-Descriptor is a Java application. padelpy is just a Python wrapper around
it. You must have Java installed.

---

## Deviations from Original Pipeline

- Original uses AutoGluon (AutoML) for initial model selection across all 8
  feature types, then identifies PaDEL + XGBoost as best
- **Our implementation** goes directly to PaDEL + XGBoost with a light
  hyperparameter grid around the paper's best params — same result, no AutoGluon
- Original uses SHAP + permutation importance for feature selection
- **Our implementation** uses permutation importance only (same signal, simpler)
- Original applies Bayesian optimization for hyperparameter tuning
- **Our implementation** uses a 4-combo grid around the paper's reported best
  params (n_estimators, learning_rate, max_depth)

---

## Environment Setup

```bash
conda create -n caliciboost_env python=3.10 -y
conda activate caliciboost_env

# Java is required for PaDEL
conda install -c conda-forge openjdk -y
java -version  # verify

# Python packages
pip install -r code/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**If PaDEL jar download is blocked (GitHub blocked on server):**

padelpy downloads `PaDEL-Descriptor.jar` from GitHub on first use. If that's
blocked, download it on a machine with internet access and copy it to the server:

```bash
# On a machine with internet:
pip install padelpy
python -c "from padelpy import from_smiles; from_smiles('CC', descriptors=True)"
# This downloads the jar to ~/.padelpy/

# Then copy to your server:
scp ~/.padelpy/PaDEL-Descriptor.jar user@server:~/.padelpy/
```

---

## Folder Structure

```
CaliciBoost/
├── README.md
├── data/               # TDC datasets (symlink to shared data dir)
├── code/
│   ├── run_benchmark.py    # Main benchmarking script
│   ├── check_install.py    # Installation verification
│   └── requirements.txt    # Dependencies
├── artifacts/          # Saved model.pkl + hyperparams.json
└── logs/               # Result JSON + tuning log
```

---

## How to Run

```bash
conda activate caliciboost_env
cd model_assets/CaliciBoost/code

# Symlink data (avoids re-downloading)
cd ..
ln -s ../MapLight_GNN/data data
cd code

# Verify installation
python check_install.py

# Run benchmark (~30-60 min — PaDEL is slow on 906 molecules × 5 seeds)
nohup python run_benchmark.py > ~/model_benchmarks/caliciboost.log 2>&1 &
disown

# Check progress
tail -f ~/model_benchmarks/caliciboost.log
```

---

## Expected Runtime

PaDEL computes 1875 descriptors per molecule using Java — it's slow. Rough
estimates per seed:
- Featurizing ~730 train molecules: ~5-10 minutes
- Featurizing ~182 test molecules: ~1-2 minutes
- XGBoost training + tuning: ~1-2 minutes

Total: ~30-60 minutes × 5 seeds = **3-5 hours**

---

## References

- [CaliciBoost paper (Huong Van Le et al., 2025)](https://arxiv.org/abs/2506.08059)
- [CaliciBoost GitHub](https://github.com/Calici/CaliciBoost)
- [TDC Caco-2 Leaderboard](https://tdcommons.ai/benchmark/admet_group/01caco2/)
- [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/)
- [padelpy](https://github.com/ecrl/padelpy)