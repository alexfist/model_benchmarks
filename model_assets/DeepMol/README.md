# DeepMol (AutoML)

## Overview

DeepMol is a Python-based machine and deep learning framework for drug discovery. It uses TensorFlow, Keras, Scikit-learn, and DeepChem to build custom ML and DL models. It uses the RDKit framework to perform operations on molecular data.

What makes DeepMol different from MiniMol is its **AutoML** approach. Rather than using a fixed pipeline architecture, DeepMol automatically searches through combinations of:
- **Featurizers** — different ways to represent a molecule (Morgan fingerprints, RDKit descriptors, Mol2Vec embeddings, graph features)
- **Models** — Random Forest, SVM, XGBoost, neural networks, graph neural networks
- **Hyperparameters** — automatically tuned per task

It finds the best pipeline for each specific ADMET task rather than using one fixed approach for all tasks.

**Paper:** [DeepMol: An Automated Machine and Deep Learning Framework for Computational Chemistry](https://www.biorxiv.org/content/10.1101/2024.05.27.595849)
**GitHub:** https://github.com/BioSystemsUM/DeepMol
**Docs:** https://deepmol.readthedocs.io

---

## Known Issues

- Loading TensorFlow models is problematic on MacOS — run on Linux only
- Do not install JAX alongside DeepMol — it causes dependency conflicts
- Requires Python 3.10 (not 3.11) for TensorFlow compatibility

---

## Environment Setup

```bash
conda create -n deepmol_env python=3.10
conda activate deepmol_env

# Install RDKit first (required by DeepMol)
conda install -c conda-forge rdkit

# Install DeepMol modules
pip install "deepmol[preprocessing]"
pip install "deepmol[machine-learning]"
pip install "deepmol[deep-learning]"

# Install Mol2Vec (additional featurizer)
pip install git+https://github.com/samoturk/mol2vec#egg=mol2vec

# Install remaining dependencies
pip install -r code/requirements.txt
```

---

## Folder Structure

```
DeepMol/
├── README.md               # This file
├── data/                   # TDC datasets downloaded at runtime (auto-populated)
├── code/
│   ├── run_benchmark.py    # Main benchmarking script
│   ├── check_install.py    # Installation verification script
│   └── requirements.txt    # Dependencies
├── artifacts/              # Saved best pipeline per task (auto-populated)
└── logs/                   # Per-task JSON result logs and tuning logs (auto-populated)
```

---

## How to Run

Make sure you are in the `deepmol_env` conda environment, then:

```bash
cd model_benchmarks/DeepMol/code

# Verify installation first
python check_install.py

# Run full benchmark across all 22 tasks
python run_benchmark.py

# Run a single task
python run_benchmark.py --task hia_hou
```

---

## Data

All datasets are sourced from the TDC ADMET Benchmark Group and downloaded automatically at runtime via the `PyTDC` library. All tasks use TDC's default **scaffold split**.

---

## How DeepMol Works

Unlike MiniMol which uses a fixed two-stage pipeline, DeepMol runs an **automated search** over multiple pipelines per task:

```
SMILES strings
      ↓
Try multiple featurizers:
  - Morgan Fingerprints (ECFP)
  - RDKit Descriptors
  - Mol2Vec embeddings
      ↓
Try multiple models:
  - Random Forest
  - SVM
  - XGBoost
  - Neural Network
      ↓
Pick best combination on validation set
      ↓
Final prediction on test set
```

This means DeepMol may use a completely different pipeline for `hia_hou` vs `cyp2d6_veith` — whichever works best for each task's specific data distribution.

---

## Results

See `logs/_all_results.json` for full results after running. Each task also has a `logs/<task_name>_tuning.json` showing which pipeline combinations were tried and their validation scores.