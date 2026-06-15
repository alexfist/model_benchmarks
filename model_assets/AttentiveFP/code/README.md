# AttentiveFP

## Overview

AttentiveFP is a **Graph Attention Network (GAT)** for molecular property prediction from the paper "Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism" (Xiong et al., 2019). It is accessed via the **DeepPurpose** library using the `DGL_AttentiveFP` encoder.

Unlike AttrMasking which uses pre-trained GIN weights, AttentiveFP **trains from scratch** on each task using learned attention over molecular graphs.

---

## How AttentiveFP Works

AttentiveFP processes molecules in two stages:

### Stage 1 — Atom-level Attention
Each atom "looks at" its neighbors and decides how much to weight each one:

```
For each atom:
    attention_weight = softmax(LeakyReLU(W × [atom_features || neighbor_features]))
    new_atom_repr = sum(attention_weight × neighbor_features)
```

This lets the model learn that, for example, a nitrogen next to a carbonyl matters more than a nitrogen next to a plain carbon — context-dependent weighting that fixed fingerprints can't capture.

### Stage 2 — Molecule-level Attention
A "super node" attends over all atom representations to build a single molecular fingerprint:

```
molecule_repr = sum(mol_attention_weight × atom_repr for each atom)
```

This means different tasks can implicitly focus on different parts of the molecule.

### Final Prediction
```
SMILES string
      ↓
Molecular graph (atoms = nodes, bonds = edges)
      ↓
Multi-head atom attention (learns neighbor importance)
      ↓
Molecule-level attention pooling (learns which atoms matter most)
      ↓
MLP predictor head (task-specific)
      ↓
ADMET property prediction
```

---

## Deviations from Original Pipeline

- Original TDC submission uses default DeepPurpose hyperparameters
- **Our implementation adds a tuning grid** of 3 hyperparameter combos per task:
  - `lr=1e-3, epochs=20`
  - `lr=1e-4, epochs=30`
  - `lr=5e-4, epochs=50`
- Removed `binary=` parameter from `generate_config()` — not available in installed DeepPurpose version
- Added `CUDA_VISIBLE_DEVICES=0` to prevent multi-GPU NCCL errors on shared server

---

## Environment Setup

AttentiveFP uses the **same conda environment as AttrMasking** (`attrmasking_env`). No additional setup is needed if that environment is already configured.

```bash
# If starting fresh:
conda create -n attrmasking_env python=3.8 -y
conda activate attrmasking_env

conda install -c conda-forge rdkit -y
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu126/repo.html
pip install dgllife
pip install descriptastorus
pip install DeepPurpose
pip install pydantic
pip install -r code/requirements.txt
```

No pre-trained weights needed — AttentiveFP trains from scratch on each task.

---

## Folder Structure

```
AttentiveFP/
├── README.md               # This file
├── data/                   # TDC datasets (auto-populated at runtime)
├── code/
│   ├── run_benchmark.py    # Main benchmarking script
│   ├── check_install.py    # Installation verification
│   └── requirements.txt    # Dependencies
├── artifacts/              # Saved fine-tuned model per task (auto-populated)
└── logs/                   # Per-task JSON result logs and tuning logs (auto-populated)
```

---

## How to Run

```bash
conda activate attrmasking_env
cd model_assets/AttentiveFP/code

# Verify installation
python check_install.py

# Run full benchmark (run overnight — ~24-36 hours with 3 tuning combos)
nohup python run_benchmark.py > ~/model_benchmarks/attentivefp.log 2>&1 &
disown

# Run a single task for testing
python run_benchmark.py --task hia_hou

# Run with fewer seeds for a quick check
python run_benchmark.py --task hia_hou --runs 1
```

---

## Tuning

3 hyperparameter combinations tried per task per seed:

| Learning Rate | Epochs |
|---|---|
| 1e-3 | 20 |
| 1e-4 | 30 |
| 5e-4 | 50 |

Best combination on validation set used for final test evaluation. All tuning results logged to `logs/<task_name>_tuning.json`.

---

## Expected Runtime

With 3 tuning combos × 5 seeds = 15 fine-tuning runs per task × 22 tasks = 330 total training runs. Expect ~24-36 hours on a GPU server depending on hardware.

---

## Known Issues

- Requires Python 3.8 — same constraint as AttrMasking due to DeepPurpose compatibility
- Slower than AttrMasking since we use 3 tuning combos instead of 2
- GPU memory shows 0.0 MiB in logs — this is a tracking limitation, GPU is actually being used
- Use `CUDA_VISIBLE_DEVICES=0` to avoid NCCL multi-GPU errors on shared servers

---

## References

- [Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism (Xiong et al., 2019)](https://pubmed.ncbi.nlm.nih.gov/31408336/)
- [DeepPurpose GitHub](https://github.com/kexinhuang12345/DeepPurpose)
- [TDC Leaderboard — AttentiveFP](https://github.com/mims-harvard/TDC/tree/main/examples/single_pred/admet)
- [DGL-LifeSci AttentiveFP](https://lifesci.dgl.ai/generated/dgllife.model.model_zoo.attentivefp_predictor.html)