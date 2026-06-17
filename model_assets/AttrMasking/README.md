# AttrMasking

## Overview

AttrMasking is a **pre-trained Graph Neural Network (GNN)** model based on the paper "Strategies for Pre-training Graph Neural Networks" (Hu et al., 2020). It is accessed via the **DeepPurpose** library using the `DGL_GIN_AttrMasking` encoder.

The model is built on a **GIN (Graph Isomorphism Network)** backbone, pre-trained using a combination of node-level and graph-level strategies before being fine-tuned on each ADMET task.

---

## The GIN Architecture
\
Unlike standard GNNs which aggregate neighbor information by averaging, GIN uses **sum aggregation** with a learned parameter:

```
new_representation = MLP((1 + ε) × own_features + sum(neighbor_features))
```

Sum aggregation is provably more powerful than mean/max at distinguishing graph structures — it's as powerful as the Weisfeiler-Lehman graph isomorphism test. For molecules this matters enormously since structural isomers can behave completely differently biologically.

---

## Pre-training Strategy

### Stage 1 — Node-level (Self-Supervised)

**Attribute Masking**
- Randomly masks atom and bond attributes
- Trains the GNN to predict masked attributes from surrounding neighborhood
- Teaches local chemical patterns and valence rules

**Context Prediction**
- For each node, predicts whether a context graph (K₁ to K₂ hops) surrounds a given inner neighborhood
- Teaches longer-range structural relationships

### Stage 2 — Graph-level (Supervised)
- Pre-trains on labeled data from multiple biological property datasets simultaneously
- Warm-starts fine-tuning so it converges faster on ADMET tasks

---

## Fine-tuning on ADMET Tasks

```
SMILES string
      ↓
Molecular graph (atoms = nodes, bonds = edges)
      ↓
GIN with sum aggregation (pre-trained weights as starting point)
      ↓
Graph-level pooling → fixed-size embedding
      ↓
MLP predictor head (task-specific)
      ↓
Fine-tune all weights end-to-end on task data
      ↓
ADMET property prediction
```

Unlike MiniMol (frozen encoder), AttrMasking **updates all GNN weights** during fine-tuning — slower but more powerful.

---

## Deviations from Original Pipeline

- Original uses fixed hyperparameters from the paper (lr=0.001, epochs=50)
- **Our implementation adds a tuning grid** of 2 hyperparameter combos per task:
  - `lr=1e-3, epochs=20`
  - `lr=1e-4, epochs=30`
- Removed `binary=` parameter from `generate_config()` — not available in installed DeepPurpose version
- Added `CUDA_VISIBLE_DEVICES=0` to prevent multi-GPU NCCL errors on shared server
- Pre-trained weights downloaded manually to `~/.dgl/` due to intermittent network issues
- Monkey-patched `dgl.data.utils.download` with `overwrite=False` to prevent re-downloading weights on every run

---

## Environment Setup

```bash
conda create -n attrmasking_env python=3.8 -y
conda activate attrmasking_env

# Install RDKit
conda install -c conda-forge rdkit -y

# Install DGL (CPU version for Linux without CUDA)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cpu/repo.html

# Or with CUDA (for GPU server):
conda install -c dglteam/label/th24_cu121 dgl -y

# Install dgllife
pip install dgllife

# Install DeepPurpose
pip install git+https://github.com/bp-kelley/descriptastorus
pip install DeepPurpose

# Install remaining dependencies
pip install -r code/requirements.txt
```

### Pre-trained weights

Download manually on first setup:

```bash
mkdir -p ~/.dgl
wget https://data.dgl.ai/dgllife/pre_trained/gin_supervised_masking.pth -O ~/.dgl/gin_supervised_masking_pre_trained.pth
```

---

## Folder Structure

```
AttrMasking/
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
cd model_assets/AttrMasking/code

# Verify installation
python check_install.py

# Run full benchmark (run overnight — ~18-24 hours)
nohup python run_benchmark.py > ~/model_benchmarks/attrmasking.log 2>&1 &
disown

# Run a single task for testing
python run_benchmark.py --task hia_hou
```

---

## Tuning

2 hyperparameter combinations tried per task per seed:

| Learning Rate | Epochs |
|---|---|
| 1e-3 | 20 |
| 1e-4 | 30 |

Best combination on validation set used for final test evaluation. All tuning results logged to `logs/<task_name>_tuning.json`.

---

## Known Issues

- Requires Python 3.8 — older than other models due to DeepPurpose compatibility
- Very slow — 2 combos × 5 seeds = 10 fine-tuning runs per task, each training a full GNN
- GPU memory shows 0.0 MiB in logs — this is a tracking limitation, GPU is actually being used
- Pre-trained weights must be downloaded manually on first run
- Use `CUDA_VISIBLE_DEVICES=0` to avoid NCCL multi-GPU errors on shared servers

---

## References

- [Strategies for Pre-training Graph Neural Networks (Hu et al., 2020)](https://arxiv.org/abs/1905.12265)
- [DeepPurpose GitHub](https://github.com/kexinhuang12345/DeepPurpose)
- [DGL-LifeSci](https://github.com/awslabs/dgl-lifesci)