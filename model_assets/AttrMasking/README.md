# AttrMasking

## Overview

AttrMasking is a **pre-trained Graph Neural Network (GNN)** model based on the paper "Strategies for Pre-training Graph Neural Networks" (Hu et al., 2020). It is accessed via the **DeepPurpose** library using the `DGL_GIN_AttrMasking` encoder.

The model is built on a **GIN (Graph Isomorphism Network)** backbone, pre-trained using a combination of node-level and graph-level strategies before being fine-tuned on each ADMET task.

---

## The GIN Architecture

Unlike standard GNNs which aggregate neighbor information by averaging, GIN uses **sum aggregation** with a learned parameter:

```
new_representation = MLP((1 + ε) × own_features + sum(neighbor_features))
```

This is important because averaging can collapse different graph structures into the same representation. For example, a node with 2 neighbors each of value 1 looks identical to a node with 1 neighbor of value 2 under mean aggregation — but sum aggregation distinguishes them.

For molecules this matters enormously — two molecules can have the same atoms but different connectivity and behave completely differently biologically. GIN is designed to catch these differences.

---

## Pre-training Strategy

The model is pre-trained in two stages before being applied to the ADMET tasks

### Stage 1 — Node-level (Self-Supervised)

Two complementary strategies teach the model local and neighborhood-level chemistry:

**1. Attribute Masking**
- Randomly masks atom and bond attributes (e.g. atom type, chirality, bond type)
- Trains the GNN to predict the masked attributes from the surrounding neighborhood
- Teaches the model: *"given the chemical context around an atom, what is that atom likely to be?"*
- Captures local chemical patterns and valence rules

**2. Context Prediction**
- For each node, defines two subgraphs: an inner neighborhood (K₁ hops) and an outer context graph (K₁ to K₂ hops)
- Trains the model to predict whether a context graph actually surrounds a given inner neighborhood
- Teaches the model: *"what does the broader molecular environment around a substructure look like?"*
- Captures structural relationships between subparts of a molecule

### Stage 2 — Graph-level (Supervised)
- Pre-trains the full model on labeled data from multiple biological property datasets simultaneously
- Teaches the model global molecule-level properties before specializing on any single task
- Acts as a warm-start so fine-tuning on ADMET tasks converges faster and more reliably

---

## Fine-tuning on ADMET Tasks

After pre-training, the model is fine-tuned on each TDC ADMET task individually:

```
SMILES string
      ↓
Convert to molecular graph (atoms = nodes, bonds = edges)
      ↓
GIN processes graph with sum aggregation
  (pre-trained weights as starting point)
      ↓
Graph-level pooling → fixed-size embedding
      ↓
MLP predictor head (task-specific)
      ↓
Fine-tune all weights end-to-end on task data
      ↓
ADMET property prediction
```

Unlike MiniMol where the encoder is frozen and only a downstream predictor is trained, AttrMasking **updates the GNN weights themselves** during fine-tuning. This makes it slower but potentially more powerful — the model can adapt its internal representations to the specific chemistry relevant to each task.

---

## Known Issues

- Requires Python 3.6–3.8 for full DeepPurpose compatibility (older than other models)
- Requires DGL (Deep Graph Library) and dgllife to be installed separately
- Pre-trained weights are downloaded automatically on first run (~200MB)
- Fine-tuning is slower than MiniMol since GNN weights are updated during training

---

## Environment Setup

```bash
conda create -n attrmasking_env python=3.8
conda activate attrmasking_env

# Install RDKit
conda install -c conda-forge rdkit

# Install DGL (CPU version for Linux)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html

# Install dgllife
pip install dgllife

# Install DeepPurpose dependencies
pip install git+https://github.com/bp-kelley/descriptastorus
pip install DeepPurpose

# Install remaining dependencies
pip install -r code/requirements.txt
```

---

## Folder Structure

```
AttrMasking/
├── README.md               # This file
├── data/                   # TDC datasets downloaded at runtime (auto-populated)
├── code/
│   ├── run_benchmark.py    # Main benchmarking script
│   ├── check_install.py    # Installation verification script
│   └── requirements.txt    # Dependencies
├── artifacts/              # Saved fine-tuned model per task (auto-populated)
└── logs/                   # Per-task JSON result logs and tuning logs (auto-populated)
```

---

## How to Run

```bash
cd model_benchmarks/AttrMasking/code

# Verify installation first
python check_install.py

# Run full benchmark across all 22 tasks
python run_benchmark.py

# Run a single task
python run_benchmark.py --task hia_hou
```

---

## Data

All datasets from TDC ADMET Benchmark Group, downloaded automatically via `PyTDC`. All tasks use TDC's default **scaffold split**.

---

## Tuning

AttrMasking tunes the GNN fine-tuning process across 4 combinations:

| Learning Rate | Epochs |
|---|---|
| 1e-3 | 30 |
| 5e-4 | 30 |
| 1e-4 | 50 |
| 1e-3 | 50 |

The best combination on the validation set is used for final test evaluation. All tuning results are logged to `logs/<task_name>_tuning.json`.

---

## Results

See `logs/_all_results.json` for full results after running.

---

## References

- [Strategies for Pre-training Graph Neural Networks (Hu et al., 2020)](https://arxiv.org/abs/1905.12265)
- [DeepPurpose GitHub](https://github.com/kexinhuang12345/DeepPurpose)
- [DGL-LifeSci](https://github.com/awslabs/dgl-lifesci)
- [TDC AttrMasking Tutorial](https://github.com/mims-harvard/TDC/blob/main/tutorials/DGL_User_Group_Demo.ipynb)