# Results

This folder contains the benchmark results for all 5 TDC ADMET models.

## Output Files

| File | Description |
|------|-------------|
| `summary_by_model.csv` | Per-model scores (mean ± std) across all 22 tasks |
| `top3_by_task.csv` | Top 3 models per task with training time and GPU memory |
| `hyperparams_summary.csv` | Best hyperparameters and all combos tried per model per task |

To regenerate these files after adding new model logs:
```bash
python results/generate_summary.py
```

---

## Model Status

| Model | Status | Notes |
|-------|--------|-------|
| MiniMol | Done | All 22 tasks completed |
| DeepMol | Done | All 22 tasks completed |
| MapLight+GNN | Done | All 22 tasks completed |
| AttrMasking | Done | All 22 tasks completed |
| ZairaChem | Failed | See ZairaChem section below |

---

## Deviations from Original Model Pipelines

### MiniMol

**Original pipeline:**
- Pre-trained GNN encoder (frozen) → MLP downstream predictor
- Fixed hyperparameters, no per-task tuning
- Pre-trained on ~6M molecules across 3,300 biological and quantum tasks

**Our implementation:**
- Pre-trained GNN encoder (frozen) → Random Forest downstream predictor
- Added grid search over 4 hyperparameter combos per task (n_estimators: 50/100/200, max_depth: None/10)
- Used seeds [1,2,3,4,5] and evaluate_many() per TDC protocol

**Known issues:**
- MiniMol pre-training data may overlap with TDC test molecules (data leakage flagged in 2026 critical assessment)
- Scores may be slightly inflated as a result
- torch 2.9.0 required — graphium dependency pulls in 2.12.0 and must be reinstalled after

---

### DeepMol

**Original pipeline:**
- Full AutoML search across 140+ model/featurizer combinations
- Uses deepmol 1.2.1 with RDKit descriptors, mol2vec, and deep learning featurizers

**Our implementation:**
- Reduced to 2 featurizers x 3 models = 6 combos due to dependency conflicts
- Featurizers: Morgan fingerprints (2048-bit, 1024-bit) only
- Models: Random Forest (100 trees), Random Forest (200 trees), XGBoost
- Used deepmol 1.1.7 instead of 1.2.1 due to rdkit/PyTDC version conflicts
- PyTDC installed with --no-deps to avoid rdkit conflict
- API changes from 1.2.1: SmilesDataset path, featurize() return value, task= parameter

**Known issues:**
- mol2vec not installed — breaks pandas version pins
- deepmol 1.1.7 has fewer featurizers than 1.2.1

---

### AttrMasking

**Original pipeline:**
- DGL_GIN_AttrMasking encoder via DeepPurpose
- Fixed hyperparameters from paper (lr=0.001, epochs=50)

**Our implementation:**
- Same DGL_GIN_AttrMasking encoder via DeepPurpose
- Added tuning grid over 4 hyperparameter combos (lr: 1e-3/5e-4/1e-4, epochs: 30/50)
- Removed binary= parameter from generate_config() — not available in installed version
- Added CUDA_VISIBLE_DEVICES=0 to prevent multi-GPU NCCL errors
- Pre-trained weights downloaded manually to ~/.dgl/ due to network restrictions
- Monkey-patched dgl.data.utils.download with overwrite=False to prevent re-downloading weights

**Known issues:**
- Very slow — 440 total fine-tuning runs (22 tasks x 4 combos x 5 seeds)
- GPU memory shows 0.0 MiB due to how DeepPurpose tracks memory

---

### ZairaChem

**Original pipeline:**
- Full AutoML pipeline using Ersilia Model Hub descriptors
- Multiple descriptor types including learned representations from Ersilia Hub models
- Multiple AutoML frameworks: FLAML, AutoGluon, Keras Tuner, TabPFN, MolMapNet
- Ensembles predictions from multiple models

**Our implementation:**
- Used zairachem-docker (v1.0.0) instead of original zaira-chem
- Several manual bug fixes applied to the source code:
  - manifolds.py: Added None check before _contribute() call
  - manifolds.py: Added None check in algo_data loop
  - pool.py: Initialized results = pd.DataFrame() when features empty
  - table.py: Added early return when tasks list is empty

**Why ZairaChem failed:**
ZairaChem relies on downloading molecular descriptors from the **Ersilia Model Hub** at runtime. The Ersilia Hub hosts 200+ pre-trained AI models accessed via GitHub. Our company server blocks external GitHub connections, causing descriptor downloads to fail silently. When descriptors return None/empty, ZairaChem's pipeline crashes with cascading errors across multiple internal modules.

This is a **network infrastructure issue**, not a code or model issue. ZairaChem would work correctly in an environment with unrestricted internet access.

**Cascading errors encountered:**
1. `manifolds.py`: TypeError — NoneType object is not iterable (descriptor data is None)
2. `pool.py`: UnboundLocalError — results variable unbound when features empty
3. `table.py`: IndexError — list index out of range when tasks list empty

**Resolution:**
ZairaChem results are not included in the benchmark summary. The model is documented here for completeness. To reproduce ZairaChem results, a server with unrestricted access to github.com and the Ersilia Model Hub is required.

---

### MapLight + GNN

**Original pipeline:**
- ECFP + Avalon + ErG + RDKit 2D descriptors + GIN supervised masking embeddings → CatBoost
- Default CatBoost parameters with minimal tuning
- 5-seed ensemble

**Our implementation:**
- Same feature pipeline with molfeat API change: 'ecfp' instead of 'ecfp:4'
- Added tuning grid over 4 CatBoost hyperparameter combos (iterations: 500/1000/2000, learning_rate: 0.03/0.1)
- Required LD_LIBRARY_PATH to be set for DGL CUDA library access
- DGL installed via conda install -c dglteam/label/th212_cu126 to match torch 2.12.0
- numpy pinned to 1.26.4 to avoid scipy/pandas compatibility issues

**Known issues:**
- molfeat ecfp:4 format not supported — use ecfp instead
- DGL requires explicit LD_LIBRARY_PATH pointing to CUDA runtime libraries

---

## Reproducibility Notes

| Model | Reproducibility | Notes |
|-------|----------------|-------|
| MiniMol | Partial | Data leakage flagged — scores may be inflated |
| DeepMol | Good | Reduced featurizer set vs original |
| AttrMasking | Good | Added tuning grid vs original fixed params |
| ZairaChem | N/A | Could not run — Ersilia Hub network access required |
| MapLight+GNN | Good | Passed all reproducibility checks in 2026 assessment |

MapLight+GNN is the most trustworthy baseline — deterministic fingerprints, no pre-training data leakage, and verified reproducible.