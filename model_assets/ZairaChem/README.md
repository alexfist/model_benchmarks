# ZairaChem

## Overview

ZairaChem is an **automated ML-based (Q)SAR pipeline** developed by the Ersilia Open Source Initiative. Unlike the other models in this benchmark which are Python libraries, ZairaChem operates as a **command-line tool** with its own installation script that sets up multiple conda environments internally — one per processing step.

It is the most "hands-off" model in this benchmark. You give it a CSV file of SMILES and activity labels, and it automatically handles featurization, model selection, hyperparameter tuning, and prediction entirely on its own.

### How It Works

ZairaChem runs a full automated pipeline:

```
Input CSV (SMILES + activity labels)
      ↓
Molecule standardization & validation
      ↓
Generate multiple descriptor types simultaneously:
  - Morgan/ECFP fingerprints
  - RDKit physicochemical descriptors
  - Shape descriptors (USR, USRCAT)
  - Signature descriptors
  - Grover embeddings (pre-trained transformer on molecules)
      ↓
Train multiple models on each descriptor set:
  - XGBoost
  - Random Forest
  - LightGBM
  - Neural networks
      ↓
Ensemble the best models
      ↓
Output predictions + confidence scores
```

The key difference from DeepMol (which also tries multiple combinations) is that ZairaChem uses **ensembling** — it combines predictions from multiple good models rather than just picking one winner.

### Important Limitation

ZairaChem v1 was benchmarked only on classification tasks on TDC. It does not natively support regression tasks. For the regression tasks in the ADMET benchmark (e.g. lipophilicity, clearance, half-life), ZairaChem results will either be skipped or require a workaround via binarization.

**Paper:** [ZairaChem: Automated ML-based QSAR](https://github.com/ersilia-os/zaira-chem)
**GitHub:** https://github.com/ersilia-os/zaira-chem
**TDC Benchmark Repo:** https://github.com/ersilia-os/zaira-chem-tdc-benchmark

---

## Known Issues

- Linux only — the install script uses bash and is not compatible with Mac or Windows
- Classification tasks only for TDC benchmark (regression tasks not supported in v1)
- Creates multiple conda environments internally — requires significant disk space (~5GB)
- Slower than other models due to the breadth of the automated search
- Requires Python 3.7

---

## Environment Setup

ZairaChem manages its own environments via an install script:

```bash
# Clone the repository
git clone https://github.com/ersilia-os/zaira-chem.git
cd zaira-chem

# Run the install script (Linux only)
# This creates a conda environment named 'zairachem'
bash install_linux.sh

# Activate the environment
conda activate zairachem

# Install TDC
pip install PyTDC pandas numpy
```

---

## Folder Structure

```
ZairaChem/
├── README.md               # This file
├── data/                   # TDC datasets saved as CSV (auto-populated)
├── code/
│   ├── run_benchmark.py    # Main benchmarking script
│   ├── check_install.py    # Installation verification script
│   └── requirements.txt    # Additional dependencies
├── artifacts/              # Saved ZairaChem model folders per task (auto-populated)
└── logs/                   # Per-task JSON result logs (auto-populated)
```

---

## How to Run

```bash
conda activate zairachem
cd model_benchmarks/ZairaChem/code

# Verify installation first
python check_install.py

# Run benchmark on classification tasks only
python run_benchmark.py

# Run a single task
python run_benchmark.py --task hia_hou
```

---

## Data

Datasets from TDC ADMET Benchmark Group. Unlike other models, ZairaChem requires data as **CSV files on disk** rather than in-memory dataframes. The script handles this conversion automatically, saving CSVs to `../data/`.

---

## How ZairaChem Differs from DeepMol

Both are AutoML approaches but with different philosophies:

| | DeepMol | ZairaChem |
|---|---|---|
| Interface | Python library | Command-line tool |
| Selection strategy | Pick best single pipeline | Ensemble multiple models |
| Descriptor types | 3 | 5+ |
| Task support | Classification + Regression | Classification only (v1) |
| Transparency | You control what to try | Fully automated black box |
| Speed | Medium | Slow (thorough search) |

---

## Known TDC Results

ZairaChem's published scores on classification tasks:

| Task | Metric | Score |
|------|--------|-------|
| HIA_Hou | AUROC | 0.957 ± 0.014 |
| Pgp_Broccatelli | AUROC | 0.944 ± 0.002 |
| BBB_Martins | AUROC | 0.930 ± 0.003 |
| AMES | AUROC | 0.863 ± 0.003 |
| DILI | AUROC | 0.936 ± 0.008 |
| hERG | AUROC | 0.861 ± 0.012 |

---

## Results

See `logs/_all_results.json` for full results after running. Tasks skipped due to regression are logged with a note.