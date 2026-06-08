"""
check_install.py
----------------
Verifies DeepMol is installed correctly before running the full benchmark.

Usage:
    python check_install.py
"""

import sys

PASS = "  ✓ PASS"
FAIL = "  ✗ FAIL"

def check(label, fn):
    print(f"\n[{label}]")
    try:
        fn()
        print(PASS)
        return True
    except Exception as e:
        print(f"{FAIL}: {e}")
        return False


# ── Check 1: Imports ───────────────────────────────────────────────────────────

def check_imports():
    import rdkit
    import deepmol
    from deepmol.compound_featurization import MorganFingerprint
    from deepmol.datasets.datasets import SmilesDataset
    from deepmol.models.sklearn_models import SklearnModel
    from tdc.benchmark_group import admet_group
    import numpy as np
    import pandas as pd
    import joblib
    import xgboost
    print(f"  rdkit:    {rdkit.__version__}")
    print(f"  deepmol:  ok")
    print(f"  xgboost: {xgboost.__version__}")
    print(f"  PyTDC:    ok")

# ── Check 2: Featurizer works ─────────────────────────────────────────────────

def check_featurizer():
    from deepmol.compound_featurization import MorganFingerprint
    from deepmol.datasets.datasets import SmilesDataset
    import numpy as np

    # Create a tiny test CSV
    smiles =[
            "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        ]
    dataset = SmilesDataset(smiles = smiles)
    fp = MorganFingerprint(radius=2, size=2048)
    result = fp.featurize(dataset)
    assert result.X is not None, print(f"Morgan fingerprints shape: {result.X.shape}")
    assert result.X.shape == (2,2048), f"Expected shape (2, 2048), got {result.X.shape}"


# ── Check 3: TDC dataset loading ──────────────────────────────────────────────

def check_tdc():
    from tdc.benchmark_group import admet_group
    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    train_val = benchmark["train_val"]
    test      = benchmark["test"]
    print(f"  hia_hou loaded: train_val={len(train_val)}, test={len(test)}")


# ── Check 4: End-to-end on hia_hou ────────────────────────────────────────────

def check_end_to_end():
    from deepmol.compound_featurization import MorganFingerprint
    from deepmol.models import SklearnModel
    from sklearn.ensemble import RandomForestClassifier
    from tdc.benchmark_group import admet_group
    from deepmol.datasets.datasets import SmilesDataset
    import pandas as pd

    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    test = benchmark["test"]
    train, valid = group.get_train_valid_split(
        benchmark="hia_hou", split_type="default", seed=0
    )
    
    train_df = pd.concat([train, valid]).reset_index(drop=True)
    #create datasets from smile lists directly
    train_dataset = SmilesDataset(smiles = train_df['Drug'].tolist(), y = train_df['Y'].values)
    test_dataset  = SmilesDataset(smiles = test['Drug'].tolist(), y = test['Y'].values)

    #featurize
    fp = MorganFingerprint(radius=2, size=2048)
    train_result = fp.featurize(train_dataset)
    test_result  = fp.featurize(test_dataset)

    #train model
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    model = SklearnModel(model = clf, task = "classification")
    model.fit(train_result)

    #predict
    preds = model.predict(test_result)
    result = group.evaluate({'hia_hou': preds}, benchmark = 'hia_hou')
    metric = list(result['hia_hou'].keys())[0]
    score = result['hia_hou'][metric]
    print(f"  End-to-end on hia_hou: {metric} = {score:.4f}")
    assert score > 0.60, f"Expected {metric} > 0.75, got {score:.4f}"



# ── Run all checks ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  DeepMol Installation Check")
    print("=" * 50)

    results = [
        check("1. Imports",              check_imports),
        check("2. Featurizer",           check_featurizer),
        check("3. TDC dataset loading",  check_tdc),
        check("4. End-to-end on hia_hou", check_end_to_end),
    ]

    print("\n" + "=" * 50)
    passed = sum(results)
    total  = len(results)

    if passed == total:
        print(f"  All {total} checks passed. Ready to run benchmark!")
    else:
        print(f"  {passed}/{total} checks passed. Fix failing checks before running benchmark.")

    print("=" * 50)
    sys.exit(0 if passed == total else 1)