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
    from deepmol.loaders import CSVLoader
    from deepmol.models import SklearnModel
    from deepmol.compound_featurization import MorganFingerprint
    from tdc.benchmark_group import admet_group
    import numpy as np
    import pandas as pd
    import joblib
    print(f"  rdkit:    {rdkit.__version__}")
    print(f"  deepmol:  ok")
    print(f"  PyTDC:    ok")


# ── Check 2: Featurizer works ─────────────────────────────────────────────────

def check_featurizer():
    from deepmol.compound_featurization import MorganFingerprint
    from deepmol.loaders import CSVLoader
    import pandas as pd
    import tempfile, os

    # Create a tiny test CSV
    test_data = pd.DataFrame({
        "smiles": [
            "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        ],
        "y": [1, 0]
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        test_data.to_csv(f, index=False)
        tmp_path = f.name

    try:
        loader = CSVLoader(
            dataset_path=tmp_path,
            smiles_field="smiles",
            labels_fields=["y"]
        )
        dataset = loader.create_dataset()
        featurizer = MorganFingerprint(radius=2, size=2048)
        featurizer.featurize(dataset)
        print(f"  Morgan fingerprints generated: shape {dataset.X.shape}")
    finally:
        os.unlink(tmp_path)


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
    from deepmol.loaders import CSVLoader
    from deepmol.compound_featurization import MorganFingerprint
    from deepmol.models import SklearnModel
    from sklearn.ensemble import RandomForestClassifier
    from tdc.benchmark_group import admet_group
    import pandas as pd
    import tempfile, os

    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    test = benchmark["test"]
    train, valid = group.get_train_valid_split(
        benchmark="hia_hou", split_type="default", seed=0
    )

    # Save train data to temp CSV for DeepMol loader
    train_df = pd.concat([train, valid]).reset_index(drop=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        train_df.to_csv(f, index=False)
        train_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        test.to_csv(f, index=False)
        test_path = f.name

    try:
        loader = CSVLoader(dataset_path=train_path, smiles_field="Drug", labels_fields=["Y"])
        train_dataset = loader.create_dataset()

        loader_test = CSVLoader(dataset_path=test_path, smiles_field="Drug", labels_fields=["Y"])
        test_dataset = loader_test.create_dataset()

        featurizer = MorganFingerprint(radius=2, size=2048)
        featurizer.featurize(train_dataset)
        featurizer.featurize(test_dataset)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        model = SklearnModel(model=clf, mode="classification")
        model.fit(train_dataset)

        preds = model.predict(test_dataset)
        result = group.evaluate({"hia_hou": preds}, benchmark="hia_hou")
        metric = list(result["hia_hou"].keys())[0]
        score  = result["hia_hou"][metric]

        print(f"  Task: hia_hou | {metric}: {score:.4f}")
        assert score > 0.75, f"Score too low: {score:.4f}"
    finally:
        os.unlink(train_path)
        os.unlink(test_path)


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