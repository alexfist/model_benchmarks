"""
check_install.py
----------------
Verifies MapLight + GNN is installed correctly before running the full benchmark.

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
    import catboost
    import molfeat
    import rdkit
    from tdc.benchmark_group import admet_group
    import numpy as np
    import pandas as pd
    import joblib
    print(f"  catboost: {catboost.__version__}")
    print(f"  molfeat:  {molfeat.__version__}")
    print(f"  rdkit:    ok")
    print(f"  PyTDC:    ok")


# ── Check 2: Fingerprints generate correctly ───────────────────────────────────

def check_fingerprints():
    from molfeat.trans import MoleculeTransformer
    import numpy as np

    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",         # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    ]

    # ECFP
    ecfp = MoleculeTransformer("ecfp:4")
    ecfp_fps = ecfp(test_smiles)
    print(f"  ECFP fingerprints shape:   {np.array(ecfp_fps).shape}")

    # Avalon
    avalon = MoleculeTransformer("avalon")
    avalon_fps = avalon(test_smiles)
    print(f"  Avalon fingerprints shape: {np.array(avalon_fps).shape}")

    # RDKit descriptors
    rdkit_desc = MoleculeTransformer("desc2D")
    rdkit_fps = rdkit_desc(test_smiles)
    print(f"  RDKit descriptors shape:   {np.array(rdkit_fps).shape}")


# ── Check 3: GIN embeddings load ──────────────────────────────────────────────

def check_gin_embeddings():
    from molfeat.trans.pretrained import PretrainedDGLTransformer
    import numpy as np

    test_smiles = ["CC(=O)Oc1ccccc1C(=O)O"]  # Aspirin

    print("  Loading GIN supervised masking embeddings (may download on first run)...")
    gin = PretrainedDGLTransformer(kind="gin_supervised_masking", dtype=float)
    embeddings = gin(test_smiles)
    print(f"  GIN embeddings shape: {np.array(embeddings).shape}  (expected: (1, 300))")


# ── Check 4: TDC dataset loading ──────────────────────────────────────────────

def check_tdc():
    from tdc.benchmark_group import admet_group
    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    train_val = benchmark["train_val"]
    test      = benchmark["test"]
    print(f"  hia_hou loaded: train_val={len(train_val)}, test={len(test)}")


# ── Check 5: End-to-end on hia_hou ────────────────────────────────────────────

def check_end_to_end():
    from molfeat.trans import MoleculeTransformer
    from molfeat.trans.pretrained import PretrainedDGLTransformer
    from catboost import CatBoostClassifier
    from tdc.benchmark_group import admet_group
    import numpy as np
    import pandas as pd

    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    test = benchmark["test"]
    train, valid = group.get_train_valid_split(
        benchmark="hia_hou", split_type="default", seed=0
    )

    train_combined = pd.concat([train, valid]).reset_index(drop=True)
    train_smiles = train_combined["Drug"].tolist()
    test_smiles  = test["Drug"].tolist()

    # Generate features
    ecfp    = MoleculeTransformer("ecfp:4")
    avalon  = MoleculeTransformer("avalon")
    desc2d  = MoleculeTransformer("desc2D")
    gin     = PretrainedDGLTransformer(kind="gin_supervised_masking", dtype=float)

    def featurize(smiles):
        return np.hstack([
            np.array(ecfp(smiles)),
            np.array(avalon(smiles)),
            np.array(desc2d(smiles)),
            np.array(gin(smiles)),
        ])

    X_train = featurize(train_smiles)
    X_test  = featurize(test_smiles)
    y_train = train_combined["Y"].values

    model = CatBoostClassifier(iterations=100, random_seed=42, verbose=0)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]

    result = group.evaluate({"hia_hou": preds}, benchmark="hia_hou")
    metric = list(result["hia_hou"].keys())[0]
    score  = result["hia_hou"][metric]

    print(f"  Task: hia_hou | {metric}: {score:.4f}  (expected ~0.98+)")
    assert score > 0.80, f"Score too low: {score:.4f}"


# ── Run all checks ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  MapLight + GNN Installation Check")
    print("=" * 50)

    results = [
        check("1. Imports",               check_imports),
        check("2. Fingerprints",           check_fingerprints),
        check("3. GIN embeddings",         check_gin_embeddings),
        check("4. TDC dataset loading",    check_tdc),
        check("5. End-to-end on hia_hou",  check_end_to_end),
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