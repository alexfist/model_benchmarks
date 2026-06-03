"""
check_install.py
----------------
Verifies MiniMol is installed correctly before running the full benchmark.

Usage:
    python check_install.py

Checks:
    1. All imports work
    2. MiniMol generates fingerprints for test molecules
    3. TDC dataset loads correctly
    4. Full end-to-end run on one small task (hia_hou)

Prints a clear PASS/FAIL for each step.
"""

import sys

PASS = "  ✓ PASS"
FAIL = "  ✗ FAIL"

def check(label, fn):
    """Runs fn(), prints PASS or FAIL, returns True/False."""
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
    import torch
    import torch_scatter
    import torch_sparse
    import torch_cluster
    import torch_geometric
    from minimol import Minimol
    from tdc.benchmark_group import admet_group
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import pandas as pd
    import joblib
    print(f"  torch:          {torch.__version__}")
    print(f"  torch_geometric: ok")
    print(f"  minimol:        ok")
    print(f"  PyTDC:          ok")
    print(f"  scikit-learn:   ok")


# ── Check 2: MiniMol fingerprints ─────────────────────────────────────────────

def check_fingerprints():
    from minimol import Minimol
    import torch
    import numpy as np
    model = Minimol()

    test_molecules = [
        "CC(=O)Oc1ccccc1C(=O)O",          # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    ]

    fps = model(test_molecules)
    fps_array = np.array(fps)
    assert fps_array.shape == (3,512), f"Expected fingerprint shape (3,512), got {fps_array.shape}"
    print(f"  Fingerprint shape: {fps_array.shape}")


# ── Check 3: TDC dataset loading ──────────────────────────────────────────────

def check_tdc():
    from tdc.benchmark_group import admet_group

    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    train_val = benchmark["train_val"]
    test      = benchmark["test"]

    print(f"  hia_hou loaded successfully")
    print(f"  train_val size: {len(train_val)}")
    print(f"  test size:      {len(test)}")


# ── Check 4: End-to-end run on hia_hou ────────────────────────────────────────

def check_end_to_end():
    from minimol import Minimol
    from tdc.benchmark_group import admet_group
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    test = benchmark["test"]

    train, valid = group.get_train_valid_split(
        benchmark="hia_hou",
        split_type="default",
        seed=0
    )

    model = Minimol()
    train_fps = model(train["Drug"].tolist()).numpy()
    valid_fps = model(valid["Drug"].tolist()).numpy()
    test_fps  = model(test["Drug"].tolist()).numpy()

    X = np.vstack([train_fps, valid_fps])
    y = np.concatenate([train["Y"].values, valid["Y"].values])

    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    preds = clf.predict_proba(test_fps)[:, 1]

    result = group.evaluate({"hia_hou": preds}, benchmark="hia_hou")
    metric = list(result["hia_hou"].keys())[0]
    score  = result["hia_hou"][metric]

    print(f"  Task:   hia_hou")
    print(f"  Metric: {metric}")
    print(f"  Score:  {score:.4f}  (expected ~0.99)")
    assert score > 0.80, f"Score too low: {score:.4f} — something may be wrong"


# ── Run all checks ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  MiniMol Installation Check")
    print("=" * 50)

    results = [
        check("1. Imports",              check_imports),
        check("2. Fingerprint generation", check_fingerprints),
        check("3. TDC dataset loading",  check_tdc),
        check("4. End-to-end on hia_hou", check_end_to_end),
    ]

    print("\n" + "=" * 50)
    passed = sum(results)
    total  = len(results)

    if passed == total:
        print(f"  All {total} checks passed. Ready to run benchmark!")
    else:
        print(f"  {passed}/{total} checks passed. Fix the failing checks before running the benchmark.")

    print("=" * 50)
    sys.exit(0 if passed == total else 1)