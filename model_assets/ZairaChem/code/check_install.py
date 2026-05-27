"""
check_install.py
----------------
Verifies ZairaChem is installed correctly before running the full benchmark.

Usage:
    python check_install.py
"""

import sys
import subprocess
import os

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


# ── Check 1: ZairaChem CLI available ──────────────────────────────────────────

def check_cli():
    result = subprocess.run(
        ["zairachem", "--help"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"zairachem CLI not found. Make sure 'zairachem' conda env is active.\n{result.stderr}")
    print("  zairachem CLI: ok")


# ── Check 2: Python imports ────────────────────────────────────────────────────

def check_imports():
    from tdc.benchmark_group import admet_group
    import pandas as pd
    import numpy as np
    print("  PyTDC:   ok")
    print("  pandas:  ok")
    print("  numpy:   ok")


# ── Check 3: TDC dataset loading ──────────────────────────────────────────────

def check_tdc():
    from tdc.benchmark_group import admet_group
    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    train_val = benchmark["train_val"]
    test      = benchmark["test"]
    print(f"  hia_hou loaded: train_val={len(train_val)}, test={len(test)}")


# ── Check 4: End-to-end fit and predict on hia_hou ────────────────────────────

def check_end_to_end():
    from tdc.benchmark_group import admet_group
    import pandas as pd
    import tempfile, shutil

    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    test = benchmark["test"]
    train, valid = group.get_train_valid_split(
        benchmark="hia_hou", split_type="default", seed=0
    )

    train_combined = pd.concat([train, valid]).reset_index(drop=True)
    train_combined = train_combined.rename(columns={"Drug": "smiles", "Y": "activity"})
    test_renamed   = test.rename(columns={"Drug": "smiles", "Y": "activity"})

    tmp_dir = tempfile.mkdtemp()
    train_path  = os.path.join(tmp_dir, "train.csv")
    test_path   = os.path.join(tmp_dir, "test.csv")
    model_dir   = os.path.join(tmp_dir, "model")
    pred_dir    = os.path.join(tmp_dir, "predictions")

    try:
        train_combined.to_csv(train_path, index=False)
        test_renamed.to_csv(test_path, index=False)

        # Fit
        fit_result = subprocess.run(
            ["zairachem", "fit", "-i", train_path, "-m", model_dir],
            capture_output=True, text=True
        )
        if fit_result.returncode != 0:
            raise RuntimeError(f"zairachem fit failed:\n{fit_result.stderr}")

        # Predict
        pred_result = subprocess.run(
            ["zairachem", "predict", "-i", test_path, "-m", model_dir, "-o", pred_dir],
            capture_output=True, text=True
        )
        if pred_result.returncode != 0:
            raise RuntimeError(f"zairachem predict failed:\n{pred_result.stderr}")

        # Read predictions
        pred_file = os.path.join(pred_dir, "predictions.csv")
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Predictions file not found at {pred_file}")

        preds_df = pd.read_csv(pred_file)
        print(f"  Predictions shape: {preds_df.shape}")
        print(f"  Columns: {list(preds_df.columns)}")
        print(f"  End-to-end check passed on hia_hou")

    finally:
        shutil.rmtree(tmp_dir)


# ── Run all checks ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  ZairaChem Installation Check")
    print("=" * 50)

    results = [
        check("1. ZairaChem CLI",         check_cli),
        check("2. Python imports",         check_imports),
        check("3. TDC dataset loading",    check_tdc),
        check("4. End-to-end on hia_hou",  check_end_to_end),
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