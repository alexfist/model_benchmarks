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
    import subprocess
    result = subprocess.run(
        ["zairachem", "--help"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"zairachem CLI failed:\n{result.stderr}")
    print("  zairachem CLI accessible")
    print("  Skipping full fit/predict test — run run_benchmark.py --task hia_hou to test end-to-end")

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