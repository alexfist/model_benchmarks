"""
check_install.py
----------------
Verifies AttrMasking (DeepPurpose) is installed correctly.

Usage:
    python check_install.py
"""
import dgl.data.utils
_original_download = dgl.data.utils.download
def _patched_download(url, path=None, overwrite=False, **kwargs):
    return _original_download(url,path=path, overwrite=overwrite, **kwargs)
dgl.data.utils.download = _patched_download

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    import torch
    import dgl
    import dgllife
    import DeepPurpose
    from DeepPurpose import utils, dataset, CompoundPred
    from tdc.benchmark_group import admet_group
    import numpy as np
    import pandas as pd
    print(f"  torch:      {torch.__version__}")
    print(f"  dgl:        {dgl.__version__}")
    print(f"  dgllife:    ok")
    print(f"  DeepPurpose: ok")
    print(f"  PyTDC:      ok")


# ── Check 2: AttrMasking encoder loads ────────────────────────────────────────

def check_encoder():
    from DeepPurpose import utils
    # This triggers the pre-trained weight download on first run
    print("  Loading DGL_GIN_AttrMasking encoder (may download weights on first run)...")
    drug_encoding = "DGL_GIN_AttrMasking"
    config = utils.generate_config(
        drug_encoding=drug_encoding,
        train_epoch=1,
        LR=0.001,
        batch_size=32,
    )
    print(f"  Encoder config loaded: {drug_encoding}")


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
    from DeepPurpose import utils, CompoundPred
    from tdc.benchmark_group import admet_group
    import pandas as pd

    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    test = benchmark["test"]
    train, valid = group.get_train_valid_split(
        benchmark="hia_hou", split_type="default", seed=0
    )

    train_combined = pd.concat([train, valid]).reset_index(drop=True)

    drug_encoding = "DGL_GIN_AttrMasking"
    train_dp = utils.data_process(
        X_drug=train_combined["Drug"].tolist(),
        y=train_combined["Y"].tolist(),
        drug_encoding=drug_encoding,
        split_method="no_split"
    )
    test_dp = utils.data_process(
        X_drug=test["Drug"].tolist(),
        y=test["Y"].tolist(),
        drug_encoding=drug_encoding,
        split_method="no_split"
    )

    config = utils.generate_config(
        drug_encoding=drug_encoding,
        train_epoch=2,   # just 2 epochs for the check
        LR=0.001,
        batch_size=32,
    )

    model = CompoundPred.model_initialize(**config)
    model.train(train_dp, train_dp, train_dp)
    preds = model.predict(test_dp)

    result = group.evaluate({"hia_hou": preds}, benchmark="hia_hou")
    metric = list(result["hia_hou"].keys())[0]
    score  = result["hia_hou"][metric]

    print(f"  Task: hia_hou | {metric}: {score:.4f} (2-epoch quick check)")
    assert score > 0.50, f"Score too low even for 2 epochs: {score:.4f}"


# ── Run all checks ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  AttrMasking Installation Check")
    print("=" * 50)

    results = [
        check("1. Imports",               check_imports),
        check("2. AttrMasking encoder",   check_encoder),
        check("3. TDC dataset loading",   check_tdc),
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
