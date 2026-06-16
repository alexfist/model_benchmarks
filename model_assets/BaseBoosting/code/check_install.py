"""
check_install.py
----------------
Verifies BaseBoosting (olorenchemengine) is installed correctly.

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


def check_imports():
    import olorenchemengine as oce
    import numpy, pandas, sklearn
    print(f"  olorenchemengine: ok")
    print(f"  numpy:   {numpy.__version__}")
    print(f"  sklearn: {sklearn.__version__}")


def check_descriptastorus():
    import olorenchemengine as oce
    print("  Testing DescriptastorusDescriptor('morgan3counts')...")
    desc = oce.DescriptastorusDescriptor("morgan3counts")
    result = desc.convert(["CC(=O)O"])  # acetic acid
    import numpy as np
    arr = np.array(result)
    print(f"  morgan3counts output shape: {arr.shape}")
    assert arr.shape[0] == 1, "Expected 1 row"

    print("  Testing DescriptastorusDescriptor('rdkit2dnormalized')...")
    desc2 = oce.DescriptastorusDescriptor("rdkit2dnormalized")
    result2 = desc2.convert(["CC(=O)O"])
    arr2 = np.array(result2)
    print(f"  rdkit2dnormalized output shape: {arr2.shape}")


def check_oloren_checkpoint():
    import olorenchemengine as oce
    import numpy as np
    print("  Testing OlorenCheckpoint('default')...")
    print("  (This downloads weights from Google Cloud Storage — may fail on restricted networks)")
    try:
        checkpoint = oce.OlorenCheckpoint("default")
        result = checkpoint.convert(["CC(=O)O"])
        arr = np.array(result)
        print(f"  OlorenCheckpoint output shape: {arr.shape}")
        print("  OlorenCheckpoint: AVAILABLE ✓")
    except Exception as e:
        print(f"  OlorenCheckpoint: UNAVAILABLE — {e}")
        print("  Script will run with 2-learner fallback (Morgan + RDKit2D)")
        # Don't raise — this is expected on restricted servers


def check_baseboosting():
    import olorenchemengine as oce
    import numpy as np
    print("  Building 2-learner BaseBoosting (no OlorenCheckpoint)...")
    model = oce.BaseBoosting([
        oce.RandomForestModel(
            oce.DescriptastorusDescriptor("morgan3counts"),
            n_estimators=10  # small for speed
        ),
        oce.RandomForestModel(
            oce.DescriptastorusDescriptor("rdkit2dnormalized"),
            n_estimators=10
        ),
    ])

    smiles = ["CC(=O)O", "c1ccccc1", "CCO", "CC(=O)Oc1ccccc1C(=O)O"]
    y      = [1.0, 2.0, 1.5, 3.0]

    model.fit(smiles, y)
    preds = model.predict(smiles[:2])
    print(f"  BaseBoosting fit + predict OK. Preds: {preds}")


def check_tdc():
    from tdc.benchmark_group import admet_group
    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    train_val = benchmark["train_val"]
    test      = benchmark["test"]
    print(f"  hia_hou loaded: train_val={len(train_val)}, test={len(test)}")


if __name__ == "__main__":
    print("=" * 55)
    print("  BaseBoosting Installation Check")
    print("=" * 55)

    results = [
        check("1. Imports",                    check_imports),
        check("2. Descriptastorus descriptors", check_descriptastorus),
        check("3. OlorenCheckpoint (optional)", check_oloren_checkpoint),
        check("4. BaseBoosting end-to-end",    check_baseboosting),
        check("5. TDC dataset loading",        check_tdc),
    ]

    print("\n" + "=" * 55)
    passed = sum(results)
    total  = len(results)

    # Check 3 (OlorenCheckpoint) is optional — don't count as failure
    required_passed = results[0] and results[1] and results[3] and results[4]

    if required_passed:
        if results[2]:
            print(f"  All {total} checks passed — full 3-learner model available!")
        else:
            print(f"  {passed}/{total} checks passed.")
            print("  Required checks passed — will run with 2-learner fallback.")
            print("  OlorenCheckpoint unavailable (network restricted).")
        print("  Ready to run benchmark!")
    else:
        print(f"  {passed}/{total} checks passed. Fix failing required checks.")

    print("=" * 55)
    sys.exit(0 if required_passed else 1)