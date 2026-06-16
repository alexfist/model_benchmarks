"""
check_install.py
----------------
Verifies CaliciBoost dependencies are installed correctly.

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
    import numpy, pandas, sklearn, xgboost, joblib
    from xgboost import XGBRegressor
    print(f"  numpy:    {numpy.__version__}")
    print(f"  sklearn:  {sklearn.__version__}")
    print(f"  xgboost:  {xgboost.__version__}")


def check_padelpy():
    from padelpy import from_smiles
    print("  padelpy imported OK")
    # Quick test on aspirin
    desc = from_smiles("CC(=O)Oc1ccccc1C(=O)O",
                       fingerprints=False, descriptors=True)
    n = len(desc)
    print(f"  PaDEL descriptors computed for aspirin: {n} features")
    assert n > 100, f"Expected >100 PaDEL descriptors, got {n}"


def check_java():
    import subprocess
    result = subprocess.run(["java", "-version"],
                            capture_output=True, text=True)
    version_line = result.stderr.strip().split("\n")[0]
    print(f"  Java: {version_line}")
    assert result.returncode == 0, "Java not found — required for PaDEL"


def check_tdc():
    from tdc.benchmark_group import admet_group
    group = admet_group(path="../data")
    benchmark = group.get("caco2_wang")
    train_val = benchmark["train_val"]
    test      = benchmark["test"]
    print(f"  caco2_wang loaded: train_val={len(train_val)}, test={len(test)}")


def check_end_to_end():
    from padelpy import from_smiles
    from tdc.benchmark_group import admet_group
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    group = admet_group(path="../data")
    benchmark = group.get("caco2_wang")
    test = benchmark["test"]
    train, valid = group.get_train_valid_split(
        benchmark="caco2_wang", split_type="default", seed=1
    )

    # Small subset for speed
    train_small = train.head(50)
    test_small  = test.head(20)

    def featurize(smiles_list):
        rows = []
        for smi in smiles_list:
            try:
                desc = from_smiles(smi, fingerprints=False, descriptors=True)
                row  = [float(v) if v not in ("", None) else 0.0
                        for v in desc.values()]
            except Exception:
                row = [0.0] * 1875
            rows.append(row)
        X = np.array(rows, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_train = featurize(train_small["Drug"].tolist())
    X_test  = featurize(test_small["Drug"].tolist())

    model = XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
    model.fit(X_train, train_small["Y"].values)
    preds = model.predict(X_test)

    mae = mean_absolute_error(test_small["Y"].values, preds)
    print(f"  Quick end-to-end MAE on 20 test molecules: {mae:.4f}")
    print(f"  (not meaningful at 50 training examples — just verifying pipeline)")


if __name__ == "__main__":
    print("=" * 50)
    print("  CaliciBoost Installation Check")
    print("=" * 50)

    results = [
        check("1. Imports",           check_imports),
        check("2. Java (for PaDEL)",  check_java),
        check("3. padelpy + PaDEL",   check_padelpy),
        check("4. TDC dataset",        check_tdc),
        check("5. End-to-end smoke",   check_end_to_end),
    ]

    print("\n" + "=" * 50)
    passed = sum(results)
    total  = len(results)

    if passed == total:
        print(f"  All {total} checks passed. Ready to run benchmark!")
    else:
        print(f"  {passed}/{total} checks passed. Fix failing checks before running.")

    print("=" * 50)
    sys.exit(0 if passed == total else 1)