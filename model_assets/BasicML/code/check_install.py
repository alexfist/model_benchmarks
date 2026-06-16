"""
check_install.py
----------------
Verifies BasicML dependencies are installed correctly.

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
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    from tdc.benchmark_group import admet_group
    print(f"  numpy:      {numpy.__version__}")
    print(f"  sklearn:    {sklearn.__version__}")
    print(f"  xgboost:    {xgboost.__version__}")
    print(f"  rdkit:      ok")
    print(f"  PyTDC:      ok")


def check_descriptors():
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    import numpy as np

    DESCRIPTOR_NAMES = [
        "MolWt", "HeavyAtomMolWt", "ExactMolWt",
        "NumValenceElectrons", "NumRadicalElectrons",
        "MaxPartialCharge", "MinPartialCharge",
        "MaxAbsPartialCharge", "MinAbsPartialCharge",
        "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
        "NumHDonors", "NumHAcceptors", "MolLogP", "MolMR",
        "TPSA", "LabuteASA", "BalabanJ", "BertzCT",
        "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
        "Chi0", "Chi0n", "Chi1", "Chi1n",
        "NumAromaticRings", "NumRotatableBonds", "RingCount",
    ]

    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    mol = Chem.MolFromSmiles(test_smiles)
    assert mol is not None, "RDKit failed to parse test SMILES"

    row = []
    for name in DESCRIPTOR_NAMES:
        if hasattr(Descriptors, name):
            val = getattr(Descriptors, name)(mol)
        elif hasattr(Lipinski, name):
            val = getattr(Lipinski, name)(mol)
        else:
            raise ValueError(f"Descriptor not found: {name}")
        row.append(float(val))

    assert len(row) == 31, f"Expected 31 descriptors, got {len(row)}"
    print(f"  Computed 31 descriptors for aspirin: OK")
    print(f"  MolWt={row[0]:.2f}, MolLogP={row[14]:.2f}, TPSA={row[16]:.2f}")


def check_tdc():
    from tdc.benchmark_group import admet_group
    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    train_val = benchmark["train_val"]
    test      = benchmark["test"]
    print(f"  hia_hou loaded: train_val={len(train_val)}, test={len(test)}")


def check_end_to_end():
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    from tdc.benchmark_group import admet_group
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    import numpy as np

    DESCRIPTOR_NAMES = [
        "MolWt", "HeavyAtomMolWt", "ExactMolWt",
        "NumValenceElectrons", "NumRadicalElectrons",
        "MaxPartialCharge", "MinPartialCharge",
        "MaxAbsPartialCharge", "MinAbsPartialCharge",
        "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
        "NumHDonors", "NumHAcceptors", "MolLogP", "MolMR",
        "TPSA", "LabuteASA", "BalabanJ", "BertzCT",
        "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
        "Chi0", "Chi0n", "Chi1", "Chi1n",
        "NumAromaticRings", "NumRotatableBonds", "RingCount",
    ]

    def featurize(smiles_list):
        rows = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            row = []
            for name in DESCRIPTOR_NAMES:
                try:
                    if hasattr(Descriptors, name):
                        val = getattr(Descriptors, name)(mol)
                    else:
                        val = getattr(Lipinski, name)(mol)
                    val = float(val) if val is not None else 0.0
                    if not np.isfinite(val):
                        val = 0.0
                except:
                    val = 0.0
                row.append(val)
            rows.append(row)
        return np.array(rows)

    group = admet_group(path="../data")
    benchmark = group.get("hia_hou")
    test = benchmark["test"]
    train, valid = group.get_train_valid_split(
        benchmark="hia_hou", split_type="default", seed=1
    )

    X_train = featurize(train["Drug"].tolist())
    X_test  = featurize(test["Drug"].tolist())

    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_train, train["Y"].values)
    preds = clf.predict_proba(X_test)[:, 1]

    result = group.evaluate({"hia_hou": preds})
    metric = list(result["hia_hou"].keys())[0]
    score  = result["hia_hou"][metric]
    print(f"  hia_hou quick check | {metric}: {score:.4f} (RF, 50 trees)")
    assert score > 0.5, f"Score too low: {score:.4f}"


if __name__ == "__main__":
    print("=" * 50)
    print("  Basic ML Installation Check")
    print("=" * 50)

    results = [
        check("1. Imports",               check_imports),
        check("2. 31 RDKit descriptors",  check_descriptors),
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