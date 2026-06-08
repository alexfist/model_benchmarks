from deepmol.compound_featurization import MorganFingerprint
from deepmol.datasets.datasets import SmilesDataset
from deepmol.models import SklearnModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
smiles =[
            "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        ]
labels = [1,0]
dataset = SmilesDataset(smiles = smiles, y=labels)
fp = MorganFingerprint(radius=2, size=2048)
feat_dataset = fp.featurize(dataset)
clf = RandomForestClassifier();
m = SklearnModel(model = clf, task = 'classification')
m.fit(feat_dataset)

proba = m.predict_proba(feat_dataset)
print(f"proba type: {type(proba)}")
print(f"proba type: {np.array(proba).shape}")
print(f"proba value: {proba}")
