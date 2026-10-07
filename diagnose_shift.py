"""
Diagnose train/test covariate shift and how stable test predictions are across
candidate models (helps decide whether a high-degree model is extrapolating).
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "BT2024199"
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = {1: [(3, 1), (4, 10), (5, 31.62), (6, 100)], 2: [(8, 0.1), (10, 1), (12, 1)]}


def diagnose(var, cands):
    tr = pd.read_csv(os.path.join(HERE, f"{ROLL}_train_var{var}.csv"))
    te = pd.read_csv(os.path.join(HERE, f"{ROLL}_test_var{var}.csv"))
    feats = list(te.columns)
    clip_tr, clip_te = tr[feats].abs() >= 1, te[feats].abs() >= 1
    print(f"\n===== var{var}")
    print("frac at +-1   train:", clip_tr.mean().round(3).to_dict())
    print("frac at +-1   test :", clip_te.mean().round(3).to_dict())
    print("n features clipped per row  train:", clip_tr.sum(1).value_counts().sort_index().to_dict())
    print("n features clipped per row  test :", clip_te.sum(1).value_counts().sort_index().to_dict())
    print("std  train:", tr[feats].std().round(3).to_dict())
    print("std  test :", te[feats].std().round(3).to_dict())

    X_tr, y_tr, X_te = tr[feats].values, tr["y"].values, te[feats].values
    preds = {}
    for d, a in cands:
        m = make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(), Ridge(alpha=a))
        m.fit(X_tr, y_tr)
        p = preds[(d, a)] = m.predict(X_te)
        print(f"deg={d:>2} a={a:<6g} test pred std={p.std():.3f} min={p.min():.2f} max={p.max():.2f}")
    keys = list(preds)
    for k1, k2 in zip(keys, keys[1:]):
        diff = preds[k1] - preds[k2]
        print(f"  RMS diff {k1} vs {k2}: {np.sqrt((diff ** 2).mean()):.3f}")


if __name__ == "__main__":
    for var, cands in CANDIDATES.items():
        diagnose(var, cands)
