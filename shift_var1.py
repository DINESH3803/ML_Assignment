"""
Covariate-shift-aware model selection for var1.

The var1 test set has far more features clipped at +-1 than the train set.
We estimate importance weights w(x) = p_test(k) / p_train(k), where k is the number
of clipped features in a row, and use them to:
  (a) compute an importance-weighted validation MSE (estimates test MSE), and
  (b) optionally fit a weighted Ridge (sample_weight = w).
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
ROLL, SEED = "BT2024199", 42
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def clip_count(df, feats):
    return (df[feats].abs() >= 1).sum(axis=1).values


def importance_weights(tr, te, feats):
    k_tr, k_te = clip_count(tr, feats), clip_count(te, feats)
    n = len(feats) + 1
    p_tr = np.bincount(k_tr, minlength=n) / len(k_tr)
    p_te = np.bincount(k_te, minlength=n) / len(k_te)
    w = p_te[k_tr] / np.maximum(p_tr[k_tr], 1e-12)
    return w / w.mean()


def evaluate(Xp, y, w, alpha, weighted_fit, splits):
    """CV on pre-expanded polynomial features Xp.

    PolynomialFeatures is a stateless transform, so expanding the full X once per
    degree is equivalent to expanding inside each fold (no leakage), but much cheaper.
    """
    plain, wtd = [], []
    for tri, vai in splits:
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        fit_kw = {"ridge__sample_weight": w[tri]} if weighted_fit else {}
        m.fit(Xp[tri], y[tri], **fit_kw)
        err = (m.predict(Xp[vai]) - y[vai]) ** 2
        plain.append(err.mean())
        wtd.append(np.average(err, weights=w[vai]))
    return np.mean(plain), np.mean(wtd)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_csv(os.path.join(HERE, f"{ROLL}_train_var1.csv"))
    te = pd.read_csv(os.path.join(HERE, f"{ROLL}_test_var1.csv"))
    feats = list(te.columns)
    X, y = tr[feats].values, tr["y"].values
    w = importance_weights(tr, te, feats)
    k_tr = clip_count(tr, feats)
    print("weights by clip-count:", {int(k): round(float(w[k_tr == k][0]), 2) for k in np.unique(k_tr)})
    print(f"effective sample size: {w.sum() ** 2 / (w ** 2).sum():.0f} / {len(w)}")

    # Same folds for every configuration (matches the original per-call RepeatedKFold)
    splits = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X))
    rows = []
    for degree in range(3, 8):
        Xp = PolynomialFeatures(degree, include_bias=False).fit_transform(X)
        for alpha in np.logspace(-1, 3, 9):
            for wf in (False, True):
                p, q = evaluate(Xp, y, w, alpha, wf, splits)
                rows.append(dict(degree=degree, alpha=alpha, weighted_fit=wf, val_mse=p, iw_val_mse=q))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "shift_var1.csv"), index=False)
    best = res.loc[res.groupby(["degree", "weighted_fit"])["iw_val_mse"].idxmin()]
    print("\nbest alpha per (degree, weighted_fit), ranked by importance-weighted val MSE:")
    print(best.sort_values("iw_val_mse").round(4).to_string(index=False))
