"""
Refinement experiments:
  - Ridge-regularised degree sweep (lets us try higher degrees safely).
  - Feature ablation: drop one feature at a time and check CV MSE.
"""
import numpy as np

from explore import cv_scores, load


def ridge_sweep(var, degrees, alphas):
    train, _ = load(var)
    X, y = train.drop(columns="y").values, train["y"].values
    print(f"\nvar{var} ridge sweep (val MSE, 5-fold CV)")
    print("deg  " + " ".join(f"a={a:<8g}" for a in alphas))
    best = (np.inf, None, None)
    for d in degrees:
        row = []
        for a in alphas:
            m = cv_scores(X, y, d, a)["val_mse"]
            row.append(m)
            if m < best[0]:
                best = (m, d, a)
        print(f"{d:>3}  " + " ".join(f"{m:<10.4f}" for m in row))
    print(f"best: val_mse={best[0]:.4f} at degree={best[1]}, alpha={best[2]}")
    return best


def ablation(var, degree, alpha):
    train, _ = load(var)
    feats = [c for c in train.columns if c != "y"]
    y = train["y"].values
    base = cv_scores(train[feats].values, y, degree, alpha)["val_mse"]
    print(f"\nvar{var} ablation @ degree={degree}, alpha={alpha}: all features val_mse={base:.4f}")
    for f in feats:
        keep = [c for c in feats if c != f]
        m = cv_scores(train[keep].values, y, degree, alpha)["val_mse"]
        print(f"  drop {f}: val_mse={m:.4f} ({m - base:+.4f})")


if __name__ == "__main__":
    alphas = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    b1 = ridge_sweep(1, range(2, 8), alphas)
    ablation(1, b1[1], b1[2])
    b2 = ridge_sweep(2, range(5, 16), alphas)
    ablation(2, b2[1], b2[2])
