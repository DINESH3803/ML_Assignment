"""
Exploration script for ML Assignment 1 (Polynomial Regression).

For each problem (var1, var2) this script:
  1. Prints basic dataset statistics.
  2. Runs K-fold cross-validation over polynomial degrees using all features,
     reporting train/validation MSE and R^2 so the bias-variance trade-off is visible.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

ROLL = "BT2024199"
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))


def load(var):
    train = pd.read_csv(os.path.join(HERE, f"{ROLL}_train_var{var}.csv"))
    test = pd.read_csv(os.path.join(HERE, f"{ROLL}_test_var{var}.csv"))
    return train, test


def make_model(degree, alpha=0.0):
    reg = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        reg,
    )


def cv_scores(X, y, degree, alpha=0.0, k=5):
    cv = KFold(n_splits=k, shuffle=True, random_state=SEED)
    res = cross_validate(
        make_model(degree, alpha), X, y, cv=cv,
        scoring=("neg_mean_squared_error", "r2"), return_train_score=True,
    )
    return {
        "train_mse": -res["train_neg_mean_squared_error"].mean(),
        "val_mse": -res["test_neg_mean_squared_error"].mean(),
        "val_r2": res["test_r2"].mean(),
    }


def describe(var):
    train, test = load(var)
    print(f"\n{'=' * 70}\nvar{var}: train {train.shape}, test {test.shape}")
    print(train.describe().T[["mean", "std", "min", "max"]].round(3))
    feats = [c for c in train.columns if c != "y"]
    # Fraction of values clipped at +-1 (data appears to be clipped to [-1, 1])
    clipped = (train[feats].abs() >= 1.0).mean().round(3)
    print("fraction of values at +-1:\n", clipped.to_string())
    print("corr with y:\n", train.corr()["y"].drop("y").round(3).to_string())


def degree_sweep(var, max_degree, alpha=0.0, k=5):
    train, _ = load(var)
    X, y = train.drop(columns="y").values, train["y"].values
    fold_size = len(X) * (k - 1) / k  # rows available to fit in each CV fold
    print(f"\nvar{var} degree sweep (all features, alpha={alpha})")
    print(f"{'deg':>4} {'n_terms':>8} {'train_mse':>12} {'val_mse':>12} {'val_r2':>8}")
    for d in range(1, max_degree + 1):
        n_terms = PolynomialFeatures(d, include_bias=False).fit(X[:1]).n_output_features_
        # Unregularised OLS becomes ill-posed as n_terms approaches the fold size
        if alpha == 0 and n_terms >= 0.8 * fold_size:
            print(f"{d:>4} {n_terms:>8}  (skipped: too many terms for OLS)")
            continue
        s = cv_scores(X, y, d, alpha, k)
        print(f"{d:>4} {n_terms:>8} {s['train_mse']:>12.4f} {s['val_mse']:>12.4f} {s['val_r2']:>8.4f}")


if __name__ == "__main__":
    for v in (1, 2):
        describe(v)
    degree_sweep(1, 10)
    degree_sweep(2, 20)
