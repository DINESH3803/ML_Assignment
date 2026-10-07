"""
Final training + inference pipeline for ML Assignment 1 (Polynomial Regression).

Model: PolynomialFeatures(degree) -> StandardScaler -> Ridge(alpha)
Selection: grid search over (degree, alpha) with repeated 5-fold CV, minimising MSE.
Outputs (per problem):
  - results/cv_var{N}.csv          full CV grid
  - results/curve_var{N}.png       train/val MSE vs degree (best alpha per degree)
  - results/heatmap_var{N}.png     val MSE over degree x alpha
  - BT2024199_pred_var{N}.csv      test predictions (sample_submission format)
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

ROLL = "BT2024199"
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

SEARCH = {
    1: {"degrees": range(2, 9), "alphas": np.logspace(-3, 3, 13)},
    2: {"degrees": range(4, 19), "alphas": np.logspace(-4, 2, 13)},
}


def build_pipeline():
    return Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scale", StandardScaler()),
        ("ridge", Ridge()),
    ])


def tune(X, y, degrees, alphas):
    grid = GridSearchCV(
        build_pipeline(),
        {"poly__degree": list(degrees), "ridge__alpha": list(alphas)},
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        refit="mse",
        cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED),
        return_train_score=True,
        n_jobs=-1,
    )
    grid.fit(X, y)
    res = pd.DataFrame({
        "degree": grid.cv_results_["param_poly__degree"].astype(int),
        "alpha": grid.cv_results_["param_ridge__alpha"].astype(float),
        "train_mse": -grid.cv_results_["mean_train_mse"],
        "val_mse": -grid.cv_results_["mean_test_mse"],
        "val_mse_std": grid.cv_results_["std_test_mse"],
        "val_r2": grid.cv_results_["mean_test_r2"],
    })
    return grid, res


def plot(var, res, best):
    # Best alpha per degree -> bias/variance curve
    per_deg = res.loc[res.groupby("degree")["val_mse"].idxmin()]
    ols = res[res["alpha"] == res["alpha"].min()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(per_deg["degree"], per_deg["train_mse"], "o-", label="train (best α)")
    ax.plot(per_deg["degree"], per_deg["val_mse"], "s-", label="validation (best α)")
    ax.plot(ols["degree"], ols["val_mse"], "x--", alpha=0.6,
            label=f"validation (α={res['alpha'].min():g}, ~OLS)")
    ax.axvline(best["poly__degree"], color="grey", ls=":", label="chosen degree")
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title(f"var{var}: MSE vs degree (repeated 5-fold CV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/curve_var{var}.png", dpi=150)
    plt.close(fig)

    pivot = res.pivot(index="degree", columns="alpha", values="val_mse")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.log10(pivot.values), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(pivot.columns)), [f"{a:.0e}" for a in pivot.columns], rotation=60)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("α (Ridge)")
    ax.set_ylabel("degree")
    ax.set_title(f"var{var}: log10(validation MSE)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"{OUT}/heatmap_var{var}.png", dpi=150)
    plt.close(fig)


def run(var):
    train = pd.read_csv(os.path.join(HERE, f"{ROLL}_train_var{var}.csv"))
    test = pd.read_csv(os.path.join(HERE, f"{ROLL}_test_var{var}.csv"))
    feats = [c for c in train.columns if c != "y"]
    X, y = train[feats].values, train["y"].values

    grid, res = tune(X, y, **SEARCH[var])
    res.to_csv(f"{OUT}/cv_var{var}.csv", index=False)
    best = grid.best_params_
    row = res.loc[res["val_mse"].idxmin()]
    plot(var, res, best)

    model = grid.best_estimator_  # already refit on the full training set
    train_pred = model.predict(X)
    n_terms = model.named_steps["poly"].n_output_features_
    print(f"\nvar{var}: best degree={best['poly__degree']}, alpha={best['ridge__alpha']:.4g}, "
          f"terms={n_terms}")
    print(f"  CV  val MSE={row.val_mse:.4f} ± {row.val_mse_std:.4f}, val R2={row.val_r2:.4f}")
    print(f"  full-train MSE={mean_squared_error(y, train_pred):.4f}, "
          f"R2={r2_score(y, train_pred):.4f}")

    pred = model.predict(test[feats].values)
    pd.DataFrame({"y": pred}).to_csv(os.path.join(HERE, f"{ROLL}_pred_var{var}.csv"), index=False)
    print(f"  wrote {ROLL}_pred_var{var}.csv ({len(pred)} rows)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for v in (1, 2):
        run(v)
