"""
Step 2 -- the dumb baseline.

UCI Superconductivity Dataset: 21,263 entries, 81 composition-derived
features, target = experimental critical temperature.

This has no bearing on the hypothesis. Its only jobs are (a) prove the
load -> split -> train -> evaluate plumbing works, and (b) give a reference
number to compare against published results on the same data.

The one thing that is NOT boilerplate here: the dataset contains 5,721
duplicate chemical formulas (15,542 unique formulas over 21,263 rows).
A plain random split puts the same material on both sides of the split and
inflates the score. We report both so the size of that inflation is visible
rather than hidden.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
RESULTS = os.path.join(ROOT, "results")
SEED = 0


def _fit_predict(X_tr, y_tr, X_te):
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=800, learning_rate=0.05,
                              num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, n_jobs=-1, verbose=-1)
    except ImportError:                                   # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=800, random_state=SEED)
    model.fit(X_tr, y_tr)
    return model.predict(X_te), model


def main() -> dict:
    train = pd.read_csv(os.path.join(RAW, "train.csv"))
    unique_m = pd.read_csv(os.path.join(RAW, "unique_m.csv"))

    assert len(train) == len(unique_m), "row alignment broken"
    assert np.allclose(train["critical_temp"], unique_m["critical_temp"]), \
        "critical_temp does not match between train.csv and unique_m.csv"

    y = train["critical_temp"].to_numpy()
    X = train.drop(columns=["critical_temp"]).to_numpy()
    groups = unique_m["material"].to_numpy()

    n_dup = len(groups) - pd.Series(groups).nunique()
    print(f"rows={len(y)}  features={X.shape[1]}  "
          f"unique formulas={pd.Series(groups).nunique()}  duplicates={n_dup}")

    out = {}

    # --- naive random split (what most write-ups of this dataset report) ---
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED)
    pred, _ = _fit_predict(Xtr, ytr, Xte)
    out["random_split"] = {"mae": float(mean_absolute_error(yte, pred)),
                           "rmse": float(np.sqrt(np.mean((yte - pred) ** 2))),
                           "r2": float(r2_score(yte, pred)),
                           "n_test": int(len(yte))}

    # --- formula-grouped split: no formula appears on both sides ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    itr, ite = next(gss.split(X, y, groups))
    assert not (set(groups[itr]) & set(groups[ite])), "group leakage"
    pred_g, _ = _fit_predict(X[itr], y[itr], X[ite])
    out["grouped_split"] = {"mae": float(mean_absolute_error(y[ite], pred_g)),
                            "rmse": float(np.sqrt(np.mean((y[ite] - pred_g) ** 2))),
                            "r2": float(r2_score(y[ite], pred_g)),
                            "n_test": int(len(ite))}

    # --- predict-the-mean control: any model must beat this ---
    out["mean_baseline"] = {
        "mae": float(mean_absolute_error(y[ite], np.full(len(ite), y[itr].mean()))),
        "r2": 0.0,
    }

    for name, m in out.items():
        print(f"  {name:15s} " + "  ".join(f"{k}={v:.4g}" for k, v in m.items()))
    print(f"  leakage inflation in R2: "
          f"{out['random_split']['r2'] - out['grouped_split']['r2']:+.4f}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "baseline_uci.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    main()
