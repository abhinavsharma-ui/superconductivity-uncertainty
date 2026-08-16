"""
Does Allen-Dynes breakdown track lambda, once the SHAPE channel is controlled?

Three things this settles, in order.

1. THE SHAPE CONFOUND.
   The mu* constant in build_physics_dataset.py was calibrated on Einstein
   spectra, where w_2/w_log = 1 exactly and therefore f2 = 1 exactly. Real
   materials are not like that: w_ratio spans roughly [1.03, 3.06]. Worse,
   w_ratio is itself correlated with lambda, so the f2 (shape) channel and the
   f1 (strong-coupling) channel are entangled in the real data. ad_error
   cannot be attributed to lambda without holding w_ratio fixed. This is
   structurally the same problem as the training-density confound, but it
   lives in physics space rather than sample space.

   Test: regress ad_error on w_ratio. If f2 error drives the post-calibration
   residual, ad_error should correlate with w_ratio and should largely vanish
   for materials near w_ratio = 1. If it does NOT correlate, the calibrated
   constant itself is off and needs re-deriving.

2. MAGNITUDE, NOT SIGN.
   Breakdown is a magnitude. Signed ad_error changes sign -- the mu* = 0 sweep
   crosses zero near lambda ~ 20, and post-calibration the median sits just
   below 1 where pre-calibration it was well above. A correlation computed on
   signed values can cancel to nothing while |ad_error| grows monotonically.
   Both are reported; the premise test is on |ad_error|.

3. THE HEADLINE.
   corr(lambda, |ad_error|) holding w_ratio fixed -- by partial correlation
   AND by stratifying, because a partial correlation assumes a linear control
   and the stratified version does not.

Everything is also reported on the uncapped subset, because rows that hit the
Matsubara cap have a Tc_ME biased low, and capping is not random in lambda.

Usage
-----
    python src/analyze_ad_error.py
    python src/analyze_ad_error.py --csv data/processed/physics_dataset.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "data", "processed", "physics_dataset.csv")


def partial_corr(x, y, z, method: str = "pearson"):
    """
    corr(x, y) with z held fixed, via the standard three-correlation identity.
    For Spearman, rank-transform first and then apply the same identity --
    which is what a rank partial correlation is.
    """
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    if method == "spearman":
        x, y, z = (stats.rankdata(v) for v in (x, y, z))
    rxy = stats.pearsonr(x, y)[0]
    rxz = stats.pearsonr(x, z)[0]
    ryz = stats.pearsonr(y, z)[0]
    denom = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 0.0))
    if denom <= 0:
        return float("nan"), float("nan")
    r = (rxy - rxz * ryz) / denom
    dof = len(x) - 3
    if dof <= 0 or abs(r) >= 1:
        return float(r), float("nan")
    t = r * np.sqrt(dof / (1 - r ** 2))
    return float(r), float(2 * stats.t.sf(abs(t), dof))


def _both(x, y):
    """Pearson and Spearman with p-values, as one line of text."""
    rp, pp = stats.pearsonr(x, y)
    rs, ps = stats.spearmanr(x, y)
    return f"pearson {rp:+.3f} (p={pp:.2g})   spearman {rs:+.3f} (p={ps:.2g})"


def report(df: pd.DataFrame, label: str) -> None:
    lam = df["lambda"].to_numpy(float)
    r = df["w_ratio"].to_numpy(float)
    ae = df["ad_error"].to_numpy(float)
    aae = np.abs(ae)
    n = len(df)

    print("\n" + "=" * 72)
    print(f"{label}   (n = {n})")
    print("=" * 72)

    print(f"\nlambda    range [{lam.min():.3f}, {lam.max():.3f}]  "
          f"median {np.median(lam):.3f}")
    print(f"w_ratio   range [{r.min():.3f}, {r.max():.3f}]  "
          f"median {np.median(r):.3f}")
    print(f"ad_error  range [{ae.min():+.4f}, {ae.max():+.4f}]  "
          f"median {np.median(ae):+.4f}   |ad_error| median {np.median(aae):.4f}")
    print(f"          Tc_ME/Tc_AD median {np.exp(np.median(ae)):.4f}")

    # ---- 1. the confound itself -------------------------------------------
    print(f"\n[confound]  corr(w_ratio, lambda)")
    print(f"            {_both(r, lam)}")

    # ---- 2. shape test ----------------------------------------------------
    print(f"\n[shape]     corr(ad_error, w_ratio)")
    print(f"            {_both(ae, r)}")
    sl, ic, rv, pv, se = stats.linregress(r, ae)
    print(f"            OLS ad_error ~ w_ratio:  slope {sl:+.4f} +/- {se:.4f}"
          f"   intercept {ic:+.4f}   R2 {rv ** 2:.3f}")
    print(f"            => extrapolated to w_ratio = 1 (the Einstein case the")
    print(f"               constant was calibrated on): ad_error = {ic + sl:+.4f}"
          f"  (Tc_ME/Tc_AD = {np.exp(ic + sl):.4f})")

    near = r < 1.10
    if near.sum() >= 5:
        print(f"            near-Einstein subset w_ratio < 1.10  (n={int(near.sum())}): "
              f"median ad_error {np.median(ae[near]):+.4f} "
              f"(Tc_ME/Tc_AD {np.exp(np.median(ae[near])):.4f})")
        print(f"            rest                w_ratio >= 1.10 (n={int((~near).sum())}): "
              f"median ad_error {np.median(ae[~near]):+.4f} "
              f"(Tc_ME/Tc_AD {np.exp(np.median(ae[~near])):.4f})")

    pr, pp_ = partial_corr(ae, r, lam)
    print(f"            corr(ad_error, w_ratio | lambda held fixed): "
          f"{pr:+.3f} (p={pp_:.2g})")

    # ---- 3. the premise test ---------------------------------------------
    print(f"\n[premise]   SIGNED   corr(lambda, ad_error)")
    print(f"            {_both(lam, ae)}")
    print(f"            MAGNITUDE corr(lambda, |ad_error|)")
    print(f"            {_both(lam, aae)}")

    for meth in ("pearson", "spearman"):
        pr, pp_ = partial_corr(lam, aae, r, method=meth)
        print(f"            partial corr(lambda, |ad_error| | w_ratio) "
              f"[{meth:8s}]: {pr:+.3f} (p={pp_:.2g})")
    for meth in ("pearson", "spearman"):
        pr, pp_ = partial_corr(r, aae, lam, method=meth)
        print(f"            partial corr(w_ratio, |ad_error| | lambda) "
              f"[{meth:8s}]: {pr:+.3f} (p={pp_:.2g})")

    # ---- stratified: does lambda survive INSIDE w_ratio bins? -------------
    # A partial correlation assumes the control enters linearly. This does not.
    print(f"\n[stratified]  median |ad_error| by lambda tertile, within w_ratio tertile")
    rq = pd.qcut(df["w_ratio"], 3, labels=["r low", "r mid", "r high"])
    lq = pd.qcut(df["lambda"], 3, labels=["lam low", "lam mid", "lam high"])
    tab = df.assign(_r=rq, _l=lq).groupby(["_r", "_l"], observed=True)["ad_error"] \
            .agg(lambda s: np.median(np.abs(s)))
    cnt = df.assign(_r=rq, _l=lq).groupby(["_r", "_l"], observed=True)["ad_error"].size()
    print(f"\n{'':10s}" + "".join(f"{c:>14s}" for c in
                                  ["lam low", "lam mid", "lam high"]))
    for rname in ["r low", "r mid", "r high"]:
        cells = []
        for lname in ["lam low", "lam mid", "lam high"]:
            try:
                cells.append(f"{tab[(rname, lname)]:.4f} (n={cnt[(rname, lname)]})")
            except KeyError:
                cells.append("--")
        print(f"{rname:10s}" + "".join(f"{c:>14s}" for c in cells))
    print("\n  Read ACROSS a row: that is lambda's effect with shape held roughly")
    print("  fixed. Read DOWN a column: that is shape's effect at fixed lambda.")


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"loaded {csv_path}  shape={df.shape}")

    ok = df["is_sc"] & np.isfinite(df["ad_error"]) & np.isfinite(df["w_ratio"])
    sc = df[ok].copy()
    n_cap = int(sc["capped"].sum())
    print(f"superconducting with finite ad_error: {len(sc)} / {len(df)}")
    print(f"of those, hit the Matsubara cap: {n_cap} "
          f"({n_cap / max(len(sc), 1):.1%})")

    if n_cap:
        # capping truncates the Matsubara sum early, which biases Tc_ME LOW.
        # It is not random in lambda, so it can manufacture a trend.
        cap = sc["capped"].astype(bool)
        print(f"  capped rows:   lambda median {sc.loc[cap, 'lambda'].median():.3f}"
              f"   Tc_ME median {sc.loc[cap, 'Tc_ME'].median():.3f} K"
              f"   ad_error median {sc.loc[cap, 'ad_error'].median():+.4f}")
        print(f"  uncapped rows: lambda median {sc.loc[~cap, 'lambda'].median():.3f}"
              f"   Tc_ME median {sc.loc[~cap, 'Tc_ME'].median():.3f} K"
              f"   ad_error median {sc.loc[~cap, 'ad_error'].median():+.4f}")
        rb, pb = stats.pointbiserialr(cap.to_numpy(), sc["lambda"].to_numpy(float))
        print(f"  corr(capped, lambda) = {rb:+.3f} (p={pb:.2g})"
              f"   <- if this is not ~0, capping is a lambda-correlated bias")

    report(sc, "ALL superconducting materials")
    if n_cap:
        report(sc[~sc["capped"].astype(bool)], "UNCAPPED ONLY (clean subset)")

    print("\n" + "=" * 72)
    print("Reminder: ad_error here is log(Tc_ME/Tc_AD) with the solver at the")
    print("CALIBRATED mu* and the closed form at Allen-Dynes's. If those two")
    print("conventions ever drift apart again, every number above is garbage.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    a = ap.parse_args()
    main(a.csv)
