"""
Sensitivity-normalised Allen-Dynes error:  dmu*_equiv = ad_error / |dlnTc/dmu*|.

Why this exists
---------------
`ad_error` = log(Tc_ME / Tc_AD) is a log-ratio of two quantities that both sit
in an exponential denominator, lambda - mu*(1 + 0.62 lambda). As lambda falls
toward mu*/(1 - 0.62 mu*) that denominator collapses and BOTH Tc's dive, at
different rates. So |ad_error| MUST grow at low lambda whatever the formula is
doing. A lambda trend in the raw measure is therefore not evidence about
Allen-Dynes; it is a property of the measure.

Dividing by the local sensitivity asks a scale-free question instead: what
shift in mu* would account for this discrepancy? A convention/calibration
offset is lambda-independent by construction; genuine formula error is not.
Same logic as the plateau/drift decomposition in diagnose_mustar.py, applied to
real spectra rather than Einstein ones.

TWO things this gets right that are easy to get wrong
-----------------------------------------------------
1. The sensitivity must be the SOLVER's, dlnTc_ME/dmu*_ME -- not the closed
   form's. The solver is roughly half as mu*-sensitive as Allen-Dynes (|S_AD| /
   |S_ME| ~ 1.9 across this dataset), so normalising by the AD derivative
   overstates the correction by ~2x. This is the same trap already logged in
   the handoff: converting a mu* shift to a Tc shift with AD's sensitivity
   overstates it by 30-50%.

2. No new solver runs are needed. build_physics_dataset.py already computes
   Tc at BOTH mu*_ME = 0.1293 and 0.1840 for every material, so the finite
   difference is sitting in the dataset:

       S_secant = [ln Tc_ME_mu13 - ln Tc_ME] / (0.1840 - 0.1293)

   That secant spans dmu* = 0.055, so it is only the local slope at 0.1293 if
   ln Tc is close to linear in mu* over that interval. IT IS -- for the SOLVER.
   Measured by central difference at 0.1293 +/- 0.005 on 45 materials spanning
   lambda = 0.28 to 2.39 (see bias_check.py):

       B_true = S_secant / S_local  =  0.88 to 1.09,  mildly DECREASING in lambda

   The obvious analytic correction, den(mu_1)/den(mu_2) from the Allen-Dynes
   form, is 1.07 to 1.96 and is WRONG here. AD's exponential denominator
   lambda - mu*(1 + 0.62 lambda) makes ln Tc strongly convex in mu*; the solver
   has no such pole and comes out very slightly CONCAVE. Applying AD's convexity
   over-corrects by 1.24x at high lambda and 1.57x at low lambda -- i.e. exactly
   along the axis under test, and it manufactures a low-lambda pedestal that is
   not in the data. Use B_MEASURED below, or just B = 1; the difference between
   them is within noise.

B_MEASURED is a quadratic in ln(lambda) fitted to those 45 points (rms residual
0.017). Re-measure it with bias_check.py if MU_STAR_ME or the cutoff changes --
it is a property of the solver at a given calibration, not a universal constant.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed", "physics_dataset.csv")

MU_STAR_ME = 0.1293
MU_STAR_ME_ALT = 0.1840
SC_THRESHOLD_K = 0.05

# B_true = S_secant / S_local, quadratic in ln(lambda), fitted to 45 materials
# measured by central difference at mu* = 0.1293 +/- 0.005. rms residual 0.017,
# valid over lambda = 0.28 to 2.39. See bias_check.py.
B_MEASURED = (0.050055, -0.050842, 0.912283)


def partial_corr(x, y, z, method="pearson"):
    """corr(x, y | z), by residualising both on z. Spearman = same on ranks."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = np.asarray(x)[m], np.asarray(y)[m], np.asarray(z)[m]
    if method == "spearman":
        x, y, z = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = int(m.sum())
    t = r * np.sqrt((n - 3) / max(1 - r * r, 1e-300))
    return r, float(2 * stats.t.sf(abs(t), n - 3)), n


def add_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Attach S_secant, bias, S_local, dmu_secant, dmu_local."""
    df = df.copy()
    ok = (df["Tc_ME"] > 0) & (df["Tc_ME_mu13"] > 0) & df["ad_error"].notna()
    df["S_secant"] = np.nan
    df.loc[ok, "S_secant"] = (
        (np.log(df.loc[ok, "Tc_ME_mu13"]) - np.log(df.loc[ok, "Tc_ME"]))
        / (MU_STAR_ME_ALT - MU_STAR_ME)
    )

    lam = df["lambda"].clip(0.28, 2.39)          # fit's range of validity
    df["bias"] = np.polyval(B_MEASURED, np.log(lam))

    df["S_local"] = df["S_secant"] / df["bias"]
    df["dmu_secant"] = df["ad_error"] / df["S_secant"].abs()
    df["dmu_local"] = df["ad_error"] / df["S_local"].abs()
    return df


def _tertile_table(s, col, scale=1.0):
    ql = s["lambda"].quantile([1 / 3, 2 / 3]).values
    qr = s["w_ratio"].quantile([1 / 3, 2 / 3]).values
    lb = np.digitize(s["lambda"], ql)
    rb = np.digitize(s["w_ratio"], qr)
    print(f"    {'':8s}{'lam low':>14}{'lam mid':>14}{'lam high':>14}")
    for i, lab in enumerate(("r low", "r mid", "r high")):
        cells = []
        for j in range(3):
            g = s[(rb == i) & (lb == j)]
            cells.append(f"{scale * g[col].abs().median():10.2f} ({len(g):3d})")
        print(f"    {lab:8s}" + "".join(cells))


def main(threshold: float = SC_THRESHOLD_K, path: str = DATA) -> pd.DataFrame:
    df = add_sensitivity(pd.read_csv(path))
    s = df[(df["Tc_ME"] > threshold) & df["dmu_local"].notna()].copy()
    print(f"n = {len(s)}  (Tc_ME > {threshold} K)   capped rows: "
          f"{int(df['capped'].sum())}\n")

    ql = s["lambda"].quantile([1 / 3, 2 / 3]).values
    lb = np.digitize(s["lambda"], ql)
    print(f"{'lambda tertile':<16}{'med lam':>9}{'|S_ME|':>9}{'bias':>7}"
          f"{'|ad_error|':>12}{'dmu_secant':>12}{'dmu_local':>11}")
    for j, lab in enumerate(("low", "mid", "high")):
        g = s[lb == j]
        print(f"{lab:<16}{g['lambda'].median():9.3f}{g.S_secant.abs().median():9.1f}"
              f"{g.bias.median():7.2f}{g.ad_error.abs().median():12.4f}"
              f"{g.dmu_secant.abs().median():12.5f}{g.dmu_local.abs().median():11.5f}")
    g0, g2 = s[lb == 0], s[lb == 2]
    ratio = lambda c: g0[c].abs().median() / g2[c].abs().median()  # noqa: E731
    print(f"\nlow/high tertile ratio:  |ad_error| {ratio('ad_error'):.2f}x"
          f"   dmu_secant {ratio('dmu_secant'):.2f}x"
          f"   dmu_local {ratio('dmu_local'):.2f}x")
    print("  -> the fraction of the raw lambda trend that is pure amplification")

    print("\npartial correlations (raw measure vs normalised):")
    for target, name in (("ad_error", "|ad_error|"), ("dmu_local", "|dmu_equiv|")):
        print(f"  {name}")
        for meth in ("pearson", "spearman"):
            r, p, _ = partial_corr(s["lambda"], s[target].abs(), s["w_ratio"], meth)
            print(f"    corr(lambda,  . | w_ratio)  {meth:9s}{r:+.3f}  p={p:.1e}")
        for meth in ("pearson", "spearman"):
            r, p, _ = partial_corr(s["w_ratio"], s[target].abs(), s["lambda"], meth)
            print(f"    corr(w_ratio, . | lambda )  {meth:9s}{r:+.3f}  p={p:.1e}")

    print("\nstratified median |ad_error| (raw):")
    _tertile_table(s, "ad_error")
    print("\nstratified median |dmu*_equiv| x 1e3 (normalised, de-biased):")
    _tertile_table(s, "dmu_local", scale=1e3)

    ne = s[s["w_ratio"] < 1.10]
    print(f"\nnear-Einstein subset (r < 1.10, n={len(ne)}): f2 = 1 exactly, so any"
          f"\n  offset here is calibration transfer + f1, NOT shape."
          f"\n  median Tc_ME/Tc_AD = {np.exp(ne.ad_error).median():.4f}"
          f"   implied constant dmu* = {ne.dmu_local.median():+.5f}")

    print("\nreporting-threshold sensitivity (the check the handoff requires):")
    print(f"  {'Tc_ME >':>9}{'n':>6}{'corr(lam,|ad_err|)':>21}"
          f"{'corr(r,|ad_err|)':>19}{'corr(lam,|dmu_eq|)':>21}"
          f"{'corr(r,|dmu_eq|)':>19}")
    for thr in (0.005, 0.01, 0.05, 0.1, 0.5, 1.0):
        t = df[(df["Tc_ME"] > thr) & df["dmu_local"].notna()]
        a, _, n = partial_corr(t["lambda"], t.ad_error.abs(), t["w_ratio"], "spearman")
        b, _, _ = partial_corr(t["w_ratio"], t.ad_error.abs(), t["lambda"], "spearman")
        c, _, _ = partial_corr(t["lambda"], t.dmu_local.abs(), t["w_ratio"], "spearman")
        e, _, _ = partial_corr(t["w_ratio"], t.dmu_local.abs(), t["lambda"], "spearman")
        print(f"  {thr:>9}{n:>6}{a:>21.3f}{b:>19.3f}{c:>21.3f}{e:>19.3f}")
    print("\n  A result that moves with the threshold is a result about the"
          "\n  threshold. Tc-thresholding is lambda-correlated selection.")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=SC_THRESHOLD_K)
    ap.add_argument("--data", type=str, default=DATA)
    a = ap.parse_args()
    main(threshold=a.threshold, path=a.data)
