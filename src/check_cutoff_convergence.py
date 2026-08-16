"""
Is cutoff_factor = 10 numerically converged, for the purposes of ad_error?

The question is NOT "does Tc_ME change with the cutoff" -- it does, and it must,
because mu* is defined at a cutoff. The question is whether it changes by a
CONSTANT FACTOR or by a lambda- and shape-DEPENDENT one.

    d = log[ Tc_ME(cf=20) / Tc_ME(cf=10) ]     at FIXED mu*_ME

    d constant                  -> cf=10 is fine. A constant log-offset is
                                   absorbed exactly by the mu*_ME calibration
                                   (that calibration is precisely a constant
                                   shift chosen to match AD in its own
                                   convention), so ad_error is unchanged.
    d varies with lambda/w_ratio -> cf=10 is NOT fine. No single mu*_ME can
                                   absorb a varying factor, so ad_error's
                                   dependence on exactly the two variables the
                                   paper is about is contaminated.

Holding mu* FIXED across the two cutoffs is the point. Recalibrating mu* per
cutoff is only needed to USE cf=20 in production (where ad_error compares
against AD in AD's convention); it is not needed, and would actively obscure
things, when asking whether the SOLVER is converged.

This is a sharper test than looking at whether a cutoff systematic is small
relative to the irreducible residual: that asks whether a systematic matters,
this asks whether it exists and what shape it has.

Usage
-----
    python src/check_cutoff_convergence.py                 # 80 materials
    python src/check_cutoff_convergence.py --n 40 --workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy import stats

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_physics_dataset import (  # noqa: E402
    MAX_MATSUBARA, MU_STAR_ME, SOLVER_FLOOR_K, _be_polite,
)
from eliashberg import eliashberg_tc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "bete_database.json")
CSV = os.path.join(ROOT, "data", "processed", "physics_dataset.csv")

CF_LO, CF_HI = 10.0, 20.0     # defaults only; the real values travel per-item


def _one(item):
    """
    The cutoffs are carried IN THE ITEM, not read from module globals.

    On Windows, Pool workers are spawned and re-import this module, where
    __name__ != "__main__" -- so anything the CLI block rebinds (as an earlier
    version did with CF_LO/CF_HI) is invisible to them and they silently fall
    back to the module defaults. That produced workers solving cf=10/20 while
    the parent believed it had asked for cf=20/40.
    """
    _be_polite()
    key, w_list, a_list, t_guess, cf_lo, cf_hi = item
    w = np.asarray(w_list, float)
    a = np.asarray(a_list, float)
    out = {"material_id": key}
    for cf in (cf_lo, cf_hi):
        out[f"Tc_cf{cf:.0f}"] = eliashberg_tc(
            w, a, mu_star=MU_STAR_ME, cutoff_factor=cf, t_guess=t_guess,
            t_floor=SOLVER_FLOOR_K, max_matsubara=MAX_MATSUBARA)
    return out


def main(n: int, workers: int | None, cf_lo: float = CF_LO,
         cf_hi: float = CF_HI) -> None:
    _be_polite()
    df = pd.read_csv(CSV)
    sc = df[df["is_sc"] & np.isfinite(df["ad_error"])].copy()
    if sc.empty:
        raise SystemExit("no superconducting rows -- is the build finished?")

    # Span lambda AND w_ratio. Quartile-cell sampling is NOT enough: lambda is
    # heavily right-skewed, so the top quartile runs 0.65..2.39 and drawing a
    # few at random from it never reaches the tail (the first run topped out at
    # 1.50 of 2.389). Systematic sampling over the SORTED order guarantees the
    # extremes are present, which is the whole point -- a flat d proves nothing
    # if the sample never visited the corners where d would bend.
    def _systematic(col: str, k: int) -> pd.Index:
        order = sc.sort_values(col).index
        if k >= len(order):
            return order
        pos = np.linspace(0, len(order) - 1, k).round().astype(int)
        return order[np.unique(pos)]

    idx = _systematic("lambda", n // 2).union(_systematic("w_ratio", n // 2))
    # force the four extreme corners in explicitly
    for col in ("lambda", "w_ratio"):
        idx = idx.union(pd.Index([sc[col].idxmin(), sc[col].idxmax()]))
    pick = sc.loc[idx]
    print(f"{len(pick)} materials spanning lambda "
          f"[{pick['lambda'].min():.2f}, {pick['lambda'].max():.2f}] and "
          f"w_ratio [{pick['w_ratio'].min():.2f}, {pick['w_ratio'].max():.2f}]")
    print(f"fixed mu*_ME = {MU_STAR_ME} at BOTH cutoffs "
          f"(cf={cf_lo:.0f} vs cf={cf_hi:.0f})\n")

    with open(RAW) as fh:
        db = json.load(fh)
    items = [(str(r.material_id), db["Freq_meV"][str(r.material_id)],
              db["a2F"][str(r.material_id)], float(r.Tc_ME), cf_lo, cf_hi)
             for r in pick.itertuples()]

    nw = workers or max(1, cpu_count() // 2)
    t0 = time.time()
    rows = []
    with Pool(nw) as pool:
        for j, r in enumerate(pool.imap_unordered(_one, items, chunksize=2)):
            rows.append(r)
            if (j + 1) % 20 == 0:
                print(f"  {j + 1}/{len(items)} ({time.time() - t0:.0f}s)",
                      flush=True)

    res = pick.merge(pd.DataFrame(rows).astype({"material_id": str}),
                     left_on=pick["material_id"].astype(str),
                     right_on="material_id", suffixes=("", "_y"))
    ok = (res[f"Tc_cf{cf_lo:.0f}"] > SOLVER_FLOOR_K) & \
         (res[f"Tc_cf{cf_hi:.0f}"] > SOLVER_FLOOR_K)
    res = res[ok]
    d = np.log(res[f"Tc_cf{cf_hi:.0f}"] / res[f"Tc_cf{cf_lo:.0f}"]).to_numpy()

    print(f"\nn = {len(d)} usable")
    print(f"d = log[Tc(cf{cf_hi:.0f})/Tc(cf{cf_lo:.0f})]  mean {d.mean():+.4f}  "
          f"median {np.median(d):+.4f}  sd {d.std(ddof=1):.4f}")
    print(f"    range [{d.min():+.4f}, {d.max():+.4f}]  "
          f"as a ratio: {np.exp(d.mean()):.4f} mean, "
          f"spread {np.exp(d.max()) - np.exp(d.min()):.4f}")

    for name in ("lambda", "w_ratio"):
        x = res[name].to_numpy(float)
        rp, pp = stats.pearsonr(x, d)
        rs, ps = stats.spearmanr(x, d)
        sl, ic, rv, pv, se = stats.linregress(x, d)
        print(f"\ncorr(d, {name}):  pearson {rp:+.3f} (p={pp:.2g})   "
              f"spearman {rs:+.3f} (p={ps:.2g})")
        print(f"    OLS slope {sl:+.5f} +/- {se:.5f}  "
              f"=> across the observed range, d moves by "
              f"{sl * (x.max() - x.min()):+.4f} "
              f"({np.expm1(abs(sl * (x.max() - x.min()))):.2%} in Tc)")

    # lambda and w_ratio are themselves correlated, so a raw corr(d, w_ratio)
    # partly just reflects corr(d, lambda). Partial them against each other.
    lam = res["lambda"].to_numpy(float)
    rat = res["w_ratio"].to_numpy(float)
    print(f"\ncorr(lambda, w_ratio) in this sample: "
          f"{stats.pearsonr(lam, rat)[0]:+.3f}")
    for a_, b_, an, bn in ((rat, lam, "w_ratio", "lambda"),
                           (lam, rat, "lambda", "w_ratio")):
        rxy = stats.pearsonr(a_, d)[0]
        rxz = stats.pearsonr(a_, b_)[0]
        ryz = stats.pearsonr(b_, d)[0]
        den = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-30))
        pr = (rxy - rxz * ryz) / den
        dof = len(d) - 3
        t = pr * np.sqrt(dof / max(1 - pr ** 2, 1e-30))
        print(f"  partial corr(d, {an} | {bn}): {pr:+.3f} "
              f"(p={2 * stats.t.sf(abs(t), dof):.2g})")

    print("\n" + "-" * 68)
    print("A constant d is absorbed by the mu*_ME calibration and cf=10 stands.")
    print("A d that trends with lambda or w_ratio is NOT absorbable and")
    print("contaminates ad_error in exactly the two variables under study.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cf-lo", type=float, default=CF_LO)
    ap.add_argument("--cf-hi", type=float, default=CF_HI,
                    help="e.g. --cf-lo 20 --cf-hi 40 to ask whether cf=20 is "
                         "itself converged")
    a = ap.parse_args()
    main(a.n, a.workers, cf_lo=a.cf_lo, cf_hi=a.cf_hi)
