"""
The moment-matched set is a CONVEX POLYTOPE. Search its extreme points instead
of sweeping a chosen spectral family.

The observation
---------------
All three Allen-Dynes moments are LINEAR functionals of alpha^2 F:

    lambda              = 2 INT a2F(w) / w  dw
    lambda * w_2^2      = 2 INT a2F(w) * w  dw
    lambda * ln w_log   = 2 INT a2F(w) * ln(w)/w  dw

So {a2F >= 0 : the three moments take prescribed values} is the intersection of
three hyperplanes with the non-negative cone -- a convex polytope. This is a
classical truncated moment problem, and it carries a theorem: a basic feasible
solution of a linear system with 3 equality constraints has at most 3 nonzero
components. The extreme points of the moment-matched set are spectra supported
on AT MOST 3 frequencies.

Why that matters
----------------
`moment_matched.py` picks a two-Gaussian family and sweeps the mixture weight.
That is one curve through the polytope, chosen by hand. Its spread is a lower
bound on the spread across the set, by an unknown factor -- and "is the family
rich enough?" has no answer, because you can always widen it further.

Parameterising by 3 support points removes the question. Six parameters
(3 positions, 3 weights) minus 3 constraints leaves 3 free dimensions, which is
small enough to search directly. Positions are chosen; the weights follow from
solving the 3x3 linear system; triples whose solution has any negative weight
lie outside the polytope and are discarded.

MEASURED against the existing two-Gaussian family, same targets, same mu*_ME:

    Ge2Mo6  lam=0.447  r=1.065   range(ln Tc)  0.0059 -> 0.0193   3.28x
    CW2     lam=0.653  r=1.595   range(ln Tc)  0.0666 -> 0.1427   2.14x

so the two-Gaussian floor understates by 2-3x. Both numbers are still LOWER
bounds: widening the position grid from [0.15 w_log, 2.2 w_2] to
[0.08 w_log, 3.0 w_2] took the feasible count from 21 to 38 and 44 to 132 and
kept growing the range. Enumeration should eventually be replaced by explicit
optimisation over the 3 free dimensions.

TWO DIFFERENT QUANTITIES, both needed
-------------------------------------
sigma(ln Tc) over a family is NOT intrinsic -- it depends on the family and on
how the family is sampled. The polytope has two well-defined statistics and the
paper needs to say which it reports:

  RANGE over the polytope -- a supremum. This is the "no model can do better
      than this from three moments" claim. Well defined, no sampling measure
      required, and what this script computes.

  SPREAD under a prior over physically realisable spectra -- this is what a
      calibrated model's aleatoric uncertainty should converge to, because the
      model sees real materials, not polytope vertices.

Reporting one and calling it the other is the easiest attack on the result.

Physicality caveat
------------------
WIDTH = 0.03 makes near-delta spectra, which are not literal phonon spectra.
The existing width-convergence check (width_frac 0.30 -> 0.04 converging rather
than diverging) is evidence the limit is well behaved. Report the delta limit
as the rigorous supremum and a physically smoothed version as the realisable
value, and bracket the floor between them.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eliashberg import (  # noqa: E402
    a2f_moments, allen_dynes_tc, eliashberg_tc, matsubara_capped,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed", "physics_dataset.csv")
RESULTS = os.path.join(ROOT, "results")

MU_STAR_AD = 0.10
MU_STAR_ME = 0.1293
CUTOFF_FACTOR = 10.0
MAX_MATSUBARA = 250_000

# Two cost bounds, both learned the expensive way.
#
# GRID_PAD: the frequency grid must extend past the highest support point, but
# no further. w_c = CUTOFF_FACTOR * w_max, and n_cut ~ w_c / T, so padding the
# grid with empty space above the spectrum inflates the Matsubara count for
# nothing. An earlier version ran the grid to 8*w_2 while the support topped out
# at 3*w_2, costing a needless ~2.7x in w_c -- and at the lowest-lambda targets
# that pushed n_cut past MAX_MATSUBARA, so the solves were CAPPED as well as
# slow. Capped values are biased low and lambda-correlated; see build_physics_
# dataset.py. Hence the explicit check in scan_target rather than trust.
#
# FLOOR_FRAC: t_floor as a fraction of Tc_AD. Every member of a family shares
# the target's moments, so its Tc sits within a factor of ~1.2 of Tc_AD -- the
# largest range(ln Tc) measured so far is 0.15. A floor 20x below Tc_AD cannot
# touch the answer, and since n_cut ~ 1/T it bounds the cost of the bracket
# descent. Same fix as in moment_matched.py and diagnose_mustar.py: a cost
# bound on a search, NOT the reporting threshold.
GRID_PAD = 1.3
FLOOR_FRAC = 0.05
SOLVER_FLOOR_K = 0.005
WIDTH = 0.03           # narrow Gaussian standing in for a delta
MOMENT_TOL = 2e-3


def _basis(w: np.ndarray, centre: float, width_frac: float = WIDTH) -> np.ndarray:
    s = width_frac * centre
    g = np.exp(-0.5 * ((w - centre) / s) ** 2)
    return g / np.trapezoid(g, w)


def measure_linewidths(max_materials: int = 250, thresh: float = 0.15) -> dict:
    """
    Characteristic RELATIVE feature width in real alpha^2 F, MEASURED from
    BETE-NET rather than chosen.

    WIDTH = 0.03 makes near-delta basis functions. That is the correct choice
    for the SUPREMUM -- polytope vertices are genuine deltas, and the supremum
    is a statement about what the moment set permits, not about what nature
    builds. But it is the wrong choice for the other statistic: what a
    calibrated model's aleatoric uncertainty converges to is the spread over
    spectra it could actually MEET, and those have real phonon linewidths.
    Reporting one as the other is the easiest attack on the result, so the
    physical width is measured here instead of picked.

    Method: for each material, locate local maxima of a2F above `thresh` of the
    global max; measure each peak's full width at half maximum by walking out
    to the half-height crossings; convert to the equivalent Gaussian relative
    sigma, FWHM / (2.355 * centre). Report the distribution over all peaks in
    all materials -- the median is what the prior uses.
    """
    import json
    raw = os.path.join(ROOT, "data", "raw", "bete_database.json")
    with open(raw) as fh:
        db = json.load(fh)
    rel = []
    for k in list(db["a2F"].keys())[:max_materials]:
        w = np.asarray(db["Freq_meV"][k], float)
        a = np.asarray(db["a2F"][k], float)
        if a.size < 5 or not np.any(a > 0):
            continue
        cut = thresh * a.max()
        for i in range(1, a.size - 1):
            if a[i] <= cut or not (a[i] >= a[i - 1] and a[i] >= a[i + 1]):
                continue
            half = 0.5 * a[i]
            L = i
            while L > 0 and a[L] > half:
                L -= 1
            R = i
            while R < a.size - 1 and a[R] > half:
                R += 1
            fwhm = w[R] - w[L]
            if w[i] > 1e-9 and fwhm > 0:
                rel.append(fwhm / (2.355 * w[i]))
    rel = np.asarray(rel)
    return {"n_peaks": int(rel.size), "median": float(np.median(rel)),
            "q25": float(np.percentile(rel, 25)),
            "q75": float(np.percentile(rel, 75))}


def build_3support(lam: float, w_log: float, w_2: float, centres, w: np.ndarray,
                   width: float = WIDTH):
    """
    a2F = sum_i c_i g_i with all three moments matched exactly (to quadrature).

    Returns None if the solution leaves the polytope (any c_i < 0) or if the
    achieved moments miss by more than MOMENT_TOL -- discarded, never fudged.
    """
    G = [_basis(w, c, width) for c in centres]
    A = np.array([
        [2 * np.trapezoid(g / w, w) for g in G],
        [2 * np.trapezoid(g * w, w) for g in G],
        [2 * np.trapezoid(g * np.log(w) / w, w) for g in G],
    ])
    b = np.array([lam, lam * w_2 ** 2, lam * np.log(w_log)])
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    if np.any(c < 0):
        return None
    a2f = sum(ci * gi for ci, gi in zip(c, G))
    m = a2f_moments(w, a2f)
    if (abs(m["lambda_"] / lam - 1) > MOMENT_TOL
            or abs(m["w_log"] / w_log - 1) > MOMENT_TOL
            or abs(m["w_2"] / w_2 - 1) > MOMENT_TOL):
        return None
    return a2f


def scan_target(name: str, lam: float, w_log: float, w_2: float,
                n_grid: int = 16, lo: float = 0.08, hi: float = 3.0,
                verbose: bool = True, width: float = WIDTH) -> dict | None:
    """Enumerate 3-support spectra over a log grid of positions, solve each."""
    tc_ad = allen_dynes_tc(lam, w_log, w_2, MU_STAR_AD)
    grid = np.exp(np.linspace(np.log(lo * w_log), np.log(hi * w_2), n_grid))
    # grid extent from the actual support, not from a fixed multiple of w_2
    w = np.linspace(1e-3, GRID_PAD * grid[-1], 2000)
    t_floor = max(SOLVER_FLOOR_K, FLOOR_FRAC * tc_ad)

    tcs, where, capped = [], [], 0
    for centres in itertools.combinations(grid, 3):
        a2f = build_3support(lam, w_log, w_2, centres, w, width)
        if a2f is None:
            continue
        tc = eliashberg_tc(w, a2f, MU_STAR_ME, cutoff_factor=CUTOFF_FACTOR,
                           t_guess=tc_ad, t_floor=t_floor,
                           max_matsubara=MAX_MATSUBARA)
        if not (np.isfinite(tc) and tc > 0):
            continue
        # a capped Tc is biased low and lambda-correlated -- never report it
        if matsubara_capped(tc, float(w.max()), CUTOFF_FACTOR, MAX_MATSUBARA):
            capped += 1
            continue
        if tc < 2.0 * t_floor:      # floor was supposed to be unreachable
            print(f"  !! {name}: Tc={tc:.5g} near t_floor={t_floor:.5g}; "
                  f"FLOOR_FRAC too high", flush=True)
        tcs.append(tc)
        where.append(centres)
    if capped:
        print(f"  !! {name}: {capped} spectra hit the Matsubara cap, discarded",
              flush=True)
    if len(tcs) < 3:
        return None

    tcs = np.array(tcs)
    lo_i, hi_i = int(np.argmin(tcs)), int(np.argmax(tcs))
    res = {
        "target": name, "lambda": lam, "w_log": w_log, "w_2": w_2,
        "w_ratio": w_2 / w_log, "Tc_AD": tc_ad, "n_feasible": len(tcs),
        "Tc_min": float(tcs.min()), "Tc_max": float(tcs.max()),
        # the supremum statistic: how far apart Tc can be at identical moments
        "range_lnTc": float(np.log(tcs.max() / tcs.min())),
        # THE OTHER STATISTIC, and it is not interchangeable with the one above.
        # A range is a supremum over the polytope: no sampling measure, no
        # comparison to any model's RMSE. This is a SPREAD, which requires a
        # measure and therefore carries one -- recorded, not implied.
        "spread_lnTc": float(np.std(np.log(tcs), ddof=1)),
        "basis_width": width,
        "measure": "uniform over feasible 3-support triples on the position grid",
        "n_capped_discarded": capped,
        "t_floor_used": t_floor,
        "argmin_w": np.round(where[lo_i], 3).tolist(),
        "argmax_w": np.round(where[hi_i], 3).tolist(),
    }
    if verbose:
        print(f"  {name:12s} lam={lam:5.2f} r={res['w_ratio']:5.3f}  "
              f"n_feas={len(tcs):4d}  Tc [{tcs.min():8.4f},{tcs.max():8.4f}]  "
              f"range(lnTc)={res['range_lnTc']:.4f}", flush=True)
        print(f"    min at w={res['argmin_w']}   max at w={res['argmax_w']}",
              flush=True)
    return res


def main(n_targets: int = 8, n_grid: int = 16):
    df = pd.read_csv(DATA)
    df = df[df["is_sc"] & df["lambda"].notna()].sort_values("lambda")
    idx = np.linspace(0, len(df) - 1, n_targets).astype(int)

    print("range(ln Tc) over 3-support spectra at IDENTICAL (lambda, w_log, w_2).")
    print("This is a supremum over the moment-matched polytope, not a spread")
    print("over a chosen family -- see the module docstring.\n")

    rows = []
    for _, r in df.iloc[idx].iterrows():
        out = scan_target(r.formula, r["lambda"], r.w_log, r.w_2, n_grid=n_grid)
        if out:
            rows.append(out)
    if not rows:
        print("no target produced a feasible set")
        return

    out = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    dest = os.path.join(RESULTS, "polytope_floor.csv")
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    print(f"range(ln Tc): median={out.range_lnTc.median():.4f}  "
          f"min={out.range_lnTc.min():.4f}  max={out.range_lnTc.max():.4f}")
    print("\nNOTE: still a lower bound. Enumeration on a finite position grid")
    print("under-samples the polytope boundary; widening the grid kept growing")
    print("the range in every test. Replace with optimisation over the 3 free")
    print("dimensions before quoting these as the supremum.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--n-grid", type=int, default=16)
    a = ap.parse_args()
    main(n_targets=a.n_targets, n_grid=a.n_grid)
