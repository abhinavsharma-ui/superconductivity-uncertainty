"""
STATUS: SUPERSEDED AS A FLOOR ESTIMATE. Retained as a sensitivity bound.
=======================================================================
This file computes a spread of Tc over the moment-matched set. Convexity gives
a well-posed SET; it does not give a canonical MEASURE on it, and a spread
requires one. Measured on the same four cells, three defensible measures span
an order of magnitude and do not even order consistently:

    target      r    vertex   2-gauss   uniform-on-polytope
    CrRh3   1.057   0.0042    0.0012      0.0006
    CoTi    1.161   0.0230    0.0422      0.0028
    AsZr    1.213   0.0236    0.0565      0.0033
    Se2V    1.561   0.0906    0.0182      0.0103

vertex beats two-Gaussian at CrRh3 and loses at CoTi/AsZr. Uniform sampling is
lowest everywhere by ~8x, by concentration of measure: uniform draws on a
77-dimensional polytope cluster near the centroid and populate every bin
similarly, so sparse configurations -- which is exactly what the vertex set and
the two-Gaussian family are built to reach -- carry vanishing weight. Uniform
is not a neutral default; it is a strong prior asserting that real a2F resemble
band-limited noise.

So NO number in this file is "the floor", including the ones it prints. The
RANGE is dropped outright rather than deprioritised: it is a supremum of a
non-convex function over a set that is unbounded in the support direction, so
it has no fixed point and is a function of how far the optimiser is allowed to
roam. Treat range_lnTc as a diagnostic only.

This also retracts the original justification for the experiment: the
floor is NOT "computed with zero reference to the training data". It is defined
only relative to a distribution over spectra.

What survives, and it is the useful part:
  - the SET is genuinely convex and interpolable (walking the segment between
    two extreme vertices holds all three moments to 1e-16 with Tc monotone)
  - the r-dependence is robust across every measure tried
  - these constructions are legitimate SENSITIVITY BOUNDS on a floor defined
    elsewhere, bracketing how far it moves under deliberately extreme priors

The definition now lives in the operational experiment: the residual of a model
trained on unlimited data from an explicitly stated generative distribution
over a2F, fitted to BETE-NET rather than invented. That makes the measure a
declared assumption with a sensitivity analysis instead of an unstated
consequence of a construction choice.

NOT quantisation, before anyone asks: eliashberg_tc defaults to tol=2e-3, which
quantises Tc at ~0.2% ~ 0.002 in ln Tc, against CrRh3's spread of 0.0042 -- two
tolerance units, and the obvious attack on the r -> 1 collapse. Re-run at
tol=1e-5 every spread moves by <= 0.0001 and CrRh3 is unchanged at 0.0042. The
low-r floor is real.

---------------------------------------------------------------------------

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

Parameterising by 3 support points was claimed here to REMOVE that question.
It does not -- it replaces one arbitrary family with a principled one that
still cannot be completed. Six parameters (3 positions, 3 weights) minus 3
constraints leaves 3 free dimensions, small enough to search directly;
positions are chosen, weights follow from the 3x3 solve, and triples with any
negative weight lie outside the polytope and are discarded. That is a clean
construction, but it is still a family, and directly tested it is not even the
widest one: smooth interior spectra matching all three moments to ~1e-9 beat
every enumerated vertex on both AsZr (Tc 8.196-9.435 vs 8.022-8.775) and CoTi
(12.810-14.222 vs 12.593-13.660).

TWO LIMITS ON THE RANGE, both measured, both real
--------------------------------------------------
1. VERTICES ARE NOT ENOUGH, because Tc is not convex in a2F. The theorem
   above is about extreme points OF THE SET. The extremes of a non-convex
   FUNCTION on that set need not be attained at them, and are not: on AsZr the
   hand-picked two-Gaussian family reaches range 0.1408 while a 3-support
   search refined over all 3 free dimensions reaches only 0.1377. A
   two-Gaussian mixture is an interior point and it beats every vertex found.
   So 3-support search is a RESTRICTION, not a completeness result. Refinement
   helps (gains of 1.30-1.45x over enumeration, and it does overtake the
   two-Gaussian value on CoTi and Se2V) but cannot be claimed to converge.

2. THE FEASIBLE SET IS UNBOUNDED IN SUPPORT POSITION. A vanishing weight at
   w -> infinity absorbs the w_2 constraint while costing only O(1/w^2) in
   lambda and in ln w_log, so support can be pushed arbitrarily high at
   negligible cost to the other two moments. Optimisation exploits this
   correctly and immediately: the refined Se2V maximum sits at
   w = [1.07, 15.4, 703] meV, a support point 51x its own w_2. Mathematically
   valid, physically absurd -- no phonon spectrum has a 703 meV mode.

Consequence: the RANGE is a supremum of a non-convex function over an unbounded
set. It has no fixed point under better search and is set by physically absurd
corners. It needs an explicit physicality bound on support before it means
anything. The SPREAD has neither problem: it is a second moment over a bounded,
sampled measure, so it converges under refinement rather than growing. Prefer
the spread; quote the range only with its support bound stated.

MEASURED against the existing two-Gaussian family, same targets, same mu*_ME:

    Ge2Mo6  lam=0.447  r=1.065   range(ln Tc)  0.0059 -> 0.0193   3.28x
    CW2     lam=0.653  r=1.595   range(ln Tc)  0.0666 -> 0.1427   2.14x

so the two-Gaussian floor understates by 2-3x. Both numbers are still LOWER
bounds: widening the position grid from [0.15 w_log, 2.2 w_2] to
[0.08 w_log, 3.0 w_2] took the feasible count from 21 to 38 and 44 to 132 and
kept growing the range. Enumeration should eventually be replaced by explicit
optimisation over the 3 free dimensions.

THREE QUANTITIES, and they are not interchangeable
--------------------------------------------------
sigma(ln Tc) over a family is NOT intrinsic -- it depends on the family and on
how the family is sampled. Name which one is being reported:

  RANGE over the polytope -- a supremum, max minus min. No sampling measure
      required. This is the "no model can do better than this from three
      moments" claim. It is NOT comparable to any model's RMSE.

  SPREAD under a MEASURE ON THE POLYTOPE -- what `spread_lnTc` computes:
      uniform over the enumerated 3-support triples. A dispersion, so it IS
      comparable to an RMSE, but the measure is an artefact of the enumeration
      grid, not a statement about nature.

  SPREAD under a LINEWIDTH PRIOR over physically realisable spectra -- what a
      calibrated model's aleatoric uncertainty should actually converge to,
      since the model meets real materials rather than polytope vertices. NOT
      YET IMPLEMENTED. Real a2F cluster away from the vertices, so this should
      come out SMALLER than the uniform-on-polytope spread.

The second and third are both "spreads" and are easy to conflate in prose; they
are different numbers. Reporting any of the three as another is the easiest
attack on the result, which is why basis_width and measure are emitted as CSV
columns rather than left to a docstring.

Basis width: measured, and it matters only at low r
---------------------------------------------------
An earlier version of this text proposed bracketing the floor between a
"rigorous but unphysical" delta limit and a "physically smoothed" value. That
framing is wrong and has been removed. measure_linewidths() puts real BETE-NET
a2F feature widths at median 0.0399, IQR [0.0173, 0.1027] over 2279 peaks --
only 1.3x the WIDTH = 0.03 basis. There is no meaningful delta-vs-realisable
gap to bracket; the delta basis is already near-physical. The real bracket is
RANGE vs SPREAD, which is a statistical distinction, not a physical one.

Width does still matter at low r, but HOW MUCH IS UNRESOLVED and the two
measurements disagree. Both sweep the measured IQR (0.0173 / 0.0399 / 0.1027);
both agree on the direction and on which cells are affected:

    r ~ 1.06   spread at q75 vs median:   -24% (config A)   -12% (config B)
    r ~ 1.21                                      --        -6%  (config B)
    r ~ 1.16                              -5%  (config A)          --
    r ~ 1.56                              -2%  (config A)   -2%  (config B)

Config A: synthetic targets at w_log = 40 meV. Config B: real-material targets
on the committed grid. The r ~ 1.06 cell is CrRh3 in both, so the factor-of-two
disagreement is not target choice and is not yet explained. DO NOT cite a
magnitude until it is; cite only the direction, which both configs agree on.

The mechanism is not in doubt: as r -> 1 the w_log and w_2 constraints nearly
coincide, the polytope approaches degeneracy, and a broad basis smears
constraint satisfaction enough to destroy feasible triples (feasible counts
drop, they do not merely reweight). So the near-Einstein collapse value is
width-dependent and must always be quoted with its width.

Direction matters for the headline result and is robust: low-r cells shrink
more than high-r ones under a broader basis, so a physical width STEEPENS the
r-dependence. The measured shape effect is conservative under this choice
whichever magnitude turns out to be right.
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


def _tc_at(lam, w_log, w_2, centres, width, tc_ad, t_floor):
    """Tc for one 3-support spectrum, or None if it leaves the polytope."""
    centres = np.sort(np.asarray(centres, float))
    if centres[0] <= 0 or not np.all(np.diff(centres) > 1e-9):
        return None
    w = np.linspace(1e-3, GRID_PAD * centres[-1], 2000)
    a2f = build_3support(lam, w_log, w_2, centres, w, width)
    if a2f is None:
        return None
    tc = eliashberg_tc(w, a2f, MU_STAR_ME, cutoff_factor=CUTOFF_FACTOR,
                       t_guess=tc_ad, t_floor=t_floor,
                       max_matsubara=MAX_MATSUBARA)
    if not (np.isfinite(tc) and tc > 0):
        return None
    if matsubara_capped(tc, float(w.max()), CUTOFF_FACTOR, MAX_MATSUBARA):
        return None
    return float(tc)


def refine_extreme(lam, w_log, w_2, seed_centres, maximise: bool,
                   width: float, tc_ad: float, t_floor: float,
                   maxiter: int = 120):
    """
    Polish one enumeration extremum over the 3 free dimensions.

    Enumeration on a finite position grid can only report positions ON the
    grid, so its extremes are grid-resolution-limited -- which is why a
    hand-picked two-Gaussian family could BEAT the polytope search on CoTi and
    AsZr. That is proof of non-convergence, not a suspicion, and no amount of
    grid refinement fixes it in principle: the extremes lie wherever they lie.

    Seeded from the enumerated extremum rather than searched cold, because
    enumeration has already located the basin and each objective evaluation
    costs a full Eliashberg solve. Nelder-Mead in log-position space, with
    points outside the polytope returned as a large penalty -- the feasible set
    is convex in the WEIGHTS but the map from positions to feasibility is not,
    so the objective is genuinely discontinuous at the boundary and a
    derivative-free method is the honest choice.
    """
    from scipy.optimize import minimize
    sign = -1.0 if maximise else 1.0
    best = {"tc": None, "centres": None}

    def obj(p):
        tc = _tc_at(lam, w_log, w_2, np.exp(p), width, tc_ad, t_floor)
        if tc is None:
            return 1e3                      # outside the polytope
        if best["tc"] is None or (sign * np.log(tc) < sign * np.log(best["tc"])):
            best["tc"], best["centres"] = tc, np.sort(np.exp(p))
        return sign * np.log(tc)

    minimize(obj, np.log(np.sort(seed_centres)), method="Nelder-Mead",
             options=dict(xatol=2e-3, fatol=1e-5, maxiter=maxiter,
                          maxfev=maxiter))
    return best["tc"], best["centres"]


def scan_target(name: str, lam: float, w_log: float, w_2: float,
                n_grid: int = 16, lo: float = 0.08, hi: float = 3.0,
                verbose: bool = True, width: float = WIDTH,
                refine: bool = False) -> dict | None:
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
        # NOT a linewidth prior. This measure is an artefact of the enumeration
        # grid; a prior weighted by realisable spectra is a different (smaller)
        # number and is not yet implemented. Do not relabel this as one.
        "measure": "uniform over enumerated 3-support triples (polytope measure)",
        "n_capped_discarded": capped,
        "t_floor_used": t_floor,
        "refined": False,
        "argmin_w": np.round(where[lo_i], 3).tolist(),
        "argmax_w": np.round(where[hi_i], 3).tolist(),
    }
    if refine:
        # the range is a MAXIMUM: it only ever grows with better sampling, and
        # has no fixed point under refinement. The spread is a second moment
        # over the feasible set and does converge -- so refinement is reported
        # for the range, and the spread is left on the enumerated measure,
        # which is the one it is defined against.
        tc_hi, c_hi = refine_extreme(lam, w_log, w_2, where[hi_i], True,
                                     width, tc_ad, t_floor)
        tc_lo, c_lo = refine_extreme(lam, w_log, w_2, where[lo_i], False,
                                     width, tc_ad, t_floor)
        tc_hi = max(tc_hi or tcs.max(), tcs.max())
        tc_lo = min(tc_lo or tcs.min(), tcs.min())
        res["range_lnTc_enumerated"] = res["range_lnTc"]
        res["range_lnTc"] = float(np.log(tc_hi / tc_lo))
        res["Tc_min"], res["Tc_max"] = tc_lo, tc_hi
        res["refined"] = True
        res["refine_gain"] = (res["range_lnTc"]
                              / max(res["range_lnTc_enumerated"], 1e-12))
        if c_lo is not None:
            res["argmin_w"] = np.round(c_lo, 3).tolist()
        if c_hi is not None:
            res["argmax_w"] = np.round(c_hi, 3).tolist()

    if verbose:
        print(f"  {name:12s} lam={lam:5.2f} r={res['w_ratio']:5.3f}  "
              f"n_feas={len(tcs):4d}  Tc [{res['Tc_min']:8.4f},"
              f"{res['Tc_max']:8.4f}]  range(lnTc)={res['range_lnTc']:.4f}"
              + (f"  ({res['refine_gain']:.2f}x vs enumerated "
                 f"{res['range_lnTc_enumerated']:.4f})" if refine else ""),
              flush=True)
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
