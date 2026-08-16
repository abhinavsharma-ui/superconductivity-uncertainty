"""
A generative distribution over alpha^2 F, FITTED to BETE-NET rather than chosen.

Why this replaces the polytope constructions
--------------------------------------------
The moment-matched set is convex and well posed, but a SPREAD needs a MEASURE
and convexity supplies none. Three defensible measures on the same set span an
order of magnitude and do not order consistently (see polytope_floor.py's
status block). Uniform-on-polytope is the worst offender: in ~77 dimensions,
concentration of measure pins draws near the centroid, so every draw populates
every bin similarly and sparse spectra carry vanishing weight. That is not a
neutral default, it is a strong prior claiming real a2F look like band-limited
noise.

So there is no measure-free floor. The honest move is to make the measure an
explicit, fitted, testable assumption instead of an accident of construction.

The construction
----------------
Write the normalised spectral weight

    g(w) = (2/lambda) a2F(w) / w,        INT g dw = 1

and rescale frequency by w_log, u = w / w_log. Then for EVERY material, by the
definitions of the moments:

    INT g~(u) du       = 1
    INT g~(u) ln u du  = 0            <- this is what w_log means
    INT g~(u) u^2 du   = r^2,  r = w_2 / w_log

So lambda and w_log are pure scale factors: lambda sets amplitude, w_log sets
the frequency unit. ALL the shape information lives in g~ at fixed r. The 806
BETE-NET spectra are therefore 806 samples from the conditional shape
distribution p(g~ | r) -- which is exactly the object a floor needs and the
polytope could not supply.

To generate at a target (lambda*, w_log*, w_2*), take a real donor shape and
move it onto the target constraints by EXPONENTIAL TILTING:

    g~_new(u)  proportional to  g~_donor(u) * u^b * exp(c u^2)

solving (b, c) so the two shape constraints hold exactly. This is the
I-projection of the donor onto the constraint set: of all densities satisfying
the constraints, it is the one minimising KL divergence from the donor. So it
is the minimal deformation, not an arbitrary one -- and it preserves
non-negativity automatically, which a least-squares projection would not.

What this does and does not claim
---------------------------------
It does NOT restore "a floor computed with zero reference to the training
data". That claim is dead; see polytope_floor.py.

It DOES preserve the identification argument in a precise weaker form. The
floor is a spread at FIXED moments, so it depends only on the CONDITIONAL
p(g~ | r) and never on the marginal density of materials over
(lambda, w_log, w_2). Data sparsity lives in that marginal. So a floor built
this way still cannot be a sparsity artifact, even though it is not data-free.
That distinction is the whole defence and belongs in the paper explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eliashberg import a2f_moments, allen_dynes_tc, eliashberg_tc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "bete_database.json")
RESULTS = os.path.join(ROOT, "results")

MU_STAR_AD = 0.10
MU_STAR_ME = 0.1293
CUTOFF_FACTOR = 10.0
MAX_MATSUBARA = 250_000
SOLVER_FLOOR_K = 0.005
FLOOR_FRAC = 0.05

# common scaled-frequency grid, u = w / w_log. Log-spaced: real a2F features
# have widths roughly proportional to frequency (median relative sigma 0.0399,
# see polytope_floor.measure_linewidths), so constant RELATIVE resolution is
# the matched choice.
U_GRID = np.geomspace(1e-2, 40.0, 800)
TILT_TOL = 1e-6


def load_shapes(max_materials: int | None = None) -> pd.DataFrame:
    """Every BETE-NET spectrum as a scale-free shape g~(u) plus its r."""
    with open(RAW) as fh:
        db = json.load(fh)
    keys = list(db["a2F"].keys())
    if max_materials:
        keys = keys[:max_materials]

    rows = []
    for k in keys:
        w = np.asarray(db["Freq_meV"][k], float)
        a = np.asarray(db["a2F"][k], float)
        m = a2f_moments(w, a)
        if not np.isfinite(m["lambda_"]) or m["lambda_"] <= 0:
            continue
        w_log, w_2 = m["w_log"], m["w_2"]
        if not (np.isfinite(w_log) and w_log > 0 and np.isfinite(w_2)):
            continue
        pos = w > 1e-9
        g = (2.0 / m["lambda_"]) * a[pos] / w[pos]          # density in w
        u = w[pos] / w_log
        gt = np.interp(U_GRID, u, g * w_log, left=0.0, right=0.0)
        area = np.trapezoid(gt, U_GRID)
        if area <= 0:
            continue
        rows.append({"key": k, "formula": db["comp"][k],
                     "lambda": m["lambda_"], "w_log": w_log, "w_2": w_2,
                     "r": w_2 / w_log, "g": gt / area})
    return pd.DataFrame(rows)


def tilt_to(g_donor: np.ndarray, r_target: float):
    """
    I-projection of a donor shape onto  INT g ln u = 0,  INT g u^2 = r^2.

    g_new  proportional to  g_donor * u^b * exp(c u^2), with (b, c) solved.
    Minimum KL from the donor among all densities meeting the constraints, so
    the deformation is as small as the constraints allow. Returns None if the
    solve does not converge to tolerance -- discarded, never fudged.
    """
    lnu, u2 = np.log(U_GRID), U_GRID ** 2
    tgt = r_target ** 2

    def resid(p):
        b, c = p
        # subtract the max for numerical stability before exponentiating
        e = b * lnu + c * u2
        gg = g_donor * np.exp(e - e.max())
        z = np.trapezoid(gg, U_GRID)
        if not np.isfinite(z) or z <= 0:
            return [1e3, 1e3]
        gg = gg / z
        return [np.trapezoid(gg * lnu, U_GRID),
                np.trapezoid(gg * u2, U_GRID) / tgt - 1.0]

    sol = least_squares(resid, [0.0, 0.0], xtol=1e-14, ftol=1e-14)
    if np.max(np.abs(resid(sol.x))) > TILT_TOL:
        return None
    b, c = sol.x
    e = b * lnu + c * u2
    gg = g_donor * np.exp(e - e.max())
    return gg / np.trapezoid(gg, U_GRID)


def floor_at(lam: float, w_log: float, w_2: float, shapes: pd.DataFrame,
             n_donors: int = 40, seed: int = 0, verbose: bool = True):
    """
    Spread of ln Tc at fixed (lambda, w_log, w_2) under the EMPIRICAL shape
    distribution: real donor shapes, I-projected onto the target moments.
    """
    rng = np.random.default_rng(seed)
    r_t = w_2 / w_log
    idx = rng.permutation(len(shapes))[:n_donors]
    tc_ad = allen_dynes_tc(lam, w_log, w_2, MU_STAR_AD)
    t_floor = max(SOLVER_FLOOR_K, FLOOR_FRAC * tc_ad)

    tcs, mom_err, n_fail = [], [], 0
    for i in idx:
        g_new = tilt_to(shapes["g"].iloc[i], r_t)
        if g_new is None:
            n_fail += 1
            continue
        w = U_GRID * w_log
        a2f = 0.5 * lam * w * (g_new / w_log)
        m = a2f_moments(w, a2f)
        err = max(abs(m["lambda_"] / lam - 1), abs(m["w_log"] / w_log - 1),
                  abs(m["w_2"] / w_2 - 1))
        if err > 5e-3:
            n_fail += 1
            continue
        tc = eliashberg_tc(w, a2f, MU_STAR_ME, cutoff_factor=CUTOFF_FACTOR,
                           t_guess=tc_ad, t_floor=t_floor,
                           max_matsubara=MAX_MATSUBARA, tol=1e-4)
        if np.isfinite(tc) and tc > 0:
            tcs.append(tc)
            mom_err.append(err)
    if len(tcs) < 5:
        return None

    tcs = np.array(tcs)
    out = {
        "lambda": lam, "w_log": w_log, "w_2": w_2, "r": r_t, "Tc_AD": tc_ad,
        "n_donors_used": len(tcs), "n_failed": n_fail,
        "Tc_min": float(tcs.min()), "Tc_max": float(tcs.max()),
        "spread_lnTc_empirical": float(np.std(np.log(tcs), ddof=1)),
        "range_lnTc": float(np.log(tcs.max() / tcs.min())),
        "max_moment_err": float(np.max(mom_err)),
        "measure": "empirical BETE-NET shapes, I-projected onto target moments",
    }
    if verbose:
        print(f"  lam={lam:5.2f} r={r_t:5.3f}  n={len(tcs):3d} "
              f"(failed {n_fail:2d})  Tc [{tcs.min():8.3f},{tcs.max():8.3f}]  "
              f"spread={out['spread_lnTc_empirical']:.4f}  "
              f"(moment err {out['max_moment_err']:.1e})", flush=True)
    return out


def main(n_donors: int = 40, max_materials: int | None = None):
    shapes = load_shapes(max_materials)
    print(f"loaded {len(shapes)} shapes;  r range "
          f"[{shapes.r.min():.3f}, {shapes.r.max():.3f}]  "
          f"median {shapes.r.median():.3f}\n")

    # the four cells the other measures were quoted on, so the comparison is
    # like-for-like rather than on a fresh set of targets
    CELLS = [
        ("CrRh3", 0.4689681433981032, 21.07425730946907, 22.265546945920107),
        ("CoTi", 1.012085030346337, 14.09374594703156, 16.361237861756496),
        ("AsZr", 0.8591682309359172, 12.014062145949564, 14.575283369318766),
        ("Se2V", 1.24959280990693, 8.821060672713381, 13.77356774582734),
    ]
    rows = []
    for name, lam, wl, w2 in CELLS:
        print(f"{name}:")
        r = floor_at(lam, wl, w2, shapes, n_donors=n_donors)
        if r:
            r["target"] = name
            rows.append(r)
    if not rows:
        print("no cell produced a usable ensemble")
        return
    out = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    dest = os.path.join(RESULTS, "empirical_floor.csv")
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    print("\nThis is a spread under a STATED, FITTED measure -- not 'the floor'.")
    print("Compare against the vertex / two-Gaussian / uniform numbers as a")
    print("sensitivity envelope; they bracket deliberately extreme priors.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-donors", type=int, default=40)
    ap.add_argument("--max-materials", type=int, default=None)
    a = ap.parse_args()
    main(n_donors=a.n_donors, max_materials=a.max_materials)
