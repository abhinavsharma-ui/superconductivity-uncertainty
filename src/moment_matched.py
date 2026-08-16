"""
The moment-matched spectral experiment.

THE POINT
---------
Allen-Dynes sees exactly three numbers: lambda, w_log, w_2. It therefore
returns exactly ONE Tc for every alpha^2 F sharing those three moments.

But alpha^2 F is a function, not three numbers. Many genuinely different
spectra share the same three moments -- and they do NOT have the same true Tc.

So: construct families of spectra with IDENTICAL (lambda, w_log, w_2) but
different shape, and solve the Eliashberg equations for each. The spread in
Tc across a family is the irreducible ("aleatoric") uncertainty floor for any
model given only the Allen-Dynes inputs. No model, however good, can predict
Tc more precisely than this from those three features, because the features
genuinely do not contain the answer.

WHY THIS IS THE LOAD-BEARING EXPERIMENT
---------------------------------------
This floor is computed WITHOUT REFERENCE TO ANY TRAINING DATA. It is pure
physics: solve the equations, measure the spread. It therefore predicts where
predictive uncertainty *should* be large in a way that cannot possibly be a
data-sparsity artifact.

That is a stronger identification strategy than any statistical adjustment
for local density, because there is no data involved to be sparse.

If the calibrated model's uncertainty tracks this floor -> the physics
interpretation is established by construction.
If it tracks local data density instead -> the effect is a sparsity artifact,
and the paper says so.

CONSTRUCTION
------------
Write the normalised spectral weight

    g(w) = (2/lambda) * alpha^2F(w) / w,      INT g dw = 1

Then, by construction:

    lambda  is set purely by the overall amplitude   -> matched exactly
    w_log   = exp( INT g ln w dw )                   -> a constraint on shape
    w_2     = sqrt( INT g w^2 dw )                   -> a constraint on shape

So the problem reduces to: generate probability densities g with prescribed
<ln w> and <w^2>. We use a two-Gaussian mixture and sweep the mixture weight,
solving for the two centres at each step. Small weight -> essentially one
phonon branch; large weight -> two well-separated branches. Physically that is
the difference between a simple metal and one with a soft mode plus a hard
optical branch, at identical Allen-Dynes inputs.

alpha^2F is then recovered as  alpha^2F(w) = (lambda/2) * w * g(w).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eliashberg import (  # noqa: E402
    a2f_moments, allen_dynes_tc, eliashberg_tc,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")

MU_STAR = 0.10
CUTOFF_FACTOR = 10.0
N_GRID = 1200


def _grid(w_hi: float) -> np.ndarray:
    return np.linspace(1e-3, w_hi, N_GRID)


def _mixture(w: np.ndarray, weight: float, m1: float, m2: float,
             s1: float, s2: float) -> np.ndarray:
    """Two-Gaussian mixture on w>0, renormalised to unit integral."""
    g = ((1.0 - weight) * np.exp(-0.5 * ((w - m1) / s1) ** 2) / s1
         + weight * np.exp(-0.5 * ((w - m2) / s2) ** 2) / s2)
    g = np.clip(g, 0.0, None)
    area = np.trapezoid(g, w)
    if area <= 0:
        return None
    return g / area


def _moments_of_g(w: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    w_log = float(np.exp(np.trapezoid(g * np.log(w), w)))
    w_2 = float(np.sqrt(np.trapezoid(g * w ** 2, w)))
    return w_log, w_2


def build_family(lam: float, w_log: float, w_2: float,
                 weights=(0.02, 0.10, 0.20, 0.30, 0.40, 0.50),
                 width_frac: float = 0.18, tol: float = 2e-3) -> list[dict]:
    """
    Family of alpha^2F all sharing (lam, w_log, w_2), indexed by mixture weight.

    For each mixture weight we solve for the two Gaussian centres (m1, m2) that
    reproduce the target w_log and w_2. Members that do not converge to within
    `tol` relative error on both moments are discarded rather than fudged.
    """
    w_hi = max(6.0 * w_2, 4.0 * w_log)
    w = _grid(w_hi)
    out = []

    for weight in weights:
        def resid(p):
            m1, m2 = np.exp(p)          # keep centres positive
            s1, s2 = width_frac * m1, width_frac * m2
            g = _mixture(w, weight, m1, m2, s1, s2)
            if g is None:
                return [1e3, 1e3]
            wl, w2 = _moments_of_g(w, g)
            return [np.log(wl / w_log), np.log(w2 / w_2)]

        # seed: one branch near w_log, a harder branch above it
        seed = np.log([max(w_log * 0.85, 1e-3), max(w_2 * 1.35, w_log * 1.5)])
        sol = least_squares(resid, seed, xtol=1e-12, ftol=1e-12)
        m1, m2 = np.exp(sol.x)
        s1, s2 = width_frac * m1, width_frac * m2
        g = _mixture(w, weight, m1, m2, s1, s2)
        if g is None:
            continue
        wl, w2 = _moments_of_g(w, g)
        if abs(wl / w_log - 1) > tol or abs(w2 / w_2 - 1) > tol:
            continue

        a2f = 0.5 * lam * w * g          # alpha^2F = (lambda/2) w g(w)
        chk = a2f_moments(w, a2f)
        out.append({
            "weight": weight, "m1": m1, "m2": m2,
            "omega": w, "a2f": a2f,
            "lambda_chk": chk["lambda_"], "w_log_chk": chk["w_log"],
            "w_2_chk": chk["w_2"],
        })
    return out


def run_target(name: str, lam: float, w_log: float, w_2: float,
               verbose: bool = True) -> dict | None:
    fam = build_family(lam, w_log, w_2)
    if len(fam) < 3:
        if verbose:
            print(f"  {name}: only {len(fam)} family members converged, skipped")
        return None

    # Allen-Dynes sees only the three moments -> ONE number for the whole family
    tc_ad = allen_dynes_tc(lam, w_log, w_2, MU_STAR)

    tcs = []
    for mem in fam:
        tc = eliashberg_tc(mem["omega"], mem["a2f"], MU_STAR,
                           cutoff_factor=CUTOFF_FACTOR, t_guess=tc_ad)
        mem["Tc_ME"] = tc
        if np.isfinite(tc) and tc > 0:
            tcs.append(tc)

    if len(tcs) < 3:
        return None
    tcs = np.array(tcs)

    # worst-case moment error actually achieved across the family
    max_mom_err = max(
        max(abs(m["lambda_chk"] / lam - 1), abs(m["w_log_chk"] / w_log - 1),
            abs(m["w_2_chk"] / w_2 - 1)) for m in fam)

    res = {
        "target": name, "lambda": lam, "w_log": w_log, "w_2": w_2,
        "n_members": len(tcs), "Tc_AD": tc_ad,
        "Tc_ME_min": tcs.min(), "Tc_ME_max": tcs.max(),
        "Tc_ME_median": float(np.median(tcs)),
        # the irreducible floor, as a fractional spread in Tc
        "sigma_log_Tc": float(np.std(np.log(tcs))),
        "spread_ratio": float(tcs.max() / tcs.min()),
        "max_moment_err": max_mom_err,
    }
    if verbose:
        print(f"  {name:14s} lam={lam:5.2f}  Tc_AD={tc_ad:7.2f}K   "
              f"Tc_ME in [{tcs.min():7.2f}, {tcs.max():7.2f}]K   "
              f"ratio={res['spread_ratio']:5.2f}x   "
              f"sigma_lnTc={res['sigma_log_Tc']:.3f}   "
              f"(moment err {max_mom_err:.1e})", flush=True)
    return res


def main(n_targets: int = 12):
    path = os.path.join(PROC, "physics_dataset.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df.get("is_sc", True) & df["lambda"].notna()]
        df = df.sort_values("lambda")
        # spread targets evenly across the lambda range actually present
        idx = np.linspace(0, len(df) - 1, n_targets).astype(int)
        targets = [(r.formula, r["lambda"], r.w_log, r.w_2)
                   for _, r in df.iloc[idx].iterrows()]
        print(f"targets drawn from {path} ({len(df)} superconducting rows)")
    else:
        print("physics_dataset.csv not found -- using synthetic targets")
        targets = [(f"synth_lam{l:.2f}", l, 15.0, 18.0)
                   for l in (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0)]

    print("\nAllen-Dynes returns ONE Tc per row. Eliashberg returns a RANGE.")
    print("That range is the irreducible uncertainty floor.\n")

    rows = [r for r in (run_target(*t) for t in targets) if r]
    if not rows:
        print("no targets produced a usable family")
        return

    out = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    dest = os.path.join(RESULTS, "moment_matched.csv")
    out.to_csv(dest, index=False)

    print(f"\nwrote {dest}")
    print(f"\nirreducible floor sigma(ln Tc): "
          f"median={out.sigma_log_Tc.median():.4f}  "
          f"min={out.sigma_log_Tc.min():.4f}  max={out.sigma_log_Tc.max():.4f}")
    print(f"i.e. a {100 * out.sigma_log_Tc.median():.1f}% typical irreducible "
          f"scatter in Tc at fixed Allen-Dynes inputs")

    if len(out) > 3:
        r = np.corrcoef(out["lambda"], out.sigma_log_Tc)[0, 1]
        print(f"\ncorr(lambda, irreducible floor) = {r:+.3f}")
        print("  positive -> the floor itself grows with coupling, which is a")
        print("  DATA-FREE prediction that uncertainty should rise with lambda")

    print("\nsanity: max moment-matching error across all families = "
          f"{out.max_moment_err.max():.2e} (must be << the spread)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=12)
    a = ap.parse_args()
    main(n_targets=a.n_targets)
