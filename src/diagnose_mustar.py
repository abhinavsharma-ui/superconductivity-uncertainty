"""
Is the Eliashberg-vs-Allen-Dynes offset a mu* convention mismatch, or real?

Background
----------
mu* has no meaning without the energy cutoff it is defined at:

    mu*(w_c) = mu / (1 + mu ln(E_F / w_c))

so mu* GROWS with the cutoff. We solve the Eliashberg equations with a cutoff
at 10 * w_max and plug in mu* = 0.10. Allen and Dynes fitted their formula
against solutions using their own convention. Same symbol, different physics.

Because Tc depends on mu* through an exponential denominator,
lambda - mu*(1 + 0.62 lambda), a small convention mismatch is amplified, and
amplified MOST at small lambda -- which is exactly the pattern observed
(2-6% disagreement at mu* = 0, growing to ~18% at mu* = 0.1, worst at low
lambda).

The test
--------
For each lambda, find the mu*_eff (at OUR cutoff) that makes the numerical
Eliashberg Tc equal the Allen-Dynes Tc evaluated at mu* = 0.10.

    mu*_eff constant in lambda  ->  pure convention mismatch. Adopt the
                                    implied convention and move on.
    mu*_eff drifts with lambda  ->  the formula is genuinely failing, and
                                    that failure is the paper's subject.

Einstein spectra are used deliberately: a single sharp phonon mode is the
cleanest case, closest to what the formula was fitted against, so any drift
here cannot be blamed on exotic spectral shape.
"""

from __future__ import annotations

import os
import sys

import argparse

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eliashberg import a2f_moments, allen_dynes_tc, eliashberg_tc  # noqa: E402
from verify_eliashberg import einstein_spectrum  # noqa: E402

W_E = 10.0          # meV, sets the scale only; results are scale free
CUTOFF = 10.0       # overridden by --cutoff-factor
MU_REF = 0.10       # overridden by --mu-ref
LAMBDAS = (0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0)


def main():
    print(f"Einstein spectrum, w_E = {W_E} meV, ME cutoff = {CUTOFF} * w_max")
    print(f"Target: Tc_AD evaluated at mu* = {MU_REF}\n")
    print(f"{'lambda':>7} {'Tc_AD(K)':>10} {'Tc_ME(K)':>10} {'ratio':>8} "
          f"{'mu*_eff':>9} {'implied w_c/w_max':>19}")

    rows = []
    for lam in LAMBDAS:
        w, a = einstein_spectrum(W_E, lam)
        m = a2f_moments(w, a)
        tad = allen_dynes_tc(m["lambda_"], m["w_log"], m["w_2"], MU_REF)
        tme = eliashberg_tc(w, a, MU_REF, cutoff_factor=CUTOFF, t_guess=tad)

        # Upper bound on the mu* search. The hard limit is where the AD
        # denominator vanishes, mu = lambda/(1+0.62 lambda), but Tc -> 0 there
        # and the ME solver gets very expensive (N ~ 1/T). We only need to
        # bracket the root, so stop at the mu* that roughly halves Tc_AD --
        # comfortably past the answer, far from the collapse.
        mu_hard = lam / (1.0 + 0.62 * lam)
        mu_hi = mu_hard
        for cand in np.linspace(MU_REF, 0.9 * mu_hard, 25):
            if allen_dynes_tc(lam, m["w_log"], m["w_2"], cand) < 0.4 * tad:
                mu_hi = cand
                break

        def f(mu):
            t = eliashberg_tc(w, a, mu, cutoff_factor=CUTOFF, t_guess=tad,
                              tol=5e-4)
            return (t if t > 0 else 0.0) - tad

        # The mu_hi guess above is keyed to where Tc_AD falls, but the root is
        # where Tc_ME falls to the target -- and Tc_ME sits well ABOVE Tc_AD at
        # low lambda (ratio 2.30 at lambda=0.4, mu_ref=0.13). So the AD-based
        # guess can stop short of the root and hand brentq a same-sign bracket,
        # which is exactly how lambda=0.4 and 0.5 were lost at --mu-ref 0.13.
        # Expand outward until Tc_ME actually crosses the target. t_floor inside
        # eliashberg_tc bounds the cost: once Tc_ME drops under 0.05 K it
        # returns 0 rather than marching toward T -> 0.
        n_exp = 0
        while mu_hi < 0.98 * mu_hard and f(mu_hi) > 0 and n_exp < 12:
            mu_hi = min(mu_hi + 0.25 * (mu_hard - mu_hi), 0.98 * mu_hard)
            n_exp += 1

        # Bracket from mu* = 0, not from 0.02: when MU_REF is small the root
        # can lie below 0.02, and starting above it silently produces a
        # same-sign bracket.
        f_lo = f(0.0)
        if f_lo <= 0:
            # No positive mu*_eff exists: Eliashberg Tc is already BELOW
            # Allen-Dynes at zero Coulomb repulsion, i.e. AD OVERSHOOTS.
            # This is not a failure -- it is the expected behaviour at large
            # lambda, where f1 approaches the sqrt(lambda) asymptote from
            # above. Record it as a sign change.
            rows.append((lam, np.nan))
            print(f"{lam:7.2f} {tad:10.3f} {tme:10.3f} {tme / tad:8.4f} "
                  f"{'AD overshoots':>9} {'(no positive root)':>19}",
                  flush=True)
            continue
        try:
            mu_eff = brentq(f, 0.0, mu_hi, xtol=2e-4)
            if MU_REF > 0:
                implied = CUTOFF * np.exp(1.0 / mu_eff - 1.0 / MU_REF)
                imp_s = f"{implied:19.3f}"
            else:
                imp_s = f"{'n/a (mu_ref=0)':>19}"
            rows.append((lam, mu_eff))
            print(f"{lam:7.2f} {tad:10.3f} {tme:10.3f} {tme / tad:8.4f} "
                  f"{mu_eff:9.4f} {imp_s}", flush=True)
        except Exception as exc:
            print(f"{lam:7.2f} {tad:10.3f} {tme:10.3f} {tme / tad:8.4f} "
                  f"   brentq failed: {exc}", flush=True)

    if len(rows) < 4:
        print("\nnot enough points to conclude")
        return

    lams = np.array([r[0] for r in rows])
    mus = np.array([r[1] for r in rows])
    n_overshoot = int(np.isnan(mus).sum())
    if n_overshoot:
        lo = float(np.nanmin(lams[np.isnan(mus)]))
        print(f"\nSIGN CHANGE: Allen-Dynes overshoots for lambda >= {lo:g} "
              f"({n_overshoot} points, no positive mu*_eff exists).")
        print("  This is the predicted turnover: f1 approaches the sqrt(lambda)")
        print("  asymptote FROM ABOVE. A numerical artifact from the shrinking")
        print("  Matsubara count could not produce a sign change.")
    keep = ~np.isnan(mus)
    lams, mus = lams[keep], mus[keep]
    if len(mus) < 4:
        print("\ntoo few finite points for the plateau/drift decomposition")
        return

    # --------------------------------------------------------------------
    # NOTE ON A BUG THAT WAS HERE.
    # The original verdict test was `mus.std() < 0.01`. Standard deviation
    # cannot distinguish random scatter about a constant from smooth
    # SYSTEMATIC DRIFT -- and drift is exactly what this diagnostic exists to
    # detect. The first run produced a plateau at low lambda followed by a
    # strictly monotone rise, and the std test called it "constant".
    #
    # Replaced with an explicit plateau/drift decomposition:
    #   a convention mismatch is lambda-INDEPENDENT by construction
    #     (it is a property of how mu* is defined, not of the material)
    #   real Allen-Dynes error is lambda-DEPENDENT
    #     (it is a fit degrading away from where it was fitted)
    # --------------------------------------------------------------------

    # plateau = the low-lambda region, which is also where mu*_eff is BEST
    # determined: |dlnTc/dmu*| is ~13x larger at lambda=0.4 than at 2.0, so
    # these points pin the constant far more tightly than the high-lambda ones.
    plat = lams <= 0.8
    if plat.sum() >= 2:
        mu_const = float(mus[plat].mean())
        plat_spread = float(mus[plat].max() - mus[plat].min())
    else:
        mu_const, plat_spread = float(mus[0]), float("nan")

    tail = ~plat
    print(f"\nplateau (lambda <= 0.8): mu*_eff = {mu_const:.4f} "
          f"(spread {plat_spread:.4f}, n={int(plat.sum())})")
    print(f"full range: [{mus.min():.4f}, {mus.max():.4f}]  "
          f"mean={mus.mean():.4f}  std={mus.std():.4f}")

    # drift test: strict monotonicity above the plateau, and size vs solver tol
    SOLVER_TOL = 2e-4
    drift = float(mus[tail].max() - mu_const) if tail.sum() else 0.0
    strictly_monotone = bool(tail.sum() >= 3
                             and np.all(np.diff(mus[tail]) > 0))
    print(f"drift above plateau: {drift:+.4f}  "
          f"({drift / SOLVER_TOL:.0f}x solver tolerance)  "
          f"strictly monotone: {strictly_monotone}")

    print()
    if abs(drift) < 5 * SOLVER_TOL:
        print("VERDICT: no resolvable drift -> pure convention mismatch.")
        print(f"         Adopt mu* = {mu_const:.4f} at cutoff {CUTOFF}*w_max.")
    elif strictly_monotone:
        print("VERDICT: plateau + strictly monotone drift -> BOTH effects.")
        print(f"         Convention constant  = {mu_const:.4f} "
              f"(read off the plateau, NOT the mean --")
        print("           averaging over the drift contaminates the constant)")
        print(f"         Residual lambda-dependent part = real Allen-Dynes "
              f"error, up to {drift:+.4f} in mu*_eff.")
        print()
        print("         BEFORE BELIEVING THE DRIFT, run the two discriminators:")
        print("           --cutoff-factor 20 (and 30), rescaled to a common")
        print("             cutoff: does the plateau+rise shape survive?")
        print("           --lambdas 1,2,3,5,10,20,40 : Allen-Dynes approaches")
        print("             the sqrt(lambda) asymptote FROM ABOVE, so real f1")
        print("             error must TURN OVER and change sign near lambda")
        print("             ~20-30. A numerical artifact from the shrinking")
        print("             Matsubara count cannot turn over -- N falls")
        print("             monotonically as Tc rises. Sign change vs monotone")
        print("             growth is the cleanest discriminator available.")
    else:
        print("VERDICT: drift present but not monotone -> suspect numerics")
        print("         before claiming physics. Run the cutoff check first.")

    print()
    print("Reminder: these are Einstein spectra, so w_2/w_log = 1 and f2 = 1")
    print("exactly. This isolates f1 (strong-coupling) and says NOTHING about")
    print("f2 (spectral shape). The shape channel is what moment_matched.py")
    print("probes -- the two are complementary, not redundant.")
    print()
    print("Also: at mu* = 0 there is no convention ambiguity at all, since mu*")
    print("is the only cutoff-dependent quantity. Running this with")
    print("--mu-ref 0 measures intrinsic Allen-Dynes error directly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff-factor", type=float, default=CUTOFF,
                    help="Matsubara cutoff in units of w_max (default 10)")
    ap.add_argument("--mu-ref", type=float, default=MU_REF,
                    help="Allen-Dynes mu* to target; 0 removes all convention "
                         "ambiguity and measures intrinsic AD error")
    ap.add_argument("--lambdas", type=str, default=None,
                    help="comma-separated lambda values, e.g. 1,2,3,5,10,20,40")
    a = ap.parse_args()
    CUTOFF = a.cutoff_factor
    MU_REF = a.mu_ref
    if a.lambdas:
        LAMBDAS = tuple(float(x) for x in a.lambdas.split(","))
    main()
