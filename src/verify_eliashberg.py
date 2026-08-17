"""
Verification suite for eliashberg.py.

These are not unit tests in the "does it run" sense. Each one is a claim the
solver has to satisfy for the physics to be right, with an independent source
of truth that does not come from my own code.

  T1  alpha^2F integration reproduces the database's stored lambda, w_log, w_2
  T2  power iteration agrees with exact eigendecomposition
  T3  Tc is converged w.r.t. the Matsubara cutoff
  T4  ASYMPTOTIC LIMIT: for an Einstein spectrum at mu*=0 and large lambda,
      k_B Tc -> 0.1827 * sqrt(lambda * <w^2>)   [Allen & Dynes 1975, eq. 5.7]
      This is an analytic result derived independently of any fitted formula
      and is the single strongest check on the solver.
  T5  rho(T) is monotonically decreasing, so the bisection is well posed
  T6  weak-coupling regime: Tc_ME and Tc_AD should agree to within a few
      percent, because Allen-Dynes was FITTED to numerical ME solutions there.
      Systematic disagreement at low lambda would indicate a convention bug.
  T7  mu* cutoff convention: Tc must be stable when mu* is rescaled to a
      different cutoff using the standard logarithmic relation.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eliashberg import (  # noqa: E402
    KB_MEV_PER_K, _arnoldi_eigenvalue, _me_dense, _me_eigenvalue, _me_matvec,
    _me_pieces, a2f_moments, allen_dynes_tc, eliashberg_tc, rescale_mu_star,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "bete_database.json")

# mu* conventions, calibrated on Einstein spectra by src/diagnose_mustar.py;
# see the MU_STAR_* block in build_physics_dataset.py for the full reasoning.
# The closed form keeps Allen-Dynes's number; the solver takes the equivalent
# of the SAME physical repulsion at our cutoff. T6 below is the transfer test:
# the constant is fitted on Einstein spectra and applied to real materials, so
# T6 passing is evidence the calibration generalises beyond the shapes it was
# read off -- not a tautology.
MU_STAR_AD = 0.10
MU_STAR_ME = 0.1293

results = []


def check(name, passed, detail):
    results.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}\n       {detail}", flush=True)


def einstein_spectrum(w_E: float, lam: float, n: int = 4000, width: float = None):
    """
    Narrow Gaussian standing in for a delta function at w_E, normalised so that
    lambda = 2 * INT a2F/w dw comes out to the requested value.
    """
    width = width or w_E / 60.0
    w = np.linspace(max(1e-4, w_E - 10 * width), w_E + 10 * width, n)
    g = np.exp(-0.5 * ((w - w_E) / width) ** 2)
    a = g / (2.0 * np.trapezoid(g / w, w)) * lam
    return w, a


def main():
    with open(RAW) as fh:
        db = json.load(fh)

    # ---------------- T1: moments vs stored values ----------------
    devs = {"lambda": [], "w_log": [], "w_2": []}
    for k in list(db["lambda"].keys())[:200]:
        w = np.asarray(db["Freq_meV"][k], float)
        a = np.asarray(db["a2F"][k], float)
        m = a2f_moments(w, a)
        devs["lambda"].append(abs(m["lambda_"] - db["lambda"][k]) / db["lambda"][k])
        devs["w_log"].append(abs(m["w_log"] - db["w_log"][k]) / db["w_log"][k])
        devs["w_2"].append(abs(m["w_2"] - np.sqrt(db["w_sq"][k])) /
                           np.sqrt(db["w_sq"][k]))
    worst = {k: float(np.max(v)) for k, v in devs.items()}
    check("T1 a2F integration vs stored moments", max(worst.values()) < 5e-3,
          f"max relative deviation over 200 materials: {worst}")

    # ---------------- T2: power iteration vs exact ----------------
    w, a = np.asarray(db["Freq_meV"]["305"], float), np.asarray(db["a2F"]["305"], float)
    errs = []
    for T in (5.0, 12.0, 25.0):
        r_pow = _me_eigenvalue(T, w, a, 0.10, 10.0, exact=False)
        r_exa = _me_eigenvalue(T, w, a, 0.10, 10.0, exact=True)
        errs.append(abs(r_pow - r_exa) / abs(r_exa))
    check("T2 power iteration vs exact eigendecomposition", max(errs) < 1e-6,
          f"max relative eigenvalue error over T=5,12,25 K: {max(errs):.2e}")

    # ---------------- T2b: matrix-free matvec vs dense matvec ----------
    # T2 compares EIGENVALUES, which are integrated over the whole operator: an
    # indexing slip in the Toeplitz embedding can perturb the spectrum by less
    # than T2's threshold and survive it. This compares the OPERATOR, applied
    # to random vectors, which is far more sensitive, and it directly
    # regression-tests the matrix-free rewrite rather than a consequence of it.
    #
    # T = 0.2 K is chosen so the physical n_cut far exceeds every target,
    # letting max_matsubara pin n_cut exactly; the sizes are asserted below
    # because a test that silently ran at the wrong N would prove nothing.
    rng = np.random.default_rng(0)
    mv_errs, sizes = [], []
    for ncut in (60, 300, 1200):
        T, w_n, lam_d, Z = _me_pieces(0.2, w, a, 10.0, ncut)
        sizes.append((ncut, w_n.size, w_n.size == 2 * ncut + 1))
        for mu in (0.0, 0.10, 0.1293):
            K = _me_dense(T, w_n, lam_d, Z, mu)
            mv = _me_matvec(T, w_n, lam_d, Z, mu)
            for _ in range(3):
                v = rng.standard_normal(w_n.size)
                ref = K @ v
                mv_errs.append(float(np.linalg.norm(mv(v) - ref)
                                     / np.linalg.norm(ref)))
    pinned = all(ok for _, _, ok in sizes)
    check("T2b matrix-free matvec vs dense matvec",
          pinned and max(mv_errs) < 1e-12,
          f"max relative error over n_cut={[s[0] for s in sizes]} x "
          f"mu*=0,0.10,0.1293 x 3 random vectors: {max(mv_errs):.2e}"
          + ("" if pinned else f"  !! n_cut NOT pinned: {sizes}"))

    # ---------------- T2c: the Arnoldi fallback vs exact ----------------
    # This path only fires when power iteration stalls -- rare, and only at
    # large N. Left untested it would sit unexercised until it silently
    # produced a wrong Tc in a production build, which is exactly how it was
    # first noticed (a stall warning during a 60-material smoke build).
    arn_errs = []
    for T_k in (5.0, 12.0, 25.0):
        T, w_n, lam_d, Z = _me_pieces(T_k, w, a, 10.0, 3000)
        r_arn = _arnoldi_eigenvalue(_me_matvec(T, w_n, lam_d, Z, 0.10),
                                    w_n.size, T_k, float("nan"))
        r_exa = float(np.max(np.linalg.eigvals(
            _me_dense(T, w_n, lam_d, Z, 0.10)).real))
        arn_errs.append(abs(r_arn - r_exa) / abs(r_exa))
    check("T2c Arnoldi fallback vs exact eigendecomposition",
          max(arn_errs) < 1e-8,
          f"max relative eigenvalue error over T=5,12,25 K: {max(arn_errs):.2e}")

    # ---------------- T2d: largest REAL part, not largest magnitude ----
    # Tc is defined by the largest real part of the gap kernel. Power iteration
    # converges to the largest MAGNITUDE. mu* enters as a negative rank-1 term
    # whose magnitude grows with mu*, so above mu* ~ 0.2 a negative eigenvalue
    # overtakes the positive one and power iteration converges cleanly onto the
    # wrong answer -- no stall, so the iteration-cap guard never fires.
    #
    # The sweep must actually REACH the negative-dominant regime, otherwise it
    # silently stops testing the thing it exists for; that is asserted too.
    we2, ae2 = einstein_spectrum(10.0, 0.4)
    lr_errs, n_neg, first_neg = [], 0, None
    for T_k in (2.0, 0.5):
        for mu in (0.10, 0.1293, 0.1840, 0.25, 0.30, 0.32):
            T, w_n, lam_d, Z = _me_pieces(T_k, we2, ae2, 10.0, 400)
            ev = np.linalg.eigvals(_me_dense(T, w_n, lam_d, Z, mu)).real
            rho_lr, rho_lm = float(ev.max()), float(ev[np.argmax(np.abs(ev))])
            if rho_lm < rho_lr - 1e-12:          # negative eigenvalue dominates
                n_neg += 1
                first_neg = first_neg if first_neg is not None else mu
            got = _me_eigenvalue(T_k, we2, ae2, mu, 10.0, max_matsubara=400)
            lr_errs.append(abs(got - rho_lr) / max(abs(rho_lr), 1e-12))
    check("T2d largest real part vs largest magnitude (mu* sweep)",
          max(lr_errs) < 1e-8 and n_neg > 0,
          f"max relative error over mu*=0.10..0.32 x T=2,0.5 K: "
          f"{max(lr_errs):.2e}; negative-dominant in {n_neg}/12 cases"
          + (f", first at mu*={first_neg}" if first_neg else
             " -- SWEEP NEVER REACHED THE REGIME IT TESTS"))

    # ---------------- T3: Matsubara cutoff, MATCHED PAIRS ----------------
    # The previous T3 asserted that Tc converges with cutoff at FIXED mu*. That
    # quantity CANNOT converge: the Coulomb kernel is constant in (n,m), so its
    # Matsubara sum grows like ln(w_c/T) while the phonon kernel's converges.
    # The old test passed only on where a 2% threshold happened to sit relative
    # to a constant ~1.4%-per-doubling drift -- it could not fail for the right
    # reason. `check_cutoff_convergence.py` was deleted for exactly this error
    # while the same error sat here in the verification suite.
    #
    # Only the PAIR (w_c, mu*(w_c)) is physical. Rescaling mu* by Morel-Anderson
    # between cutoffs, the matched sequence must converge. Both legs are
    # asserted so the test can fail in either direction:
    #
    #   matched   -- increment 20->40 below 0.2%  (it converges)
    #   fixed mu* -- increments constant per doubling to within 15% (it does
    #                NOT converge, and it diverges in the specific logarithmic
    #                way the mechanism predicts)
    #
    # Al is included because low lambda amplifies mu*: its fixed-mu* drift is
    # ~4x Nb's, which is what makes the failure mode visible rather than
    # marginal. cf=5 is excluded -- there the phonon kernel itself is not
    # converged and the matched pair breaks too (Nb +1.9%).
    _inv = {str(v): k for k, v in db["comp"].items()}
    t3_detail, t3_pass = [], True
    for _nm in ("Nb", "Al"):
        _k = _inv.get(_nm)
        if _k is None:
            continue
        wm = np.asarray(db["Freq_meV"][_k], float)
        am = np.asarray(db["a2F"][_k], float)
        w_max_m = float(wm.max())
        matched, fixed = {}, {}
        for cf in (10.0, 20.0, 40.0):
            mu_cf = rescale_mu_star(MU_STAR_ME, 10.0 * w_max_m, cf * w_max_m)
            matched[cf] = eliashberg_tc(wm, am, mu_cf, cutoff_factor=cf,
                                        tol=5e-4)
            fixed[cf] = eliashberg_tc(wm, am, MU_STAR_ME, cutoff_factor=cf,
                                      tol=5e-4)
        dm = {cf: 100.0 * (matched[cf] / matched[10.0] - 1.0) for cf in matched}
        df_ = {cf: 100.0 * (fixed[cf] / fixed[10.0] - 1.0) for cf in fixed}
        step_matched = abs(dm[40.0] - dm[20.0])          # must be small
        inc1, inc2 = df_[20.0], df_[40.0] - df_[20.0]    # must be ~equal
        const_dev = abs(inc2 / inc1 - 1.0) if abs(inc1) > 1e-9 else np.inf
        ok = (step_matched < 0.2) and (const_dev < 0.15)
        t3_pass &= ok
        t3_detail.append(
            f"{_nm}: matched {dm[20.0]:+.2f}%/{dm[40.0]:+.2f}% "
            f"(step {step_matched:.3f}%) | fixed {df_[20.0]:+.2f}%/"
            f"{df_[40.0]:+.2f}% (increments {inc1:+.2f}/{inc2:+.2f}, "
            f"const to {const_dev:.1%})")
    check("T3 Matsubara cutoff, matched pairs vs fixed mu*", t3_pass,
          "   ".join(t3_detail))

    # ---------------- T4: Allen-Dynes asymptotic limit ----------------
    # k_B Tc -> 0.1827 sqrt(lambda <w^2>) as lambda -> infinity, at mu* = 0.
    w_E = 10.0
    rows = []
    for lam in (5.0, 10.0, 25.0, 50.0, 100.0):
        we, ae = einstein_spectrum(w_E, lam)
        mom = a2f_moments(we, ae)
        tc = eliashberg_tc(we, ae, mu_star=0.0, cutoff_factor=20.0, tol=5e-4)
        pred = 0.1827 * np.sqrt(lam * mom["w_2"] ** 2) / KB_MEV_PER_K
        rows.append((lam, tc, pred, tc / pred))
    ratios = [r[3] for r in rows]
    check("T4 Allen-Dynes strong-coupling asymptote (analytic)",
          abs(ratios[-1] - 1.0) < 0.03,
          "lambda -> Tc_ME / Tc_asymptotic: " +
          ", ".join(f"{l:g}->{r:.4f}" for l, _, _, r in rows))

    # ---------------- T5: monotonicity of rho(T) ----------------
    Ts = np.linspace(2.0, 40.0, 25)
    rho = [_me_eigenvalue(T, w, a, 0.10, 10.0) for T in Ts]
    mono = bool(np.all(np.diff(rho) < 1e-9))
    check("T5 rho(T) monotonically decreasing", mono,
          f"rho spans {rho[0]:.3f} -> {rho[-1]:.3f} over T=2..40 K; "
          f"max positive step {max(np.diff(rho)):.2e}")

    # ---------------- T6: weak-coupling agreement with Allen-Dynes ----
    ratios_by_lam = []
    for k in list(db["lambda"].keys()):
        lam = db["lambda"][k]
        if not (0.3 < lam < 0.8):
            continue
        wk = np.asarray(db["Freq_meV"][k], float)
        ak = np.asarray(db["a2F"][k], float)
        m = a2f_moments(wk, ak)
        tad = allen_dynes_tc(m["lambda_"], m["w_log"], m["w_2"], MU_STAR_AD)
        if tad < 0.5:
            continue
        tme = eliashberg_tc(wk, ak, MU_STAR_ME, t_guess=tad)
        if tme > 0:
            ratios_by_lam.append((lam, tme / tad))
        if len(ratios_by_lam) >= 40:
            break
    r = np.array([x[1] for x in ratios_by_lam])
    check("T6 weak/moderate coupling agrees with Allen-Dynes",
          abs(np.median(r) - 1.0) < 0.10,
          f"n={len(r)} materials with 0.3<lambda<0.8: "
          f"Tc_ME/Tc_AD median={np.median(r):.4f}, "
          f"mean={r.mean():.4f}, IQR=[{np.percentile(r, 25):.3f},"
          f"{np.percentile(r, 75):.3f}]")

    # ---------------- T7: mu* cutoff rescaling consistency ------------
    w_max = float(np.max(w))
    mu_a, cf_a = 0.10, 10.0
    cf_b = 30.0
    mu_b = rescale_mu_star(mu_a, cf_a * w_max, cf_b * w_max)
    tc_a = eliashberg_tc(w, a, mu_a, cutoff_factor=cf_a)
    tc_b = eliashberg_tc(w, a, mu_b, cutoff_factor=cf_b)
    dev = abs(tc_a - tc_b) / tc_a
    check("T7 mu* cutoff-rescaling consistency", dev < 0.05,
          f"Nb: mu*={mu_a} at {cf_a}x -> Tc={tc_a:.3f}K ; "
          f"mu*={mu_b:.4f} at {cf_b}x -> Tc={tc_b:.3f}K ; deviation {dev:.2%}")

    print("\n" + "=" * 68)
    n_pass = sum(p for _, p, _ in results)
    print(f"{n_pass}/{len(results)} checks passed")
    for name, passed, _ in results:
        if not passed:
            print(f"  FAILED: {name}")
    return results


if __name__ == "__main__":
    main()
