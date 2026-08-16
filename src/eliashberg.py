"""
Tc from the isotropic Eliashberg spectral function alpha^2 F(omega).

Two levels of theory:

1. `allen_dynes_tc`  -- the closed-form McMillan/Allen-Dynes approximation.
   This is the formula whose *breakdown* the paper is about.

2. `eliashberg_tc`   -- numerically solves the linearized isotropic
   Migdal-Eliashberg equations on the imaginary (Matsubara) axis.
   This is the reference Tc that Allen-Dynes approximates.

Everything is in meV for energies; temperatures are returned in kelvin.

Linearized Migdal-Eliashberg on the imaginary axis
--------------------------------------------------
Matsubara frequencies:      w_n = pi T (2n+1)
Pairing kernel:             lam(n-m) = 2 * INT dw  w a2F(w) / (w^2 + (w_n - w_m)^2)
Mass renormalisation:       Z_n = 1 + (pi T / w_n) * SUM_m lam(n-m) sgn(w_m)
Linearised gap equation:    Z_n D_n = pi T SUM_m [lam(n-m) - mu*] D_m / |w_m|

Tc is the temperature at which the largest eigenvalue of the linear operator
acting on D equals 1. We bracket and bisect on T.

The Coulomb pseudopotential mu* is defined at a cutoff w_c; we take the
Matsubara sum out to w_c = `cutoff_factor` * w_max, which is the standard
convention (cutoff_factor ~ 10). `mu_star_at_cutoff` rescales mu* if you want
to compare cutoffs -- see `rescale_mu_star`.

References
----------
P.B. Allen and R.C. Dynes, Phys. Rev. B 12, 905 (1975).
W.L. McMillan, Phys. Rev. 167, 331 (1968).
P.B. Allen and B. Mitrovic, Solid State Physics 37, 1 (1982).
"""

from __future__ import annotations

import warnings

import numpy as np

# meV per kelvin (k_B)
KB_MEV_PER_K = 0.08617333262

# power-iteration budget; exposed so callers can detect a stall
_POWER_ITERS = 2000

# Matsubara backstop. 1500 was the DENSE kernel's memory ceiling (N = 2*n_cut+1
# and it built three N x N arrays); the matrix-free path is O(N) in memory, so
# that ceiling is obsolete. It stayed as the signature default long enough for
# three call sites to silently inherit it and run capped -- the failure mode
# capping causes is a lambda-CORRELATED bias in Tc, not noise, so a stale
# default here quietly poisons downstream correlations. 250k covers the worst
# material in the 806 (BeGeSc needs n_cut = 22,380 at cutoff_factor = 10).
MAX_MATSUBARA = 250_000

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x / 2.x


# --------------------------------------------------------------------------
# spectral-function moments
# --------------------------------------------------------------------------
def a2f_moments(omega: np.ndarray, a2f: np.ndarray) -> dict:
    """lambda, w_log, w_2 from alpha^2 F on a grid. omega in meV."""
    omega = np.asarray(omega, dtype=float)
    a2f = np.asarray(a2f, dtype=float)
    if omega.shape != a2f.shape:
        raise ValueError("omega and a2f must have the same shape")

    m = omega > 1e-9
    w, f = omega[m], a2f[m]
    if w.size < 3 or not np.any(f > 0):
        return dict(lambda_=np.nan, w_log=np.nan, w_2=np.nan, w_max=np.nan)

    lam = 2.0 * _trapz(f / w, w)
    if lam <= 0:
        return dict(lambda_=np.nan, w_log=np.nan, w_2=np.nan, w_max=float(w.max()))

    w_log = float(np.exp((2.0 / lam) * _trapz(f * np.log(w) / w, w)))
    w_2 = float(np.sqrt((2.0 / lam) * _trapz(f * w, w)))
    return dict(lambda_=float(lam), w_log=w_log, w_2=w_2, w_max=float(w.max()))


def a2f_shape_features(omega: np.ndarray, a2f: np.ndarray) -> dict:
    """
    Shape descriptors of alpha^2 F beyond (lambda, w_log, w_2).

    These are exactly the information Allen-Dynes throws away: AD compresses
    the whole spectral function into three numbers. Anything here that predicts
    the AD error is, by construction, information AD discarded.
    """
    omega = np.asarray(omega, dtype=float)
    a2f = np.asarray(a2f, dtype=float)
    m = omega > 1e-9
    w, f = omega[m], a2f[m]
    out = {}
    if w.size < 3 or not np.any(f > 0):
        return {k: np.nan for k in
                ("w_1", "w_3", "skew_a2f", "kurt_a2f", "frac_low", "frac_high",
                 "n_peaks", "peak_w", "spectral_entropy", "w_max")}

    lam = 2.0 * _trapz(f / w, w)
    # normalised weight  g(w) = (2/lam) a2F(w)/w  integrates to 1
    g = (2.0 / lam) * f / w
    norm = _trapz(g, w)
    g = g / norm

    mom = lambda k: float(_trapz(g * w ** k, w))
    w1 = mom(1)
    var = max(mom(2) - w1 ** 2, 1e-12)
    sd = np.sqrt(var)
    out["w_1"] = w1
    out["w_3"] = float(mom(3) ** (1.0 / 3.0))
    out["skew_a2f"] = float(_trapz(g * ((w - w1) / sd) ** 3, w))
    out["kurt_a2f"] = float(_trapz(g * ((w - w1) / sd) ** 4, w))

    wmax = float(w.max())
    lo = w <= wmax / 3.0
    hi = w >= 2.0 * wmax / 3.0
    out["frac_low"] = float(_trapz(g[lo], w[lo])) if lo.sum() > 2 else 0.0
    out["frac_high"] = float(_trapz(g[hi], w[hi])) if hi.sum() > 2 else 0.0

    # peak structure of a2F itself
    thresh = 0.15 * f.max()
    above = f > thresh
    n_peaks = int(np.sum(above[1:-1] & (f[1:-1] >= f[:-2]) & (f[1:-1] >= f[2:])))
    out["n_peaks"] = n_peaks
    out["peak_w"] = float(w[int(np.argmax(f))])

    p = g / max(_trapz(g, w), 1e-30)
    pp = p[p > 0]
    ww = w[p > 0]
    out["spectral_entropy"] = float(-_trapz(pp * np.log(pp + 1e-30), ww))
    out["w_max"] = wmax
    return out


# --------------------------------------------------------------------------
# closed-form approximations
# --------------------------------------------------------------------------
def mcmillan_tc(lam: float, w_log: float, mu_star: float = 0.10) -> float:
    """
    Original McMillan form written with w_log/1.45 in place of theta_D/1.45.
    Returned in kelvin. (The paper's eq. 1.)
    """
    denom = lam - mu_star * (1.0 + 0.62 * lam)
    if denom <= 0:
        return 0.0
    tc_mev = (w_log / 1.45) * np.exp(-1.04 * (1.0 + lam) / denom)
    return float(tc_mev / KB_MEV_PER_K)


def allen_dynes_tc(lam: float, w_log: float, w_2: float | None = None,
                   mu_star: float = 0.10, corrections: bool = True) -> float:
    """
    Allen-Dynes Tc in kelvin.

    corrections=False -> the bare exponential prefactor w_log/1.2 form.
    corrections=True  -> includes the strong-coupling f1 and shape f2 factors,
                         which is the *best* version of the closed form and
                         therefore the honest baseline to call "breakdown" on.
    """
    denom = lam - mu_star * (1.0 + 0.62 * lam)
    if denom <= 0 or lam <= 0 or w_log <= 0:
        return 0.0
    base = (w_log / 1.2) * np.exp(-1.04 * (1.0 + lam) / denom)

    if corrections and w_2 is not None and w_2 > 0:
        r = w_2 / w_log
        lam1 = 2.46 * (1.0 + 3.8 * mu_star)
        lam2 = 1.82 * (1.0 + 6.3 * mu_star) * r
        f1 = (1.0 + (lam / lam1) ** 1.5) ** (1.0 / 3.0)
        f2 = 1.0 + ((r - 1.0) * lam ** 2) / (lam ** 2 + lam2 ** 2)
        base *= f1 * f2

    return float(base / KB_MEV_PER_K)


# --------------------------------------------------------------------------
# numerical linearized Migdal-Eliashberg
# --------------------------------------------------------------------------
def _lambda_kernel(omega: np.ndarray, a2f: np.ndarray, dw_n: np.ndarray,
                   chunk: int = 4096) -> np.ndarray:
    """
    lam(dw) = 2 * INT dw' w' a2F(w') / (w'^2 + dw^2)   evaluated for each
    bosonic frequency difference dw_n.

    Chunked over dw_n. The obvious fully-vectorised form allocates a
    (len(dw_n), len(w)) array, which is fine at n_cut ~ 1e3 and several GB at
    the n_cut ~ 1e5 the matrix-free solver makes reachable. Chunking caps that
    at chunk * len(w) while doing arithmetic identical to the unchunked form.
    """
    m = omega > 1e-9
    w, f = omega[m], a2f[m]
    wf = w * f
    w_sq = w ** 2
    dw_n = np.asarray(dw_n, float)
    out = np.empty(dw_n.shape[0], dtype=float)
    for s in range(0, dw_n.shape[0], chunk):
        d = dw_n[s:s + chunk]
        integrand = wf[None, :] / (w_sq[None, :] + d[:, None] ** 2)
        out[s:s + chunk] = 2.0 * _trapz(integrand, w, axis=1)
    return out


def _toeplitz_matvec(c: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Symmetric-Toeplitz matrix-vector product, T[i,j] = c[|i-j|], without ever
    forming T.

    Embedding c in a circulant of length m >= 2N-1 turns T @ v into a circular
    convolution, so one forward pair of FFTs and one inverse do the whole
    product: O(N log N) time and O(N) memory, against O(N^2) for both in the
    dense form. That is the entire reason an uncapped Matsubara sum is
    affordable -- see the note in `eliashberg_tc`.

    rfft/irfft rather than the complex transforms because lam(n-m) and the gap
    vector are both real.
    """
    N = v.shape[0]
    m = 1
    while m < 2 * N - 1:
        m *= 2
    emb = np.zeros(m)
    emb[:N] = c[:N]
    emb[m - N + 1:] = c[1:N][::-1]
    return np.fft.irfft(np.fft.rfft(emb) * np.fft.rfft(v, m), m)[:N]


def _dominant_eigenvalue(matvec, n: int, iters: int = _POWER_ITERS,
                         tol: float = 1e-10) -> tuple:
    """
    Largest-magnitude eigenvalue by power iteration, given only a matrix-vector
    product. The ME gap kernel is real and non-symmetric, but its dominant
    eigenvector (the gap function) is node-free and positive, so power
    iteration converges reliably.

    Returns (rho, iterations_used). Callers should treat
    iterations_used == iters as a stall, not as a converged answer.
    """
    v = np.ones(n) / np.sqrt(n)
    rho_old = 0.0
    for it in range(1, iters + 1):
        w = matvec(v)
        nrm = np.linalg.norm(w)
        if nrm < 1e-300:
            return 0.0, it
        v = w / nrm
        rho = float(v @ matvec(v))
        if abs(rho - rho_old) < tol * max(abs(rho), 1e-12):
            return rho, it
        rho_old = rho
    return rho_old, iters


def _me_pieces(T_kelvin: float, omega: np.ndarray, a2f: np.ndarray,
               cutoff_factor: float, max_matsubara: int) -> tuple:
    """
    Everything both the dense and the matrix-free kernel are built from:
    (T in meV, w_n, lam_d, Z).

    lam(n-m) depends only on |n-m|, so the whole kernel is determined by the
    1-D array lam_d -- which is what makes it Toeplitz, and therefore what
    makes the matrix-free path possible. Only N values are needed, since
    |n-m| <= N-1; the old dense path computed 2N of them and then materialised
    an N x N gather.
    """
    T = T_kelvin * KB_MEV_PER_K  # meV
    w_max = float(np.max(omega))
    w_c = cutoff_factor * w_max
    n_cut = int(np.floor((w_c / (np.pi * T) - 1.0) / 2.0))
    n_cut = int(np.clip(n_cut, 8, max_matsubara))

    n = np.arange(-n_cut, n_cut + 1)
    w_n = np.pi * T * (2 * n + 1)                       # (N,)
    N = w_n.size

    dw = 2.0 * np.pi * T * np.arange(N)                 # bosonic differences
    lam_d = _lambda_kernel(omega, a2f, dw)              # (N,)

    # mass renormalisation: also a Toeplitz product, against sgn(w_m)
    Z = 1.0 + (np.pi * T / w_n) * _toeplitz_matvec(lam_d, np.sign(w_n))
    return T, w_n, lam_d, Z


def _me_matvec(T: float, w_n: np.ndarray, lam_d: np.ndarray, Z: np.ndarray,
               mu_star: float):
    """
    v -> K @ v for the linearized gap kernel
        K[n,m] = pi T (lam(n-m) - mu*) / (|w_m| Z_n)
    without forming K.

    Substituting u_m = v_m/|w_m| splits it into a Toeplitz product against
    lam_d plus a rank-1 term, because mu* is CONSTANT in (n,m): the mu* part
    of K is -mu* * outer(1/Z, 1/|w_m|), whose action is just a scaled sum.
    """
    aw = np.abs(w_n)
    piT_over_Z = np.pi * T / Z

    def matvec(v: np.ndarray) -> np.ndarray:
        u = v / aw
        return piT_over_Z * (_toeplitz_matvec(lam_d, u) - mu_star * u.sum())

    return matvec


def _me_dense(T: float, w_n: np.ndarray, lam_d: np.ndarray, Z: np.ndarray,
              mu_star: float) -> np.ndarray:
    """The explicit N x N kernel. Reference path only -- O(N^2) memory."""
    i = np.arange(w_n.size)
    lam_nm = lam_d[np.abs(i[:, None] - i[None, :])]
    return (np.pi * T) * (lam_nm - mu_star) / (np.abs(w_n)[None, :] * Z[:, None])


def _me_eigenvalue(T_kelvin: float, omega: np.ndarray, a2f: np.ndarray,
                   mu_star: float, cutoff_factor: float,
                   max_matsubara: int = 3000, exact: bool = False) -> float:
    """
    Largest eigenvalue of the linearized ME gap operator at temperature T.
    rho > 1  =>  superconducting;  rho < 1  =>  normal.  Tc solves rho = 1.

    `exact=True` builds the dense kernel and diagonalises it. That path is
    O(N^2) in memory and exists to cross-check the matrix-free one (T2 and the
    direct matvec test in verify_eliashberg.py); it is not usable at the
    Matsubara counts the fast path handles routinely.
    """
    if T_kelvin <= 0:
        return np.inf

    T, w_n, lam_d, Z = _me_pieces(T_kelvin, omega, a2f, cutoff_factor,
                                  max_matsubara)
    if exact:
        K = _me_dense(T, w_n, lam_d, Z, mu_star)
        return float(np.max(np.linalg.eigvals(K).real))

    mv = _me_matvec(T, w_n, lam_d, Z, mu_star)
    rho, iters = _dominant_eigenvalue(mv, w_n.size)
    if iters >= _POWER_ITERS or rho <= 0.0:
        # Two different failures, one fallback.
        #
        # STALL: power iteration returns whatever it had reached, which is not
        # the eigenvalue. Happens at large N where the spectrum clusters.
        #
        # WRONG EIGENVALUE: Tc is defined by the largest REAL part, but power
        # iteration converges to the largest MAGNITUDE. mu* enters the kernel as
        # -mu* * outer(1/Z_n, 1/|w_m|), a negative rank-1 term whose magnitude
        # grows with mu*, so above mu* ~ 0.2 it produces a negative eigenvalue
        # that overtakes the positive one. Power iteration then converges
        # CLEANLY -- no stall, so the iters guard never fires -- onto a negative
        # rho, and eliashberg_tc reads rho < 1 and calls a superconducting
        # material normal.
        #
        # rho <= 0 is a sound trigger: a negative value cannot be the largest
        # real part while any positive eigenvalue exists, and where none exists
        # Arnoldi (which="LR") returns the correct negative one anyway.
        #
        # Production mu* (0.1293, 0.1840) sits below the onset, but
        # diagnose_mustar.py brackets up to lambda/(1+0.62 lambda) -- 0.32 at
        # lambda=0.4 -- which is well past it.
        rho = _arnoldi_eigenvalue(mv, w_n.size, T_kelvin, rho)
    return rho


def _arnoldi_eigenvalue(matvec, n: int, T_kelvin: float, fallback: float) -> float:
    """
    Largest-real-part eigenvalue via ARPACK, for the case power iteration
    cannot resolve.

    Power iteration converges like |rho_2/rho_1|^k, which crawls when the top
    of the spectrum is clustered. That happens at low T, where N is large and
    the Matsubara eigenvalues bunch -- precisely the regime uncapping opened
    up. Arnoldi builds a Krylov subspace instead and separates clustered
    eigenvalues properly.
    """
    try:
        from scipy.sparse.linalg import LinearOperator, eigs
    except ImportError:  # pragma: no cover
        warnings.warn("scipy.sparse.linalg unavailable; returning unconverged "
                      f"rho={fallback:.6g}", RuntimeWarning, stacklevel=3)
        return fallback
    op = LinearOperator((n, n), matvec=matvec, dtype=float)
    try:
        vals = eigs(op, k=1, which="LR", return_eigenvectors=False,
                    maxiter=100_000, tol=1e-11)
        return float(np.real(vals[0]))
    except Exception as exc:                      # ARPACK non-convergence
        warnings.warn(
            f"both power iteration and Arnoldi failed at T={T_kelvin:.4g} K, "
            f"N={n}: {exc}; returning unconverged rho={fallback:.6g}",
            RuntimeWarning, stacklevel=3)
        return fallback


def eliashberg_tc(omega: np.ndarray, a2f: np.ndarray, mu_star: float = 0.10,
                  cutoff_factor: float = 10.0, tol: float = 2e-3,
                  t_guess: float | None = None, t_floor: float = 0.005,
                  max_matsubara: int = MAX_MATSUBARA,
                  exact: bool = False) -> float:
    """
    Tc in kelvin from the linearized isotropic Migdal-Eliashberg equations.

    The bracket is seeded from the Allen-Dynes estimate (or `t_guess`) and
    expanded outward, which keeps the Matsubara cutoff -- and therefore the
    matrix size -- modest.

    Returns 0.0 for a material that is not superconducting above `t_floor`,
    NaN if bracketing fails.

    Performance note
    ----------------
    The number of Matsubara frequencies scales as w_c / T: N ~ 2 * cutoff_factor
    * w_max / (2 pi k_B T), so halving T doubles N.

    This USED to be quadratic in cost and quadratic in memory, because the
    kernel was built as an explicit N x N matrix -- which is why `t_floor` was
    originally set at 0.05 K and why `max_matsubara` had to bind at 1500. It no
    longer is: the kernel is Toeplitz and power iteration only needs
    matrix-vector products, so `_toeplitz_matvec` does each step in
    O(N log N) time and O(N) memory. Uncapping is now cheap and `t_floor`
    exists only to stop the bracket search walking to literal zero.

    `t_floor` is therefore a NUMERICAL floor, not a physical one. It is
    deliberately far below any temperature that counts as superconducting:
    deciding what counts is a REPORTING threshold and belongs downstream in the
    analysis, where it can be varied and its effect measured. Conflating the
    two silently selects on Tc -- and since low lambda gives low Tc, that is a
    lambda-correlated cut on the analysis sample.

    `max_matsubara` remains the backstop; if it binds, the effective cutoff is
    lower than requested and the result is flagged by `matsubara_capped()`.
    Capping biases Tc LOW and preferentially hits low-Tc (hence low-lambda)
    materials, so a nonzero capped count is a correlated bias, not noise.
    """
    omega = np.asarray(omega, float)
    a2f = np.asarray(a2f, float)
    if not np.any(a2f > 0):
        return 0.0

    def f(T):
        return _me_eigenvalue(T, omega, a2f, mu_star, cutoff_factor,
                              max_matsubara, exact) - 1.0

    if t_guess is None or not np.isfinite(t_guess) or t_guess <= 0:
        mom = a2f_moments(omega, a2f)
        t_guess = allen_dynes_tc(mom["lambda_"], mom["w_log"], mom["w_2"],
                                 mu_star)
    t_guess = float(np.clip(t_guess if t_guess > 0 else 1.0, t_floor, 400.0))

    T_CEIL = 1500.0
    lo, hi = max(t_guess / 2.0, t_floor), t_guess * 2.0

    # expand upward while still superconducting at the top of the bracket
    n = 0
    while f(hi) > 0:
        lo, hi = hi, hi * 2.0
        n += 1
        if hi > T_CEIL or n > 20:
            return np.nan

    # expand downward while already normal at the bottom of the bracket
    n = 0
    while f(lo) < 0:
        hi, lo = lo, lo / 2.0
        n += 1
        if lo < t_floor or n > 20:
            # one final check at the floor rather than marching to T -> 0
            return 0.0 if f(t_floor) < 0 else np.nan
        lo = max(lo, t_floor)

    # bisection: rho(T) decreases monotonically in T
    while (hi - lo) > tol * max(lo, 1e-6) and (hi - lo) > 1e-4:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def matsubara_capped(tc_kelvin: float, w_max: float, cutoff_factor: float = 10.0,
                     max_matsubara: int = MAX_MATSUBARA) -> bool:
    """True if `max_matsubara` bound the sum at this Tc, so the effective
    cutoff was lower than `cutoff_factor * w_max` and the value is suspect."""
    if not np.isfinite(tc_kelvin) or tc_kelvin <= 0:
        return False
    T = tc_kelvin * KB_MEV_PER_K
    n_needed = (cutoff_factor * w_max / (np.pi * T) - 1.0) / 2.0
    return bool(n_needed > max_matsubara)


def rescale_mu_star(mu_star: float, w_c_from: float, w_c_to: float) -> float:
    """
    Standard logarithmic rescaling of the Coulomb pseudopotential between
    cutoffs:  1/mu*(w2) = 1/mu*(w1) + ln(w1/w2).
    """
    inv = 1.0 / mu_star + np.log(w_c_from / w_c_to)
    return float(1.0 / inv)
