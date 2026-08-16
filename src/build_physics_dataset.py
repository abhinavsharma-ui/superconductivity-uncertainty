"""
Build the physics dataset (step 3).  PARALLEL VERSION.

Input : data/raw/bete_database.json -- 806 DFT electron-phonon calculations
        (Hennig group BETE-NET training set), each with the full Eliashberg
        spectral function alpha^2 F(omega) plus lambda, w_log, w_sq.

Output: data/processed/physics_dataset.csv with, per material,
          - the Allen-Dynes inputs           lambda, w_log, w_2
          - shape descriptors of alpha^2 F   (what AD throws away)
          - Tc_ME    : linearized Migdal-Eliashberg Tc  (the reference)
          - Tc_AD    : Allen-Dynes Tc with f1,f2 corrections
          - Tc_AD_nc : Allen-Dynes without corrections
          - Tc_McM   : the McMillan form in the project handoff
          - ad_error : log(Tc_ME / Tc_AD) -- the breakdown magnitude
        in Allen-Dynes's mu* convention (0.10 primary, 0.13 sensitivity). The
        solver runs at the calibrated equivalents of those, not the same
        numbers -- see the MU_STAR_* block below, which is the whole reason
        ad_error means anything.

Why Tc_ME and not a tabulated Tc: the hypothesis is about *Allen-Dynes
breakdown*. Breakdown is the gap between the closed form and the theory it
approximates, both evaluated from the same alpha^2 F. Using an experimental Tc
instead would mix in DFT error, anisotropy, and sample quality, none of which
are Allen-Dynes breakdown.

Usage
-----
    python src/build_physics_dataset.py                 # all cores
    python src/build_physics_dataset.py --workers 4
    python src/build_physics_dataset.py --limit 30      # smoke test
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

# One BLAS thread per worker: the parallelism is across materials, and letting
# each worker spawn its own thread pool oversubscribes the CPU and runs slower.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _be_polite() -> None:
    """
    Drop to below-normal priority.

    This is a long batch job on someone's only machine. At normal priority it
    saturates every core and the desktop stops responding, which is how the
    first two attempts got killed. Below-normal means the OS preempts us for
    anything interactive: the build takes marginally longer in wall-clock and
    the machine stays usable, which is the trade that actually gets it
    finished. Worker processes inherit the priority class from the parent.
    """
    try:
        if os.name == "nt":
            import ctypes
            BELOW_NORMAL = 0x00004000
            k32 = ctypes.windll.kernel32
            # GetCurrentProcess returns the pseudo-handle (HANDLE)-1. ctypes
            # defaults restype to c_int, which mangles it on 64-bit, and
            # SetPriorityClass then fails by RETURN CODE rather than by
            # raising -- so the first version of this silently did nothing.
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            h = ctypes.c_void_p(k32.GetCurrentProcess())
            if not k32.SetPriorityClass(h, BELOW_NORMAL):
                raise OSError(f"SetPriorityClass failed: {ctypes.WinError()}")
        else:
            os.nice(10)
    except Exception as exc:
        # never fatal -- but say so, rather than quietly saturating the machine
        print(f"  [warn] could not lower process priority: {exc}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eliashberg import (  # noqa: E402
    a2f_moments, a2f_shape_features, allen_dynes_tc, eliashberg_tc,
    matsubara_capped, mcmillan_tc,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "bete_database.json")
OUT = os.path.join(ROOT, "data", "processed", "physics_dataset.csv")
PARTIAL = OUT + ".partial"

# Coulomb pseudopotential, in the two conventions that must be kept DISTINCT.
# mu* has no meaning without the cutoff it is defined at,
#     mu*(w_c) = mu / (1 + mu ln(E_F/w_c))
# so the same physical repulsion carries different numbers at different
# cutoffs. The closed forms are evaluated in Allen-Dynes's convention; the
# solver runs at ours (CUTOFF_FACTOR * w_max). Collapsing both onto a single
# number is the bug that made verify_eliashberg T6 fail by 18%.
#
# src/diagnose_mustar.py calibrates the map on Einstein spectra (where
# w_2/w_log = 1 so f2 = 1 exactly), reading the constant off the low-lambda
# plateau -- where AD was actually fitted, and where |dlnTc/dmu*| is ~13x
# larger than at lambda=2 so the constant is most tightly pinned:
#
#     AD mu* = 0.10  <->  ME mu* = 0.1293  at 10*w_max   (plateau, n=4)
#     AD mu* = 0.13  <->  ME mu* = 0.1840  at 10*w_max   (plateau, n=4)
#
# Above the plateau mu*_eff drifts upward with lambda. That drift is real
# Allen-Dynes f1 error, not convention -- confirmed by the sign change at
# mu* = 0 (where no convention ambiguity exists), which a monotone Matsubara
# artifact cannot produce. It is deliberately NOT absorbed into these
# constants: it is signal, and averaging it in would contaminate them.
MU_STAR_AD = 0.10
MU_STAR_ME = 0.1293
MU_STAR_AD_ALT = 0.13
MU_STAR_ME_ALT = 0.1840
CUTOFF_FACTOR = 10.0

# --- the two roles that used to be one constant --------------------------
# SOLVER_FLOOR_K is NUMERICAL: where the Tc bisection stops descending. It was
# 0.05 K only because the old dense kernel made small T ruinous; the
# matrix-free solver is O(N log N) / O(N) so it can go far lower.
#
# SC_THRESHOLD_K is a REPORTING choice: what counts as superconducting. It is
# applied downstream, documented, and swept in analyze_ad_error.py, because
# thresholding Tc is lambda-correlated selection -- low lambda gives low Tc, so
# the cut drops low-lambda materials preferentially. That is structurally the
# same bias as Matsubara capping, just milder, and it must not be silently
# baked into the dataset. Do NOT tune it to a preferred superconducting count.
SOLVER_FLOOR_K = 0.005
SC_THRESHOLD_K = 0.05
# high enough that nothing at cutoff_factor=10 caps: worst observed n_needed
# over the 806 is 22,380 (BeGeSc). Capping biases Tc low and correlates with
# lambda, so a nonzero capped count invalidates the analysis.
MAX_MATSUBARA = 250_000


def _process_one(item: tuple) -> dict | None:
    """Worker: everything for a single material. Must be top level to pickle."""
    _be_polite()   # workers are spawned fresh on Windows; make it stick
    key, formula, source_name, lam_db, wlog_db, wsq_db, w_list, a_list = item

    w = np.asarray(w_list, float)
    a = np.asarray(a_list, float)

    mom = a2f_moments(w, a)
    lam, w_log, w_2 = mom["lambda_"], mom["w_log"], mom["w_2"]
    if not np.isfinite(lam) or lam <= 0:
        return None

    row = {
        "material_id": key,
        "formula": formula,
        "source_name": source_name,
        # stored values, kept so we can prove our integration matches
        "lambda_db": lam_db,
        "w_log_db": wlog_db,
        "w_2_db": float(np.sqrt(wsq_db)),
        # our own moments
        "lambda": lam,
        "w_log": w_log,
        "w_2": w_2,
        "w_ratio": w_2 / w_log if w_log > 0 else np.nan,
    }
    row.update(a2f_shape_features(w, a))

    # closed forms: Allen-Dynes's own convention
    tc_ad = allen_dynes_tc(lam, w_log, w_2, MU_STAR_AD, corrections=True)
    tc_ad_nc = allen_dynes_tc(lam, w_log, w_2, MU_STAR_AD, corrections=False)
    tc_mcm = mcmillan_tc(lam, w_log, MU_STAR_AD)

    # solver: our cutoff, hence the calibrated equivalent of the SAME repulsion
    tc_me = (0.0 if tc_ad < SOLVER_FLOOR_K else
             eliashberg_tc(w, a, mu_star=MU_STAR_ME, cutoff_factor=CUTOFF_FACTOR,
                           t_guess=tc_ad, t_floor=SOLVER_FLOOR_K,
                           max_matsubara=MAX_MATSUBARA))

    tc_ad_alt = allen_dynes_tc(lam, w_log, w_2, MU_STAR_AD_ALT)
    tc_me_alt = (0.0 if tc_ad_alt < SOLVER_FLOOR_K else
                 eliashberg_tc(w, a, mu_star=MU_STAR_ME_ALT,
                               cutoff_factor=CUTOFF_FACTOR, t_guess=tc_ad_alt,
                               t_floor=SOLVER_FLOOR_K,
                               max_matsubara=MAX_MATSUBARA))

    row.update({
        "Tc_ME": tc_me, "Tc_AD": tc_ad, "Tc_AD_nc": tc_ad_nc, "Tc_McM": tc_mcm,
        "Tc_ME_mu13": tc_me_alt, "Tc_AD_mu13": tc_ad_alt,
        # flag any row where the Matsubara cap bound the sum -> value suspect
        "capped": matsubara_capped(tc_me, mom["w_max"], CUTOFF_FACTOR,
                                   MAX_MATSUBARA),
    })
    return row


def main(limit: int | None = None, workers: int | None = None,
         verbose: bool = True, resume: bool = False) -> pd.DataFrame:
    with open(RAW) as fh:
        db = json.load(fh)

    keys = list(db["lambda"].keys())
    if limit:
        keys = keys[:limit]

    # Uncapped, this build runs for hours, so it checkpoints: completed rows
    # are flushed to PARTIAL periodically and --resume skips them. Without it a
    # single interruption costs the whole run.
    # A --limit run must NOT write to the canonical path. A 60-row smoke file
    # sitting there is indistinguishable from a full build to everything
    # downstream, which is precisely how a stale dataset gets analysed as if it
    # were real.
    out_path = OUT if not limit else OUT.replace(".csv", f".limit{limit}.csv")
    partial_path = out_path + ".partial"

    rows: list[dict] = []
    if resume and os.path.exists(partial_path):
        done = pd.read_csv(partial_path)
        rows = done.to_dict("records")
        have = {str(k) for k in done["material_id"]}
        keys = [k for k in keys if str(k) not in have]
        if verbose:
            print(f"resuming from {partial_path}: {len(rows)} done, "
                  f"{len(keys)} remaining", flush=True)

    items = [(k, db["comp"][k], db["comp_name"][k], db["lambda"][k],
              db["w_log"][k], db["w_sq"][k], db["Freq_meV"][k], db["a2F"][k])
             for k in keys]

    _be_polite()
    # half the cores, not all-but-one: leaves the machine usable. Override
    # with --workers if you actually want the box to yourself.
    n_workers = workers or max(1, cpu_count() // 2)
    t0 = time.time()
    if verbose:
        print(f"{len(items)} materials on {n_workers} workers", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    def _tick(j: int, n_new: int) -> None:
        if verbose and (j + 1) % 50 == 0:
            print(f"  {j + 1}/{n_new} ({time.time() - t0:.0f}s)", flush=True)
        if (j + 1) % 50 == 0 and rows:
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    if n_workers == 1:
        for j, it in enumerate(items):
            r = _process_one(it)
            if r:
                rows.append(r)
            _tick(j, len(items))
    else:
        with Pool(n_workers) as pool:
            for j, r in enumerate(pool.imap_unordered(_process_one, items,
                                                      chunksize=4)):
                if r:
                    rows.append(r)
                _tick(j, len(items))

    df = pd.DataFrame(rows).sort_values("material_id",
                                        key=lambda s: s.astype(int))

    # breakdown magnitude: how far the closed form is from the theory it
    # approximates. log-ratio so it is symmetric and scale-free.
    # defined as widely as the solver resolves, NOT at the reporting threshold:
    # narrowing it here would bake a lambda-correlated cut into the dataset
    ok = ((df["Tc_ME"] > SOLVER_FLOOR_K) & (df["Tc_AD"] > SOLVER_FLOOR_K)
          & (df["Tc_AD_nc"] > SOLVER_FLOOR_K))
    df["ad_error"] = np.nan
    df["ad_error_nc"] = np.nan
    df.loc[ok, "ad_error"] = np.log(df.loc[ok, "Tc_ME"] / df.loc[ok, "Tc_AD"])
    df.loc[ok, "ad_error_nc"] = np.log(df.loc[ok, "Tc_ME"]
                                       / df.loc[ok, "Tc_AD_nc"])

    # same measure in the mu* = 0.13 convention, so step 8 can check that the
    # headline result survives the sensitivity column rather than re-deriving it
    ok13 = ((df["Tc_ME_mu13"] > SOLVER_FLOOR_K)
            & (df["Tc_AD_mu13"] > SOLVER_FLOOR_K))
    df["ad_error_mu13"] = np.nan
    df.loc[ok13, "ad_error_mu13"] = np.log(df.loc[ok13, "Tc_ME_mu13"]
                                           / df.loc[ok13, "Tc_AD_mu13"])
    # reporting threshold, not a physical fact -- see SC_THRESHOLD_K above.
    # Tc_ME is stored raw so downstream can re-threshold without a rebuild.
    df["is_sc"] = df["Tc_ME"] > SC_THRESHOLD_K

    df.to_csv(out_path, index=False)
    if os.path.exists(partial_path):
        os.remove(partial_path)   # only now is the run genuinely complete

    if verbose:
        dt = time.time() - t0
        print(f"\nwrote {out_path}  shape={df.shape}  ({dt:.0f}s, "
              f"{dt / max(len(df), 1):.2f}s/material)")
        print(f"superconducting (Tc_ME > {SC_THRESHOLD_K} K, REPORTING "
              f"threshold): {int(df.is_sc.sum())} / {len(df)}")
        for thr in (0.01, 0.05, 0.1):
            print(f"    at {thr:>5} K: {int((df['Tc_ME'] > thr).sum()):3d}")
        n_bad = int(df["Tc_ME"].isna().sum())
        n_cap = int(df["capped"].sum())
        if n_bad:
            print(f"  WARNING: {n_bad} rows failed to bracket (Tc_ME = NaN)")
        if n_cap:
            print(f"  WARNING: {n_cap} rows hit the Matsubara cap")
        for c in ("lambda", "w_log", "w_2"):
            rel = np.abs(df[c] - df[f"{c}_db"]) / np.abs(df[f"{c}_db"])
            print(f"  {c:6s} vs stored: max rel. dev = {rel.max():.2e}")
        sc = df[df.is_sc]
        print(f"  Tc_ME/Tc_AD  median={np.exp(sc.ad_error).median():.4f}  "
              f"IQR=[{np.exp(sc.ad_error).quantile(.25):.3f},"
              f"{np.exp(sc.ad_error).quantile(.75):.3f}]")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test on the first N; writes to a SEPARATE file")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the .partial checkpoint")
    a = ap.parse_args()
    main(limit=a.limit, workers=a.workers, resume=a.resume)
