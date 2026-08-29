"""The lambda ladder: seven targets at matched r ~ 1.19, lambda spanning 4.4x.

Why this design. lambda cannot be resolved on the nine-cell grid at any donor
count -- lambda and r are collinear there, and partial(sd, lambda | r) fails
leave-one-out on a single cell. Holding r fixed by construction and varying
lambda removes the confounder instead of partialling it out. r ~ 1.19 sits near
the donor pool's median, so reachability stays ~90-99% at every rung: this is
the rare design where precision does not degrade along the axis of interest.

Recorded in advance, before the run:
  the ladder should return ~5.2x over its dlambda = 1.636, from the nine donor
  cells' own power-form coefficient (-1.007). 4-8x counts as a hit. ~50x means
  the nine-cell coefficient was wrong; ~1x means the construction carries no
  lambda dependence at all. The real-materials pair estimate (-1.9 to -2.7,
  which would imply 25-80x) is NOT the prediction under test: it measures real
  materials, the ladder measures the construction.

Checkpointed per donor. Cheapest rung first, so five of the seven land early
and an interrupt costs at most the rung in flight.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

from build_physics_dataset import _be_polite
from spectral_generator import (load_shapes, tilt_to, a2f_moments, U_GRID,
                                MU_STAR_ME, MU_STAR_AD, CUTOFF_FACTOR,
                                SOLVER_FLOOR_K, FLOOR_FRAC, MAX_MATSUBARA)
from eliashberg import allen_dynes_tc, eliashberg_tc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ladder_donors.csv")
LADDER = ["Hf2Zn", "HfRuSb", "Pb3Y", "ScSe", "AlLa3", "AsIn", "BiPb"]
SOLVER_TOL = 1e-4          # matches floor_at and the rebuilt dataset


def _one(item):
    _be_polite()           # spawned fresh on Windows; make it stick
    tgt, lam, w_log, w_2, key, g_donor, donor_r = item
    r_t = w_2 / w_log
    tc_ad = allen_dynes_tc(lam, w_log, w_2, MU_STAR_AD)
    g = tilt_to(g_donor, r_t)
    if g is None:
        return dict(target=tgt, key=key, donor_r=donor_r, status="tilt_failed",
                    Tc=np.nan, moment_err=np.nan)
    w = U_GRID * w_log
    a2f = 0.5 * lam * w * (g / w_log)
    m = a2f_moments(w, a2f)
    err = max(abs(m["lambda_"] / lam - 1), abs(m["w_log"] / w_log - 1),
              abs(m["w_2"] / w_2 - 1))
    if err > 5e-3:
        return dict(target=tgt, key=key, donor_r=donor_r, status="moment_err",
                    Tc=np.nan, moment_err=float(err))
    tc = eliashberg_tc(w, a2f, MU_STAR_ME, cutoff_factor=CUTOFF_FACTOR,
                       t_guess=tc_ad, t_floor=max(SOLVER_FLOOR_K, FLOOR_FRAC * tc_ad),
                       max_matsubara=MAX_MATSUBARA, tol=SOLVER_TOL)
    ok = np.isfinite(tc) and tc > 0
    return dict(target=tgt, key=key, donor_r=donor_r,
                status="ok" if ok else "solver_fail",
                Tc=float(tc) if ok else np.nan, moment_err=float(err))


def main(workers, limit):
    _be_polite()
    shapes = load_shapes()
    d = pd.read_csv(os.path.join(ROOT, "data/processed/physics_dataset.csv"))
    cells = d[d.formula.isin(LADDER)].drop_duplicates("formula")
    cells = cells.assign(_tc=cells.Tc_AD).sort_values("_tc", ascending=False)  # cheap first

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.target, prev.key.astype(np.int64)))
        print(f"resuming: {len(done)} donor records already on disk", flush=True)

    keys = shapes["key"].astype(np.int64).to_numpy()
    rs = shapes["r"].to_numpy()
    n_shapes = len(shapes) if limit is None else min(limit, len(shapes))
    t_start = time.time()

    for _, c in cells.iterrows():
        tgt, lam, wl, w2 = c.formula, c["lambda"], c.w_log, c.w_2
        work = [(tgt, lam, wl, w2, int(keys[i]), shapes["g"].iloc[i], float(rs[i]))
                for i in range(n_shapes) if (tgt, int(keys[i])) not in done]
        if not work:
            print(f"  {tgt:>8}: complete, skipping", flush=True)
            continue
        print(f"  {tgt:>8}  lam={lam:.3f} r={w2/wl:.3f} Tc_AD={c.Tc_AD:.2f}"
              f"  {len(work)} to do  [{(time.time()-t_start)/60:.0f} min]", flush=True)
        t0, buf, n_ok = time.time(), [], 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, rec in enumerate(ex.map(_one, work, chunksize=4), 1):
                buf.append(rec)
                n_ok += rec["status"] == "ok"
                if len(buf) >= 25 or k == len(work):
                    pd.DataFrame(buf).to_csv(
                        OUT, mode="a", header=not os.path.exists(OUT), index=False)
                    buf = []
                if k % 100 == 0 or k == len(work):
                    print(f"      {k}/{len(work)}  ok={n_ok}  "
                          f"{(time.time()-t0)/k:.2f} s/donor  "
                          f"eta {(time.time()-t0)/k*(len(work)-k)/60:.0f} min", flush=True)
        print(f"    {tgt} done in {(time.time()-t0)/60:.1f} min, n_ok={n_ok}", flush=True)

    summarise()


def summarise():
    if not os.path.exists(OUT):
        return
    dd = pd.read_csv(OUT)
    ok = dd[dd.status == "ok"].copy()
    ok["ln"] = np.log(ok.Tc)
    d = pd.read_csv(os.path.join(ROOT, "data/processed/physics_dataset.csv"))
    lam = d.drop_duplicates("formula").set_index("formula")["lambda"]
    g = ok.groupby("target").agg(n=("Tc", "size"), sd=("ln", lambda v: v.std(ddof=1)))
    g["lam"] = lam.reindex(g.index)
    g = g.sort_values("lam")
    print("\n=== ladder ===")
    print(g.to_string(float_format=lambda v: f"{v:9.5f}"))
    if len(g) >= 3:
        b = np.polyfit(g.lam, np.log(g.sd), 1)
        drop = np.exp(-b[0] * (g.lam.max() - g.lam.min()))
        print(f"\n  d ln sigma / d lambda = {b[0]:+.2f}"
              f"   over dlam={g.lam.max()-g.lam.min():.3f}: {drop:.1f}-fold drop")
        print(f"  prediction was 5.2-fold (hit window 4-8x); "
              f"50x => nine-cell coeff wrong; ~1x => no lambda dependence")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--summarise", action="store_true")
    ns = a.parse_args()
    if ns.summarise:
        summarise()
    else:
        main(ns.workers, ns.limit)
