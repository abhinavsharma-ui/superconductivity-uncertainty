"""The descending limb: synthetic targets at fixed r = 1.19, lambda 0.40-0.60.

Complements the high side. The ladder covered lambda 0.48-2.12 using real
materials as targets; the vertex sits near 0.85 and 41.5% of the declared
population lies below the ladder's lowest rung, so the descending limb is the
part that matters for coverage and it is the part that is missing.

Targets are synthetic. The construction needs a moment triple, not a material,
which is what lets lambda be placed exactly rather than hunted for -- and the
only reason a real-material target was ever used is that the grid cells came
from a population. w_log is set per rung so every target lands at the same
Tc_AD, following the same convention as the high-side rungs; sigma is invariant
to it (measured: identical to 5 dp across a 5.8x range), so this affects
comparability of nothing and legibility of the log.

Cost is set by lambda alone and cannot be bought down by choice of scale:
N_Matsubara ~ w_c / T, and Tc/w_log is a function of lambda at fixed r, so
raising w_log raises w_max and T together and leaves N unchanged.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import brentq

from build_physics_dataset import _be_polite
from spectral_generator import load_shapes, MU_STAR_AD
from eliashberg import allen_dynes_tc
from run_ladder import _one          # identical worker: same tilt, moment check, solve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "lowlam_donors.csv")
R_TARGET = 1.19
TC_AD_TARGET = 5.0                   # K, held equal across rungs
LAMBDAS = [0.60, 0.50, 0.40]         # cheapest first


def w_log_for(lam):
    """w_log giving Tc_AD = TC_AD_TARGET at this lambda and r."""
    f = lambda wl: allen_dynes_tc(lam, wl, R_TARGET * wl, MU_STAR_AD) - TC_AD_TARGET
    return brentq(f, 1.0, 1e5, xtol=1e-9)


def main(workers, limit):
    _be_polite()
    shapes = load_shapes()
    keys = shapes["key"].astype(np.int64).to_numpy()
    rs = shapes["r"].to_numpy()
    n = len(shapes) if limit is None else min(limit, len(shapes))

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.target.astype(str), prev.key.astype(np.int64)))
        print(f"resuming: {len(done)} records on disk", flush=True)

    t_start = time.time()
    for lam in LAMBDAS:
        wl = w_log_for(lam)
        w2 = R_TARGET * wl
        tag = f"lam{lam:.2f}"
        work = [(tag, lam, wl, w2, int(keys[i]), shapes["g"].iloc[i], float(rs[i]))
                for i in range(n) if (tag, int(keys[i])) not in done]
        if not work:
            print(f"  {tag}: complete, skipping", flush=True); continue
        print(f"\n  {tag}  w_log={wl:.2f} meV  Tc_AD={TC_AD_TARGET:.1f} K  "
              f"{len(work)} to do   [{(time.time()-t_start)/60:.0f} min]", flush=True)
        t0, buf, ok = time.time(), [], 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, rec in enumerate(ex.map(_one, work, chunksize=4), 1):
                buf.append(rec); ok += rec["status"] == "ok"
                if len(buf) >= 25 or k == len(work):
                    pd.DataFrame(buf).to_csv(OUT, mode="a",
                                             header=not os.path.exists(OUT), index=False)
                    buf = []
                if k % 100 == 0 or k == len(work):
                    el = time.time() - t0
                    print(f"      {k}/{len(work)}  ok={ok}  {el/k:.2f} s/donor"
                          f"  eta {el/k*(len(work)-k)/60:.0f} min", flush=True)
        print(f"    {tag} done in {(time.time()-t0)/60:.1f} min, n_ok={ok}", flush=True)
    summarise()


def summarise():
    if not os.path.exists(OUT): return
    d = pd.read_csv(OUT); ok = d[d.status == "ok"].copy()
    ok["ln"] = np.log(ok.Tc)
    from scipy import stats
    print("\n=== descending limb ===")
    print(f"{'rung':>10}{'n':>6}{'sd':>10}{'kurt':>8}")
    for t, g in ok.groupby("target"):
        print(f"{t:>10}{len(g):6d}{g.ln.std(ddof=1):10.5f}{stats.kurtosis(g.ln):8.2f}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--summarise", action="store_true")
    ns = a.parse_args()
    summarise() if ns.summarise else main(ns.workers, ns.limit)
