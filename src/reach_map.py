"""Exact pre-solve survivor count for all 806 donors x all 9 floor_grid cells.
Reproduces floor_at's two non-solver failure paths (tilt_failed, moment_err)
with no Eliashberg solve, so it prices n before the exhaustive run finishes."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from spectral_generator import load_shapes, tilt_to, a2f_moments, U_GRID

shapes = load_shapes()
rd = shapes.r.to_numpy()
cells = pd.read_csv('results/floor_grid.csv')
rows, det = [], []
t0 = time.time()
for _, c in cells.iterrows():
    lam, w_log, w_2 = c['lambda'], c.w_log, c.w_2
    r_t = w_2 / w_log
    n_tilt_fail = n_mom_fail = 0
    surv_r = []
    for i in range(len(shapes)):
        g = tilt_to(shapes['g'].iloc[i], r_t)
        if g is None:
            n_tilt_fail += 1; continue
        w = U_GRID * w_log
        a2f = 0.5 * lam * w * (g / w_log)
        m = a2f_moments(w, a2f)
        err = max(abs(m["lambda_"]/lam - 1), abs(m["w_log"]/w_log - 1),
                  abs(m["w_2"]/w_2 - 1))
        if err > 5e-3:
            n_mom_fail += 1; continue
        surv_r.append(rd[i])
        det.append({"target": c.target, "key": shapes['key'].iloc[i],
                    "donor_r": rd[i], "moment_err": err})
    n = len(surv_r)
    rows.append({"target": c.target, "lambda": lam, "r": r_t,
                 "n_reach": n, "reach_pct": 100*n/len(shapes),
                 "tilt_fail": n_tilt_fail, "mom_fail": n_mom_fail,
                 "surv_r_med": float(np.median(surv_r)) if n else np.nan,
                 "rel_sd_s": np.nan})
    print(f"  {c.target:>8} r={r_t:.3f} lam={lam:.3f}  n={n:4d} "
          f"({100*n/len(shapes):5.1f}%)  tiltfail={n_tilt_fail:3d} "
          f"momfail={n_mom_fail:3d}  surv r med={np.median(surv_r):.3f}"
          f"  [{time.time()-t0:.0f}s]", flush=True)

o = pd.DataFrame(rows).sort_values('r')
o.to_csv('results/reach_map.csv', index=False)
pd.DataFrame(det).to_csv('results/reach_map_donors.csv', index=False)
print("\n" + o[['target','lambda','r','n_reach','reach_pct','surv_r_med']]
      .to_string(index=False))
from scipy import stats
print(f"\nspearman(r, n_reach)      = {stats.spearmanr(o.r, o.n_reach).statistic:+.3f}")
print(f"spearman(lambda, n_reach) = {stats.spearmanr(o['lambda'], o.n_reach).statistic:+.3f}")
print(f"spearman(r, surv_r_med - r) = "
      f"{stats.spearmanr(o.r, o.surv_r_med - o.r).statistic:+.3f}")
print(f"\nall-donor r: median {np.median(rd):.3f}  "
      f"IQR [{np.percentile(rd,25):.3f}, {np.percentile(rd,75):.3f}]  "
      f"max {rd.max():.3f}")
