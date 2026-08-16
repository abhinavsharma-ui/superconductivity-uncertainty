"""Floor over a lambda x w_ratio grid, targets chosen so Tc is computable.

The lowest-lambda is_sc rows have Tc_AD ~ 0.04 K, which costs n_cut ~ 88,000 AT
THE ANSWER -- no floor bound helps, the converged solve is the expensive part.
Those targets also sit below the 0.05 K reporting threshold, so they are the
least interesting points in the set. Select on Tc_AD, not on linspace order.
"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, 'src')
import polytope_floor as P
import moment_matched as M
from eliashberg import allen_dynes_tc, eliashberg_tc

d = pd.read_csv('data/processed/physics_dataset.csv')
d = d[d.is_sc & (d.Tc_AD > 1.0)].copy()          # computable in reasonable time
ql = d['lambda'].quantile([1/3, 2/3]).values
qr = d.w_ratio.quantile([1/3, 2/3]).values
d['cell'] = np.digitize(d['lambda'], ql)*3 + np.digitize(d.w_ratio, qr)

rows, t0 = [], time.time()
for cell, g in d.groupby('cell'):
    r = g.iloc[len(g)//2]                        # median member of each cell
    out = P.scan_target(r.formula, r['lambda'], r.w_log, r.w_2, n_grid=14)
    if not out: continue
    # same target through the existing two-Gaussian family, for the ratio
    fam = M.build_family(r['lambda'], r.w_log, r.w_2)
    tad = allen_dynes_tc(r['lambda'], r.w_log, r.w_2, P.MU_STAR_AD)
    f = np.array([eliashberg_tc(m['omega'], m['a2f'], P.MU_STAR_ME,
                                cutoff_factor=10.0, t_guess=tad,
                                t_floor=max(0.005, 0.05*tad),
                                max_matsubara=250_000) for m in fam])
    f = f[(f > 0) & np.isfinite(f)]
    out['gauss_range'] = float(np.log(f.max()/f.min())) if len(f) >= 3 else np.nan
    rows.append(out)
    print(f"      vs two-Gaussian {out['gauss_range']:.4f}  ->  "
          f"{out['range_lnTc']/out['gauss_range']:.2f}x   [{time.time()-t0:.0f}s]",
          flush=True)

o = pd.DataFrame(rows); o.to_csv('results/floor_grid.csv', index=False)
print(f"\n=== {len(o)} cells, {time.time()-t0:.0f}s ===")
print(o[['target','lambda','w_ratio','n_feasible','range_lnTc','gauss_range',
         'n_capped_discarded']].to_string(index=False))
print(f"\nrange(ln Tc) : median {o.range_lnTc.median():.4f}  "
      f"min {o.range_lnTc.min():.4f}  max {o.range_lnTc.max():.4f}")
print(f"as Tc %      : median {100*(np.exp(o.range_lnTc.median())-1):.1f}%  "
      f"max {100*(np.exp(o.range_lnTc.max())-1):.1f}%")
print(f"vs two-Gaussian: median understatement {(o.range_lnTc/o.gauss_range).median():.2f}x")
print(f"\nXie et al. 2021 best moment-based formula: 15.1% RMSE")
