import numpy as np, pandas as pd
from scipy import stats
K = 2*0.6745/np.sqrt(np.pi)
b = pd.read_csv('data/processed/physics_dataset.csv')
for pop, lab in [(b.is_sc, "is_sc"), (b.is_sc & (b.Tc_AD > 1.0), "is_sc & Tc_AD>1")]:
    d = b[pop & np.isfinite(b.ad_error)].copy()
    M = d[['lambda','w_log','w_2']].to_numpy(float)
    ae = d.ad_error.to_numpy(float); lam = d['lambda'].to_numpy(); rv = d.w_ratio.to_numpy()
    rows = []
    for i in range(len(d)):
        rel = np.abs(M[i+1:]/M[i] - 1).max(axis=1)
        for j in np.where(rel < 0.10)[0]:
            k = i+1+j
            rows.append((abs(ae[i]-ae[k]), 0.5*(lam[i]+lam[k]), 0.5*(rv[i]+rv[k])))
    p = pd.DataFrame(rows, columns=['dae','lam','r'])
    print(f"\n=== {lab}: conditional sd by lambda tertile (moments within 10%) ===")
    q = p.lam.quantile([1/3, 2/3]).values
    for name, s in [(f"lam < {q[0]:.2f}", p[p.lam < q[0]]),
                    (f"{q[0]:.2f}-{q[1]:.2f}", p[(p.lam >= q[0]) & (p.lam < q[1])]),
                    (f"lam > {q[1]:.2f}", p[p.lam >= q[1]])]:
        print(f"  {name:>12}  n={len(s):5d}  cond sd = {s.dae.median()/K/np.sqrt(2):.4f}"
              f"   (median r {s.r.median():.3f})")
    # lambda effect controlling r by restricting to a narrow r band
    band = p[(p.r > 1.15) & (p.r < 1.30)]
    if len(band) > 60:
        qb = band.lam.quantile([.5]).values[0]
        lo, hi = band[band.lam < qb], band[band.lam >= qb]
        print(f"  within r in [1.15,1.30]:  lam<{qb:.2f} n={len(lo)} sd={lo.dae.median()/K/np.sqrt(2):.4f}"
              f"   lam>={qb:.2f} n={len(hi)} sd={hi.dae.median()/K/np.sqrt(2):.4f}"
              f"   ratio {(hi.dae.median()/lo.dae.median()):.2f}x")
