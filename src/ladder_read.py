import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv('results/ladder_donors.csv')
ok = d[d.status == 'ok'].copy(); ok['ln'] = np.log(ok.Tc)
ds = pd.read_csv('data/processed/physics_dataset.csv').drop_duplicates('formula').set_index('formula')
rows = []
for t, g in ok.groupby('target'):
    rows.append({"target": t, "lam": ds.loc[t, 'lambda'], "r": ds.loc[t, 'w_ratio'],
                 "n": len(g), "sd": g.ln.std(ddof=1), "kurt": stats.kurtosis(g.ln)})
o = pd.DataFrame(rows).sort_values('lam').reset_index(drop=True)
o["prec"] = 100*np.sqrt((2+o["kurt"])/(4*o["n"]))
o["ci_lo"] = o.sd*(1-1.96*o.prec/100); o["ci_hi"] = o.sd*(1+1.96*o.prec/100)
print(o.to_string(index=False, float_format=lambda v: f"{v:9.5f}"))

i = o.sd.idxmin()
print(f"\nminimum at {o.target[i]}, lambda = {o.lam[i]:.3f}, sd = {o.sd[i]:.5f}")
lo, hi = o.iloc[:i+1], o.iloc[i:]
for nm, s in [("descending limb", lo), ("ascending limb", hi)]:
    b = np.polyfit(s.lam, np.log(s.sd), 1)
    print(f"  {nm:>16}: lambda {s.lam.min():.3f}-{s.lam.max():.3f}  "
          f"ratio {s.sd.iloc[-1]/s.sd.iloc[0]:.2f}x   d ln sigma/d lambda = {b[0]:+.2f}")

print(f"\n  a single linear fit through all seven: {np.polyfit(o.lam, np.log(o.sd),1)[0]:+.2f}"
      f"   (meaningless on a U)")
print(f"  quadratic R2: {1-np.polyval(np.polyfit(o.lam,np.log(o.sd),2),o.lam).var(ddof=0)*0:.0f}", end="")
q = np.polyfit(o.lam, np.log(o.sd), 2); pred = np.polyval(q, o.lam); y = np.log(o.sd)
print(f"\r  quadratic in lambda: R2 = {1-((y-pred)**2).sum()/((y-y.mean())**2).sum():.3f}"
      f"   vertex at lambda = {-q[1]/(2*q[0]):.2f}")
lin = np.polyval(np.polyfit(o.lam, y, 1), o.lam)
print(f"  linear    in lambda: R2 = {1-((y-lin)**2).sum()/((y-y.mean())**2).sum():.3f}")

print("\n  is the U bigger than the error bars?")
print(f"    ScSe (min) 95% CI [{o.ci_lo[i]:.5f}, {o.ci_hi[i]:.5f}]")
for k in (0, 6):
    sep = "above" if o.ci_lo[k] > o.ci_hi[i] else "OVERLAPS"
    print(f"    {o.target[k]:>7} 95% CI [{o.ci_lo[k]:.5f}, {o.ci_hi[k]:.5f}]  -> {sep}")
print(f"\n  r is held: [{o.r.min():.3f}, {o.r.max():.3f}], spread {o.r.max()-o.r.min():.3f}")
print(f"  spearman(r, sd) across the ladder = {stats.spearmanr(o.r, o.sd).statistic:+.3f}"
      f"  (must be near zero or r is leaking)")
