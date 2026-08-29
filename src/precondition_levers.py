import numpy as np, pandas as pd
from scipy import stats

# ---------- ATTACK 3: the mu* offset lever, measured ----------
d = pd.read_csv('data/processed/physics_dataset.csv')
print("=== mu* convention lever (the asserted, unmeasured precondition) ===")
print("  AD 0.10 / ME 0.1293   vs   AD 0.13 / ME 0.1840")
for thr, lab in [(0.05, "is_sc"), (1.0, "Tc_ME>1.0")]:
    m = d.Tc_ME > thr
    for col, name in [("ad_error", "mu*=0.10"), ("ad_error_mu13", "mu*=0.13")]:
        a = d.loc[m & np.isfinite(d[col]), col].to_numpy(float)
        print(f"  {lab:>10} {name:>9}  n={len(a):4d}  med|e|={np.median(abs(a)):.4f}"
              f"  RMS={np.sqrt(np.mean(a**2)):.4f}  mean={a.mean():+.4f}")
    both = m & np.isfinite(d.ad_error) & np.isfinite(d.ad_error_mu13)
    a0 = d.loc[both, "ad_error"].to_numpy(float)
    a1 = d.loc[both, "ad_error_mu13"].to_numpy(float)
    print(f"  {lab:>10}  LEVER on matched n={both.sum()}:"
          f"  med|e| {np.median(abs(a1))/np.median(abs(a0)):.2f}x"
          f"   RMS {np.sqrt(np.mean(a1**2))/np.sqrt(np.mean(a0**2)):.2f}x")

# ---------- ATTACK 2: is support-not-weighting general or directional? ----------
print("\n=== reweighting feasibility per cell ===")
dn = pd.read_csv('data/external/donors9.csv'); dn['key'] = dn.key.astype(np.int64)
ok = dn[dn.status == 'ok']
allr = dn.drop_duplicates('key').set_index('key').donor_r
g9 = pd.read_csv('data/external/grid9.csv').sort_values('r')
bins = np.quantile(allr, np.linspace(0, 1, 21)); bins[0] -= 1e-9; bins[-1] += 1e-9
pool_h = np.histogram(allr, bins=bins)[0] / len(allr)
print(f"{'cell':>9}{'r':>7}{'n':>5}{'r range of its set':>22}"
      f"{'pool mass uncovered':>21}{'ESS->pool':>11}{'ESS->Se2V':>11}")
se2v = set(ok[ok.target == 'Se2V'].key)
se2v_h = np.histogram(allr[list(se2v)], bins=bins)[0] / len(se2v)
for _, x in g9.iterrows():
    s = ok[ok.target == x['target']]; rr = allr[list(s.key)]
    h = np.histogram(rr, bins=bins)[0] / len(rr)
    unc = pool_h[h == 0].sum()
    def ess(tgt):
        idx = np.clip(np.digitize(rr, bins) - 1, 0, 19)
        w = np.where(h[idx] > 0, tgt[idx] / np.where(h[idx] > 0, h[idx], 1), 0.0)
        return (w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0
    print(f"{x['target']:>9}{x['r']:7.3f}{len(s):5d}"
          f"   [{rr.min():.3f}, {rr.max():.3f}]{'':>4}{100*unc:19.1f}%"
          f"{ess(pool_h):11.0f}{ess(se2v_h):11.0f}")
