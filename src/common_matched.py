import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv(r'C:\Users\abhinav\Downloads\donors9.csv')
g9 = pd.read_csv(r'C:\Users\abhinav\Downloads\grid9.csv').sort_values('r')
d['key'] = d.key.astype(np.int64)
ok = d[d.status == 'ok'].copy()
ok['ln'] = np.log(ok.Tc)

# --- reproduce grid9 from donors9 before trusting either ---
rep = ok.groupby('target').agg(n=('Tc','size'), sd=('ln', lambda v: v.std(ddof=1)),
                               kurt=('ln', lambda v: stats.kurtosis(v)))
chk = g9.set_index('target').join(rep, rsuffix='_rep')
print("reproduce grid9 from donors9:")
print(f"  max |n - n_rep|        = {int((chk.n - chk.n_rep).abs().max())}")
print(f"  max rel |sd - sd_rep|  = {((chk.sd - chk.sd_rep).abs()/chk.sd).max():.2e}")
print(f"  max rel |kurt - kurt_rep| = {((chk["kurt"]-chk["kurt_rep"]).abs()/chk["kurt"]).max():.2e}")
print(f"  attempts per cell: {sorted(d.groupby('target').size().unique())}"
      f"   duplicate (target,key): {d.duplicated(['target','key']).sum()}")

# --- common set, derived from donors9 itself ---
sets = {t: set(g.key) for t, g in ok.groupby('target')}
inter = set.intersection(*sets.values())
mine = set(pd.read_csv('results/common_donors_194.csv').key.astype(np.int64))
print(f"\ncommon set from donors9 : {len(inter)}")
print(f"my common_donors_194    : {len(mine)}   overlap {len(inter & mine)}"
      f"   (mine-only {len(mine-inter)}, theirs-only {len(inter-mine)})")

common = sorted(inter)
rows = []
for _, r in g9.iterrows():
    s = ok[(ok.target == r["target"]) & (ok.key.isin(common))]
    rows.append({"target": r["target"], "r": r["r"], "lam": r["lam"],
                 "n_full": int(r["n"]), "sd_full": r["sd"], "kurt_full": r["kurt"],
                 "n_c": len(s), "sd_c": s.ln.std(ddof=1),
                 "kurt_c": stats.kurtosis(s.ln),
                 "donor_r_full": ok[ok.target == r["target"]].donor_r.median(),
                 "donor_r_c": s.donor_r.median()})
o = pd.DataFrame(rows)
o["ratio"] = o.sd_c / o.sd_full
o["prec_c"] = np.sqrt((2 + o.kurt_c) / (4 * o.n_c))

print(f"\n{'cell':>9}{'r':>7}{'n':>6}{'sd full':>10}{'sd@common':>11}{'x':>7}"
      f"{'k full':>8}{'k comm':>8}{'prec':>7}")
for _, x in o.iterrows():
    print(f"{x['target']:>9}{x['r']:7.3f}{x['n_c']:6d}{x['sd_full']:10.5f}{x['sd_c']:11.5f}"
          f"{x['ratio']:7.2f}{x['kurt_full']:8.2f}{x['kurt_c']:8.2f}{100*x['prec_c']:6.1f}%")

def part(y, x, z):
    ry, rx, rz = (stats.rankdata(v) for v in (y, x, z))
    A = np.c_[np.ones(len(rz)), rz]
    res = lambda a: a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rr = stats.pearsonr(res(ry), res(rx))[0]
    df = len(y) - 3
    return rr, 2 * stats.t.sf(abs(rr * np.sqrt(df / (1 - rr**2))), df)

print("\n                          full n        common 194")
for lab, f_, c_ in [
    ("spearman(sd, r)", stats.spearmanr(o.sd_full, o.r), stats.spearmanr(o.sd_c, o.r)),
    ("spearman(sd, lambda)", stats.spearmanr(o.sd_full, o.lam), stats.spearmanr(o.sd_c, o.lam))]:
    print(f"  {lab:<22} {f_.statistic:+.3f} p={f_.pvalue:.4f}   "
          f"{c_.statistic:+.3f} p={c_.pvalue:.4f}")
for lab, a, b in [("partial(sd, r | lambda)", part(o.sd_full, o.r, o.lam), part(o.sd_c, o.r, o.lam)),
                  ("partial(sd, lambda | r)", part(o.sd_full, o.lam, o.r), part(o.sd_c, o.lam, o.r))]:
    print(f"  {lab:<22} {a[0]:+.3f} p={a[1]:.4f}   {b[0]:+.3f} p={b[1]:.4f}")

print(f"\nspread of the floor across cells:  full n {o.sd_full.max()/o.sd_full.min():.1f}x"
      f"   common {o.sd_c.max()/o.sd_c.min():.1f}x")
print(f"donor_r median: full {o.donor_r_full.min():.3f}-{o.donor_r_full.max():.3f}"
      f"   common {o.donor_r_c.min():.3f}-{o.donor_r_c.max():.3f}")
o.to_csv('results/common_matched.csv', index=False)

print("\n--- kappa gradient: composition or mechanism? ---")
print(f"  spearman(r, kappa)  full n {stats.spearmanr(o.r, o.kurt_full).statistic:+.3f}"
      f"   common {stats.spearmanr(o.r, o.kurt_c).statistic:+.3f}")
print(f"  HfRu2Ta/Se2V kappa  full n {o.kurt_full.iloc[1]/o.kurt_full.iloc[8]:.2f}x"
      f"     common {o.kurt_c.iloc[1]/o.kurt_c.iloc[8]:.2f}x")

print("\n--- LOO on partial(sd, lambda | r), common set ---")
for i, t in enumerate(o.target):
    s = o.drop(i)
    v, p = part(s.sd_c, s.lam, s.r)
    rv, rp = part(s.sd_c, s.r, s.lam)
    print(f"  drop {t:>8}: lam|r = {v:+.3f} p={p:.3f}    r|lam = {rv:+.3f} p={rp:.4f}")

print("\n--- does the sd ratio track the composition shift? ---")
sh = o.donor_r_c - o.donor_r_full
print(f"  spearman(donor_r shift, sd ratio) = {stats.spearmanr(sh, o.ratio).statistic:+.3f}"
      f"  p={stats.spearmanr(sh, o.ratio).pvalue:.4f}")
for _, x in o.iterrows():
    print(f"  {x['target']:>9}  shift {x['donor_r_c']-x['donor_r_full']:+.3f}"
          f"   sd x{x['ratio']:.2f}")
