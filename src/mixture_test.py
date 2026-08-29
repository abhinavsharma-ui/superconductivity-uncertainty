import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv(r'data/external/donors9.csv'); d['key'] = d.key.astype(np.int64)
ok = d[d.status == 'ok'].copy(); ok['ln'] = np.log(ok.Tc)
g9 = pd.read_csv(r'data/external/grid9.csv').sort_values('r')
sets = {t: set(g.key) for t, g in ok.groupby('target')}
common = sorted(set.intersection(*sets.values()))

rows = []
for _, x in g9.iterrows():
    t = x['target']; nat = ok[ok.target == t]; com = nat[nat.key.isin(common)]
    rows.append({"target": t, "r": x['r'], "lam": x['lam'], "n": len(nat),
                 "k_full": stats.kurtosis(nat.ln), "k_com": stats.kurtosis(com.ln),
                 "sd_full": nat.ln.std(ddof=1), "sd_com": com.ln.std(ddof=1),
                 "width_sd": nat.donor_r.std(ddof=1),
                 "width_iqr": nat.donor_r.quantile(.75) - nat.donor_r.quantile(.25)})
o = pd.DataFrame(rows); o["drop"] = o.k_full - o.k_com
print(o[["target","r","lam","n","width_sd","width_iqr","k_full","k_com","drop"]]
      .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

print("\n--- CW's mixture prediction, tested every coherent way (spearman, n=9) ---")
for lab, a, b in [("k_full   vs natural-set width", o.k_full, o.width_sd),
                  ("k_full   vs cell's own r",      o.k_full, o.r),
                  ("k_drop   vs natural-set width", o["drop"], o.width_sd),
                  ("k_drop   vs cell's own r",      o["drop"], o.r),
                  ("k_common vs natural-set width", o.k_com, o.width_sd),
                  ("k_common vs cell's own r",      o.k_com, o.r)]:
    s = stats.spearmanr(a, b)
    print(f"  {lab:<32} {s.statistic:+.3f}  p={s.pvalue:.3f}")

print("\n--- naturally composition-matched pairs (Jaccard of reachable sets) ---")
names = list(o.target)
for i in range(9):
    for j in range(i+1, 9):
        a, b = names[i], names[j]
        J = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
        if J >= 0.80:
            ra, rb = o.r[i], o.r[j]; la, lb = o.lam[i], o.lam[j]
            print(f"  {a:>8} / {b:<8} J={J:.2f}  dr={rb-ra:+.3f}  dlam={lb-la:+.3f}"
                  f"   sd {o.sd_full[i]:.5f} -> {o.sd_full[j]:.5f} = {o.sd_full[j]/o.sd_full[i]:.2f}x"
                  f"   | common {o.sd_com[j]/o.sd_com[i]:.2f}x")

print("\n--- deformation, the repair of the mixture story ---")
rows2 = []
for _, x in g9.iterrows():
    t = x['target']; nat = ok[ok.target == t]; com = nat[nat.key.isin(common)]
    dn = np.log(nat.donor_r / x['r']); dc = np.log(com.donor_r / x['r'])
    rows2.append({"target": t, "r": x['r'],
                  "def_med_nat": dn.median(), "def_sd_nat": dn.std(ddof=1),
                  "def_med_com": dc.median(), "def_sd_com": dc.std(ddof=1),
                  "k_full": stats.kurtosis(nat.ln), "k_com": stats.kurtosis(com.ln)})
p = pd.DataFrame(rows2); p["dsd"] = p.def_sd_com - p.def_sd_nat
p["drop"] = p.k_full - p.k_com
print(p.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
for lab, a, b in [("k_drop vs deformation-sd change", p["drop"], p.dsd),
                  ("k_drop vs natural deformation sd", p["drop"], p.def_sd_nat),
                  ("k_full vs natural deformation sd", p.k_full, p.def_sd_nat)]:
    s = stats.spearmanr(a, b); print(f"  {lab:<34} {s.statistic:+.3f} p={s.pvalue:.3f}")

print("\n--- the decisive pair for the mixture story ---")
a, b = ok[ok.target=='CrRh3'], ok[ok.target=='HfRu2Ta']
print(f"  CrRh3 / HfRu2Ta share {len(sets['CrRh3']&sets['HfRu2Ta'])} donors "
      f"of {len(sets['CrRh3'])}/{len(sets['HfRu2Ta'])}  (Jaccard 0.98)")
print(f"  same donors, r 1.057 vs 1.070 (dr=+0.013), lam 0.469 vs 0.742")
print(f"  kappa {stats.kurtosis(a.ln):.2f} vs {stats.kurtosis(b.ln):.2f}"
      f"   = {stats.kurtosis(b.ln)/stats.kurtosis(a.ln):.2f}x on one donor set")
