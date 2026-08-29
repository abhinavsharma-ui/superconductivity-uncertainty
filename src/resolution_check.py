import numpy as np, pandas as pd
b = pd.read_csv('data/processed/physics_dataset.csv')
f = b[np.isfinite(b.ad_error)].copy()
f['ratio'] = f.Tc_ME / f.Tc_AD
vc = f.ad_error.round(12).value_counts()
rep = vc[vc > 1]
print(f"distinct ad_error values repeated exactly: {len(rep)}  covering {rep.sum()} rows of {len(f)}")
print(rep.head(10).to_string())
for v in rep.index[:4]:
    s = f[f.ad_error.round(12) == v]
    print(f"\n  ad_error = {v:.12f}   ratio Tc_ME/Tc_AD = {s.ratio.iloc[0]:.12f}   n={len(s)}")
    print(s[['material_id','formula','lambda','w_ratio','Tc_ME','Tc_AD','capped','is_sc']]
          .head(6).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\n=== how many rows sit on a repeated ad_error, by population ===")
mask = f.ad_error.round(12).isin(rep.index)
for lab, m in [("all defined", np.ones(len(f), bool)), ("is_sc", f.is_sc.values),
               ("is_sc & Tc_AD>1", (f.is_sc & (f.Tc_AD > 1.0)).values)]:
    print(f"  {lab:>18}: {int((mask & m).sum()):4d} of {int(m.sum()):4d}"
          f"  ({100*(mask&m).sum()/m.sum():.1f}%)")
print(f"\n  are they capped?  {f[mask].capped.sum()} of {mask.sum()} flagged capped")
print(f"  ratio values seen: {sorted(f[mask].ratio.round(9).unique())[:8]}")
