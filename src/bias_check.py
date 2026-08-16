"""Is the AD-convexity bias factor (den1/den2) the right de-biasing for the
secant sensitivity, or does the SOLVER's convexity differ enough to matter?

Measures the true local slope dlnTc_ME/dmu*_ME by central difference at
mu* = 0.1293 +/- 0.005, and compares B_true = S_secant/S_local against the
closed-form B_AD used in sensitivity_normalise.py.
"""
import json, os, sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, 'src')
from eliashberg import eliashberg_tc

MU, H = 0.1293, 0.005
KW = dict(cutoff_factor=10.0, t_floor=0.005, max_matsubara=250_000, tol=1e-3)

d = pd.read_csv('data/processed/physics_dataset.csv')
db = json.load(open('data/raw/bete_database.json'))
s = d[(d.Tc_ME > 0.05) & (d.Tc_ME_mu13 > 0)].copy()
s = s.sort_values('lambda')
idx = np.unique(np.linspace(0, len(s)-1, 45).astype(int))
sel = s.iloc[idx]

def work(row):
    k = str(row['material_id'])
    w = np.asarray(db['Freq_meV'][k], float); a = np.asarray(db['a2F'][k], float)
    lo = eliashberg_tc(w, a, mu_star=MU-H, t_guess=row['Tc_ME'], **KW)
    hi = eliashberg_tc(w, a, mu_star=MU+H, t_guess=row['Tc_ME'], **KW)
    if lo <= 0 or hi <= 0:
        return None
    return dict(material_id=row['material_id'], lam=row['lambda'],
                w_ratio=row['w_ratio'], Tc_ME=row['Tc_ME'],
                S_local_true=(np.log(hi)-np.log(lo))/(2*H))

if __name__ == '__main__':
    t0 = time.time()
    with Pool(2) as p:
        out = [r for r in p.map(work, [r for _, r in sel.iterrows()]) if r]
    r = pd.DataFrame(out).merge(
        d[['material_id','Tc_ME','Tc_ME_mu13','ad_error']], on='material_id')
    r['S_secant'] = (np.log(r.Tc_ME_mu13)-np.log(r.Tc_ME_x if 'Tc_ME_x' in r else r.Tc_ME))/(0.1840-0.1293)
    den1 = r.lam - MU*(1+0.62*r.lam); den2 = r.lam - 0.1840*(1+0.62*r.lam)
    r['B_AD'] = den1/den2
    r['B_true'] = r.S_secant/r.S_local_true
    r.to_csv('bias_check.csv', index=False)
    print(f"n={len(r)}  {time.time()-t0:.0f}s\n")
    print(f"{'lam':>7}{'S_secant':>10}{'S_local':>10}{'B_true':>8}{'B_AD':>7}{'B_AD/B_true':>13}")
    for _, x in r.iterrows():
        print(f"{x.lam:7.3f}{x.S_secant:10.2f}{x.S_local_true:10.2f}"
              f"{x.B_true:8.3f}{x.B_AD:7.3f}{x.B_AD/x.B_true:13.3f}")
    q = r.lam.quantile([1/3,2/3]).values; lb = np.digitize(r.lam, q)
    print(f"\n{'tertile':<8}{'med lam':>9}{'B_true':>9}{'B_AD':>8}{'over-correction':>17}")
    for j,lab in enumerate(('low','mid','high')):
        g = r[lb==j]
        print(f"{lab:<8}{g.lam.median():9.3f}{g.B_true.median():9.3f}"
              f"{g.B_AD.median():8.3f}{(g.B_AD/g.B_true).median():17.3f}")
