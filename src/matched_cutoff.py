"""The cutoff question without the Delta-mu* bookkeeping.

d at FIXED mu* cannot converge as w_c -> infinity: the Coulomb kernel is
constant in (n,m), so its Matsubara sum grows like ln(w_c/T). Only the PAIR
(w_c, mu*(w_c)) is physical. So compare matched pairs instead:

    A = Tc(cf=10, mu* = 0.1293)
    B = Tc(cf=20, mu* = rescale(0.1293, 10 -> 20) = 0.14203)

d' = ln(B/A) is then the residual cutoff error directly -- no constant to
predict or subtract. Small and flat => cf=10 is fine and the question is
closed. This is T7 (which passes at 0.59% on Nb) generalised across lambda
and w_ratio.
"""
import json, os, sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
# ROOT-relative, so this runs the same from any working directory and writes
# its output next to the other results rather than wherever it was launched
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from eliashberg import eliashberg_tc, rescale_mu_star

MU10 = 0.1293
MU20 = rescale_mu_star(MU10, 10.0, 20.0)
MU40 = rescale_mu_star(MU20, 20.0, 40.0)
KW = dict(t_floor=0.005, max_matsubara=250_000, tol=1e-3)

OUT = os.path.join(ROOT, 'results', 'matched_cutoff.csv')
d = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'physics_dataset.csv'))
with open(os.path.join(ROOT, 'data', 'raw', 'bete_database.json')) as fh:
    db = json.load(fh)
s = d[(d.Tc_ME > 0.2)].copy()          # keep it affordable; spans lam and r fully
ql = s['lambda'].quantile([1/3,2/3]).values; qr = s.w_ratio.quantile([1/3,2/3]).values
s['cell'] = np.digitize(s['lambda'],ql)*3 + np.digitize(s.w_ratio,qr)
sel = s.groupby('cell', group_keys=False).apply(
    lambda g: g.sort_values('lambda').iloc[np.linspace(0,len(g)-1,4).astype(int)])

def work(row):
    k = str(row['material_id'])
    w = np.asarray(db['Freq_meV'][k],float); a = np.asarray(db['a2F'][k],float)
    A = eliashberg_tc(w,a,mu_star=MU10,cutoff_factor=10.0,t_guess=row['Tc_ME'],**KW)
    B = eliashberg_tc(w,a,mu_star=MU20,cutoff_factor=20.0,t_guess=row['Tc_ME'],**KW)
    C = eliashberg_tc(w,a,mu_star=MU10,cutoff_factor=20.0,t_guess=row['Tc_ME'],**KW)
    if min(A,B,C) <= 0: return None
    return dict(material_id=row['material_id'], lam=row['lambda'],
                w_ratio=row['w_ratio'], A=A, B=B, C=C,
                d_matched=np.log(B/A), d_fixed=np.log(C/A))

if __name__ == '__main__':
    print(f"mu*: cf10 {MU10:.5f}  cf20 {MU20:.5f} (AM rescale, d={MU20-MU10:+.5f})"
          f"  cf40 {MU40:.5f} (d={MU40-MU20:+.5f})", flush=True)
    t0=time.time()
    with Pool(2) as p:
        out=[r for r in p.map(work,[r for _,r in sel.iterrows()]) if r]
    r=pd.DataFrame(out)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    r.to_csv(OUT, index=False)
    print(f"n={len(r)}  {time.time()-t0:.0f}s\n")
    print(f"d_matched  mean {r.d_matched.mean():+.5f}  median {r.d_matched.median():+.5f}"
          f"  sd {r.d_matched.std():.5f}  range [{r.d_matched.min():+.4f},{r.d_matched.max():+.4f}]")
    print(f"           in Tc%: median {100*(np.exp(r.d_matched.median())-1):+.2f}%"
          f"   max |{100*(np.exp(r.d_matched.abs().max())-1):.2f}|%")
    print(f"d_fixed    mean {r.d_fixed.mean():+.5f}  median {r.d_fixed.median():+.5f}"
          f"  range [{r.d_fixed.min():+.4f},{r.d_fixed.max():+.4f}]")
    from scipy import stats
    for nm,c in (('d_matched','d_matched'),('d_fixed','d_fixed')):
        print(f"\n  {nm}: spearman vs lam {stats.spearmanr(r.lam,r[c]).statistic:+.3f}"
              f"   vs w_ratio {stats.spearmanr(r.w_ratio,r[c]).statistic:+.3f}")
    q=r.lam.quantile([1/3,2/3]).values; lb=np.digitize(r.lam,q)
    print(f"\n{'tertile':<8}{'med lam':>9}{'med d_matched':>15}{'Tc %':>9}{'med d_fixed':>13}")
    for j,lab in enumerate(('low','mid','high')):
        g=r[lb==j]
        print(f"{lab:<8}{g.lam.median():9.3f}{g.d_matched.median():15.5f}"
              f"{100*(np.exp(g.d_matched.median())-1):9.2f}{g.d_fixed.median():13.5f}")
