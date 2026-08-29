import numpy as np, pandas as pd
b = pd.read_csv('data/processed/physics_dataset.csv')
d = b[b.is_sc & (b.Tc_AD > 1.0) & np.isfinite(b.ad_error)].copy()
M = d[['lambda','w_log','w_2']].to_numpy(float)
ae = d.ad_error.to_numpy(float); lam = d['lambda'].to_numpy(); rv = d.w_ratio.to_numpy()
ln = np.log(d.Tc_ME.to_numpy(float))

def pairs(tol):
    o = []
    for i in range(len(d)):
        rel = np.abs(M[i+1:]/M[i] - 1).max(axis=1)
        for j in np.where(rel < tol)[0]:
            k = i+1+j
            o.append((ae[i]-ae[k], ln[i]-ln[k], 0.5*(lam[i]+lam[k]), 0.5*(rv[i]+rv[k])))
    return pd.DataFrame(o, columns=['dae','dln','lam','r'])

MY_K = (2*0.6745/np.sqrt(np.pi)) * np.sqrt(2)     # what my code divided by
RIGHT_MED = 0.6745 * np.sqrt(2)                    # median(|D|) = sigma*sqrt2*0.6745
print(f"my divisor {MY_K:.4f}   correct median divisor {RIGHT_MED:.4f}"
      f"   -> my values low by {MY_K/RIGHT_MED:.4f}x\n")
print(f"{'tol':>5}{'pairs':>7}{'sd/sqrt2':>10}{'med(correct)':>14}{'med(mine)':>11}"
      f"{'raw sd dae':>12}{'raw sd dlnTc':>14}")
for tol in (0.03, 0.05, 0.10, 0.20):
    p = pairs(tol); a = p.dae.abs()
    print(f"{tol:5.0%}{len(p):7d}{p.dae.std(ddof=1)/np.sqrt(2):10.4f}"
          f"{a.median()/RIGHT_MED:14.4f}{a.median()/MY_K:11.4f}"
          f"{p.dae.std(ddof=1):12.4f}{p.dln.std(ddof=1):14.4f}")

p = pairs(0.10)
print("\nr split at 10% (sd/sqrt2):")
for nm, s in [("r<1.2", p[p.r < 1.2]), ("r>=1.3", p[p.r >= 1.3])]:
    print(f"  {nm:>7} n={len(s):4d}  {s.dae.std(ddof=1)/np.sqrt(2):.4f}")
lo, hi = p[p.r < 1.2], p[p.r >= 1.3]
print(f"  ratio {hi.dae.std(ddof=1)/lo.dae.std(ddof=1):.2f}x")

print("\n=== lambda slope from pairs: sigma-hat in (lambda, r) bins ===")
for lab, dd in [("is_sc & Tc_AD>1", d), ]:
    pass
rows = []
rb = [1.0, 1.15, 1.25, 1.40, 3.0]
lb = np.quantile(p.lam, [0, .25, .5, .75, 1.0])
for i in range(len(rb)-1):
    for j in range(len(lb)-1):
        s = p[(p.r >= rb[i]) & (p.r < rb[i+1]) & (p.lam >= lb[j]) & (p.lam < lb[j+1])]
        if len(s) >= 15:
            rows.append({"r_mid": s.r.median(), "lam_mid": s.lam.median(),
                         "n": len(s), "sig": s.dae.std(ddof=1)/np.sqrt(2)})
B = pd.DataFrame(rows)
print(B.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
X = np.c_[np.ones(len(B)), B.lam_mid, B.r_mid]
beta, *_ = np.linalg.lstsq(X, np.log(B.sig), rcond=None)
pred = X @ beta; ss = 1 - ((np.log(B.sig)-pred)**2).sum()/((np.log(B.sig)-np.log(B.sig).mean())**2).sum()
print(f"\n  ln sigma = {beta[0]:+.3f} {beta[1]:+.3f}*lambda {beta[2]:+.3f}*r"
      f"   R2={ss:.2f}  ({len(B)} bins)")
print(f"  d ln sigma / d lambda = {beta[1]:+.2f}  -> over the ladder's dlam=1.64: "
      f"{np.exp(beta[1]*1.64):.3f}x  ({1/np.exp(beta[1]*1.64):.1f}-fold drop)")
print(f"  CW's recorded falsifier from the CrRh3/HfRu2Ta pair: -2.4 -> 50-fold")

print("\n=== does the fitted form respect the r=1 boundary? ===")
y = np.log(B.sig)
for name, X_ in [("a + b*lam + c*r        ", np.c_[np.ones(len(B)), B.lam_mid, B.r_mid]),
                 ("a + b*lam + p*ln(r-1)  ", np.c_[np.ones(len(B)), B.lam_mid, np.log(B.r_mid-1)])]:
    be, *_ = np.linalg.lstsq(X_, y, rcond=None)
    r2 = 1 - ((y - X_@be)**2).sum()/((y-y.mean())**2).sum()
    print(f"  {name} R2={r2:.3f}   coeffs {np.array2string(be, precision=3)}")
    if 'ln(r-1)' in name:
        print(f"     -> sigma ~ (r-1)^{be[2]:.2f} exp({be[1]:.2f} lam);  sigma->0 as r->1: "
              f"{'YES' if be[2] > 0 else 'NO'}")
    else:
        print(f"     -> sigma at r=1 is {np.exp(be[0]+be[1]*B.lam_mid.median()+be[2]):.4f}, not 0")
# nine-cell donor floor, same two forms
g9 = pd.read_csv('data/external/grid9.csv')
y9 = np.log(g9.sd)
for name, X_ in [("a + b*lam + c*r        ", np.c_[np.ones(9), g9.lam, g9.r]),
                 ("a + b*lam + p*ln(r-1)  ", np.c_[np.ones(9), g9.lam, np.log(g9.r-1)])]:
    be, *_ = np.linalg.lstsq(X_, y9, rcond=None)
    r2 = 1 - ((y9 - X_@be)**2).sum()/((y9-y9.mean())**2).sum()
    print(f"  [donor 9 cells] {name} R2={r2:.3f}  coeffs {np.array2string(be, precision=3)}")
