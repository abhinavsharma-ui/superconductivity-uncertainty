"""Binning is a free parameter. Remove it: for each pair, ln|de| = ln sigma(lam,r)
+ ln|Z|, so regressing ln|de| on (lam, ln(r-1)) recovers the slopes with the
standardised term absorbed into the intercept. Bootstrap over MATERIALS, since
pairs share members and are not independent."""
import numpy as np, pandas as pd
b = pd.read_csv('data/processed/physics_dataset.csv')
d = b[b.is_sc & (b.Tc_AD > 1.0) & np.isfinite(b.ad_error)].reset_index(drop=True)
M = d[['lambda','w_log','w_2']].to_numpy(float)
ae = d.ad_error.to_numpy(float); lam = d['lambda'].to_numpy(); rv = d.w_ratio.to_numpy()

def build(idx, tol=0.10):
    m, a, l, r = M[idx], ae[idx], lam[idx], rv[idx]
    I, J = [], []
    for i in range(len(idx)):
        rel = np.abs(m[i+1:]/m[i] - 1).max(axis=1)
        j = np.where(rel < tol)[0] + i + 1
        I.append(np.full(len(j), i)); J.append(j)
    I, J = np.concatenate(I), np.concatenate(J)
    de = np.abs(a[I] - a[J])
    keep = de > 1e-12
    return (np.log(de[keep]), 0.5*(l[I]+l[J])[keep], 0.5*(r[I]+r[J])[keep])

def fit(y, L, R):
    X = np.c_[np.ones(len(y)), L, np.log(R - 1)]
    return np.linalg.lstsq(X, y, rcond=None)[0]

def qfit(y, L, R, it=60):
    X = np.c_[np.ones(len(y)), L, np.log(R - 1)]
    w = np.ones(len(y)); beta = fit(y, L, R)
    for _ in range(it):                      # IRLS for the median (L1)
        res = np.abs(y - X @ beta); w = 1.0 / np.maximum(res, 1e-3)
        W = X * w[:, None]
        beta = np.linalg.solve(W.T @ X, W.T @ y)
    return beta

full = np.arange(len(d))
y, L, R = build(full)
bo, bq = fit(y, L, R), qfit(y, L, R)
print(f"unbinned, n = {len(y)} pairs from {len(d)} materials, tol 10%")
print(f"  OLS on ln|de|    : lam {bo[1]:+.2f}   p (exponent on r-1) {bo[2]:+.2f}")
print(f"  median (L1) fit  : lam {bq[1]:+.2f}   p {bq[2]:+.2f}")

rng = np.random.default_rng(0); B = []
for _ in range(300):
    idx = np.sort(rng.choice(len(d), len(d), replace=True))
    try:
        yy, LL, RR = build(idx)
        if len(yy) > 40: B.append(fit(yy, LL, RR))
    except Exception:
        pass
B = np.array(B)
print(f"\n  bootstrap over MATERIALS, {len(B)} resamples:")
for k, nm in [(1, "lambda coeff"), (2, "exponent p ")]:
    lo, hi = np.percentile(B[:, k], [2.5, 97.5])
    print(f"    {nm}: {B[:,k].mean():+.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]")

print("\n  tolerance sensitivity (OLS, unbinned):")
for tol in (0.05, 0.10, 0.15, 0.20):
    yy, LL, RR = build(full, tol); bb = fit(yy, LL, RR)
    print(f"    tol {tol:4.0%}  n={len(yy):5d}   lam {bb[1]:+.2f}   p {bb[2]:+.2f}")

print(f"\n  donor nine cells for comparison: lam -1.01, p +1.31")
print(f"  ladder prediction from donor coeff: {np.exp(-1.007*1.636):.3f}x"
      f"  ({1/np.exp(-1.007*1.636):.1f}-fold)")
