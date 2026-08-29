import numpy as np, pandas as pd
base = pd.read_csv('data/processed/physics_dataset.csv')
K = 2*0.6745/np.sqrt(np.pi)          # median|X1-X2| = K*sd for gaussian
def run(d, label):
    d = d[np.isfinite(d.ad_error)].copy()
    M = d[['lambda','w_log','w_2']].to_numpy(float)
    ae = d.ad_error.to_numpy(float); rv = d.w_ratio.to_numpy()
    print(f"\n--- {label}  n={len(d)} ---")
    print(f"{'tol':>6}{'pairs':>7}{'cond sd':>10}{'r<1.2':>18}{'r>=1.3':>18}")
    for tol in (0.01, 0.02, 0.03, 0.05, 0.10, 0.20):
        rows = []
        for i in range(len(d)):
            rel = np.abs(M[i+1:]/M[i] - 1).max(axis=1)
            for j in np.where(rel < tol)[0]:
                k = i+1+j
                rows.append((abs(ae[i]-ae[k]), 0.5*(rv[i]+rv[k])))
        if len(rows) < 5:
            print(f"{tol:6.0%}{len(rows):7d}     too few"); continue
        p = pd.DataFrame(rows, columns=['dae','r'])
        lo, hi = p[p.r < 1.2], p[p.r >= 1.3]
        f = lambda s: f"{s.dae.median()/K/np.sqrt(2):.4f} (n={len(s)})" if len(s) > 4 else "--"
        print(f"{tol:6.0%}{len(p):7d}{p.dae.median()/K/np.sqrt(2):10.4f}{f(lo):>18}{f(hi):>18}")

run(base[base.is_sc], "is_sc  (the sec-5 baseline population)")
run(base[base.is_sc & (base.Tc_AD > 1.0)], "is_sc & Tc_AD>1.0  (the sec-4 floor population)")
print("\ndonor-ensemble floor, nine cells: 0.00105 - 0.02421,  median 0.00847")
