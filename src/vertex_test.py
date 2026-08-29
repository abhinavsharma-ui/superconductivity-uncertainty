"""Does the REAL-material conditional spread also have a minimum near lam=0.83?
The ladder found the U in the donor construction. This asks the same question
of real materials, which owes nothing to the construction."""
import numpy as np, pandas as pd
b = pd.read_csv('data/processed/physics_dataset.csv')

def pairs(d, tol):
    d = d[np.isfinite(d.ad_error)].reset_index(drop=True)
    M = d[['lambda','w_log','w_2']].to_numpy(float)
    ae = d.ad_error.to_numpy(); lam = d['lambda'].to_numpy()
    I, J = [], []
    for i in range(len(d)):
        rel = np.abs(M[i+1:]/M[i] - 1).max(axis=1)
        j = np.where(rel < tol)[0] + i + 1
        I.append(np.full(len(j), i)); J.append(j)
    I, J = np.concatenate(I), np.concatenate(J)
    return pd.DataFrame({"dae": ae[I]-ae[J], "lam": 0.5*(lam[I]+lam[J])})

for pop, plab in [(b.is_sc, "is_sc, n=520"),
                  (b.is_sc & (b.Tc_AD > 1.0), "is_sc & Tc_AD>1, n=304")]:
    for tol in (0.10, 0.20):
        p = pairs(b[pop], tol)
        print(f"\n=== {plab}, matching {tol:.0%}, {len(p)} pairs ===")
        edges = [0.30, 0.60, 0.83, 1.10, 3.00]
        print(f"  {'lambda band':>14}{'pairs':>7}{'cond sd':>10}{'vs ladder':>28}")
        lad = {"[0.30,0.60)": "0.00697 @0.48", "[0.60,0.83)": "0.00324-0.00445",
               "[0.83,1.10)": "0.00289 @0.83 (min)", "[1.10,3.00)": "0.00805-0.01584"}
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = p[(p.lam >= lo) & (p.lam < hi)]
            key = f"[{lo:.2f},{hi:.2f})"
            if len(s) < 8:
                print(f"  {key:>14}{len(s):7d}      too few{'':>10}{lad[key]:>20}")
                continue
            sd = s.dae.std(ddof=1)/np.sqrt(2)
            print(f"  {key:>14}{len(s):7d}{sd:10.5f}{lad[key]:>28}")
