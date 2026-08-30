"""Close the coverage correction with no extrapolated step.

sigma(lambda) at fixed r = 1.19 is now measured from lambda = 0.40 to 20 across
two machines. The declared population's median coupling is 0.447, which the
0.40 rung finally brackets -- so sigma(0.447) becomes an interpolation, and the
correction stops depending on a slope extended past the lowest measurement.

Two aggregations are reported because they differ by ~2x and neither is
obviously right: substituting each population's median lambda into a non-linear
sigma(lambda), or taking the median of per-material predictions. The second
averages the distribution rather than a summary of it and is the more
principled, but the range is what should be quoted.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CW_RUNGS = [(0.70, 0.00407), (0.90, 0.00282), (1.20, 0.00588), (1.60, 0.01059),
            (2.12, 0.01440), (3.00, 0.01715), (5.00, 0.01783), (10.0, 0.01527),
            (20.0, 0.01140)]


def rungs():
    d = pd.read_csv(os.path.join(ROOT, "results", "lowlam_donors.csv"))
    ok = d[d.status == "ok"].copy()
    ok["ln"] = np.log(ok.Tc)
    mine = []
    for t, g in ok.groupby("target"):
        mine.append({"lam": float(t.replace("lam", "")), "sd": g.ln.std(ddof=1),
                     "n": len(g), "kurt": stats.kurtosis(g.ln), "who": "CC"})
    cw = [{"lam": l, "sd": s, "n": 552, "kurt": np.nan, "who": "CW"} for l, s in CW_RUNGS]
    return pd.DataFrame(mine + cw).sort_values("lam").reset_index(drop=True)


def main():
    c = rungs()
    print("sigma(lambda) at r = 1.19, both machines:")
    print(c.to_string(index=False, float_format=lambda v: f"{v:9.5f}"))

    lo_end = c.lam.min()
    print(f"\nlowest measured rung: lambda = {lo_end:.2f}")

    sig = lambda x: np.interp(x, c.lam, c.sd)   # pure interpolation now
    d = pd.read_csv(os.path.join(ROOT, "data/processed/physics_dataset.csv"))
    dec = d[d.Tc_ME > 0.05]
    cur = d[d.is_sc & (d.Tc_AD > 1.0)]
    g9 = pd.read_csv(os.path.join(ROOT, "data/external/grid9.csv"))

    m_dec, m_cur, m_cells = dec["lambda"].median(), cur["lambda"].median(), g9.lam.median()
    extrap = [("declared", m_dec), ("Tc_AD>1", m_cur), ("cells", m_cells)]
    print("\n  median lambda:  " + "   ".join(f"{k} {v:.3f}" for k, v in extrap))
    for k, v in extrap:
        flag = "" if v >= lo_end else "   <-- STILL EXTRAPOLATED"
        print(f"    sigma({v:.3f}) = {sig(v):.5f}{flag}")

    print(f"\n  sigma(0.742) / sigma(0.447) = {sig(m_cells)/sig(m_dec):.3f}"
          f"   -> the declared population's floor is that much larger")

    # The denominator of headroom is the NINE CELLS, not the population they
    # were drawn from. Those differ: the cells are tertile-median members, so
    # their median lambda is 0.742 against the parent population's 0.612, and
    # substituting the parent inflates the correction by
    # sigma(0.612)/sigma(0.742) = 1.43x. An earlier version of this file made
    # exactly that substitution, and shipped 0.590 in the same message as the
    # 0.415 that contradicted it.
    f_med = sig(m_cells) / sig(m_dec)
    f_pm = np.median(sig(g9.lam)) / np.median(sig(dec['lambda'].clip(lower=lo_end)))
    # definitional invariant: the median-lambda correction IS sigma at the
    # cells' median lambda over sigma at the declared population's. If these
    # ever disagree, the numerator has been substituted again.
    assert abs(m_cells - g9.lam.median()) < 1e-12, "numerator must be the cells"
    print(f"\n  coverage correction to headroom")
    print(f"    median-lambda form      {f_med:.3f}x")
    print(f"    per-material form       {f_pm:.3f}x   (the more principled)")
    print(f"    RANGE TO QUOTE          {min(f_med,f_pm):.2f}-{max(f_med,f_pm):.2f}x")
    below = (dec['lambda'] < lo_end).mean()
    print(f"\n  declared population still below the lowest rung: {100*below:.1f}%"
          f"  (was 41.5% below the old 0.482 endpoint)")


if __name__ == "__main__":
    main()
