"""
Donor-window sweep for the empirical floor, with per-donor records.

WHAT THIS IS FOR, AND ITS KNOWN DESIGN FLAW
-------------------------------------------
`spectral_generator.floor_at` selects donors by an explicit window on
|ln(r_donor / r_target)|. Sweeping the window is meant to measure how much the
empirical floor depends on that choice.

It does NOT cleanly isolate the window, and the AsZr anomaly is probably that.
For each window the driver draws a FRESH `rng.permutation(pool)[:n_donors]`, so:

  - the ensembles are NOT nested; a narrower window is not a subset of a wider
    one, it is a different random subset of a different pool
  - the reported n is SUCCESSFUL TILTS out of a fixed budget, not pool size,
    so windows with more failures are more heavily selected, and selection is
    not neutral (see section 4 of HANDOFF_v3: it favours wide-support donors)

So window effect, resampling variance and tilt-selection are confounded in the
aggregate spread. Two of the four cells came out non-monotone in the window
(AsZr 0.0035 -> 0.0021 -> 0.0039, Se2V 0.0184 -> 0.0196 -> 0.0165), and "all"
sits BELOW window=0.40 on three of four cells, which nested sampling could not
produce. That is a property of this driver, not necessarily of the tilt.

Hence `--dump-donors`: per-donor (key, donor_r, status, Tc) records let the
overlap between two windows be measured directly, and separate "one outlier
donor" from "a genuinely different distribution". Do that before concluding
anything about the tilt itself.

Usage
-----
    python src/window_sweep.py --dump-donors
    python src/window_sweep.py --n-donors 40 --seed 1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_generator import floor_at, load_shapes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

CELLS = [
    ("CrRh3", 0.4689681433981032, 21.07425730946907, 22.265546945920107),
    ("CoTi", 1.012085030346337, 14.09374594703156, 16.361237861756496),
    ("AsZr", 0.8591682309359172, 12.014062145949564, 14.575283369318766),
    ("Se2V", 1.24959280990693, 8.821060672713381, 13.77356774582734),
]
WINDOWS = [0.05, 0.15, 0.40, None]


def main(n_donors: int, seed: int, dump_donors: bool):
    shapes = load_shapes()
    print(f"{len(shapes)} shapes;  r [{shapes.r.min():.3f}, {shapes.r.max():.3f}]"
          f"  median {shapes.r.median():.3f}\n")
    print(f"{'target':>7}{'r':>7}{'window':>9}{'n_ok':>6}{'fail':>7}"
          f"{'spread':>9}{'donor r':>10}")

    rows, donor_rows = [], []
    for name, lam, wl, w2 in CELLS:
        for win in WINDOWS:
            r = floor_at(lam, wl, w2, shapes, n_donors=n_donors, window=win,
                         seed=seed, verbose=False)
            tag = "all" if win is None else f"{win:.2f}"
            if r is None:
                print(f"{name:>7}{w2 / wl:7.3f}{tag:>9}   too few donors")
                continue
            for d in r.pop("donors"):
                donor_rows.append({"target": name, "window": tag, **d})
            r["target"], r["window_tag"] = name, tag
            rows.append(r)
            print(f"{name:>7}{r['r']:7.3f}{tag:>9}{r['n_donors_used']:6d}"
                  f"{r['fail_frac']:7.0%}{r['spread_lnTc_empirical']:9.4f}"
                  f"{r['donor_r_median']:10.3f}")
        print()

    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "window_sweep.csv"),
                              index=False)
    print(f"wrote {os.path.join(RESULTS, 'window_sweep.csv')}")

    if dump_donors:
        dd = pd.DataFrame(donor_rows)
        dest = os.path.join(RESULTS, "window_sweep_donors.csv")
        dd.to_csv(dest, index=False)
        print(f"wrote {dest}  ({len(dd)} donor records)")

        # the diagnostic the aggregate cannot give: do two windows even share
        # donors, and is any single donor carrying the spread?
        print("\ndonor-set overlap between windows (shared successful donors):")
        for name, _, _, _ in CELLS:
            sub = dd[(dd.target == name) & (dd.status == "ok")]
            tags = list(dict.fromkeys(sub.window))
            sets = {t: set(sub[sub.window == t].key) for t in tags}
            for a in range(len(tags)):
                for b in range(a + 1, len(tags)):
                    ta, tb = tags[a], tags[b]
                    inter = len(sets[ta] & sets[tb])
                    print(f"  {name:>7} {ta:>5} vs {tb:>5}: "
                          f"{inter:3d} shared of {len(sets[ta]):3d}/"
                          f"{len(sets[tb]):3d}")
        print("\nleave-one-out on ln Tc: max |spread change| from dropping one")
        for name, _, _, _ in CELLS:
            for t in dd[dd.target == name].window.unique():
                v = dd[(dd.target == name) & (dd.window == t)
                       & (dd.status == "ok")].Tc.to_numpy()
                if len(v) < 6:
                    continue
                base = np.std(np.log(v), ddof=1)
                loo = [np.std(np.log(np.delete(v, i)), ddof=1)
                       for i in range(len(v))]
                worst = max(loo, key=lambda x: abs(x - base))
                print(f"  {name:>7} {t:>5}: spread {base:.4f} -> "
                      f"{worst:.4f} ({worst / base:5.2f}x) dropping 1 of {len(v)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-donors", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-donors", action="store_true")
    a = ap.parse_args()
    main(a.n_donors, a.seed, a.dump_donors)
