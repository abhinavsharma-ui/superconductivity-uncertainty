"""Is the r-trend a composition effect? Matched-n equalises sample SIZE.
The reachable SETS are r-selected and differ cell to cell. Measure the overlap."""
import numpy as np, pandas as pd
from itertools import combinations
d = pd.read_csv('results/reach_map_donors.csv')
rm = pd.read_csv('results/reach_map.csv').sort_values('r')
sets = {t: set(g.key) for t, g in d.groupby('target')}
order = list(rm.target)

print("pairwise Jaccard / containment of reachable donor sets (r-ordered):")
print(f"{'':>9}" + "".join(f"{t[:7]:>8}" for t in order))
for a in order:
    row = f"{a[:8]:>9}"
    for b in order:
        row += "     -- " if a == b else f"{len(sets[a]&sets[b])/len(sets[a]|sets[b]):8.2f}"
    print(row)

inter = set.intersection(*[sets[t] for t in order])
union = set.union(*[sets[t] for t in order])
print(f"\ncommon to all nine cells : {len(inter)}   (Se2V, the smallest, has {len(sets['Se2V'])})")
print(f"union over all nine      : {len(union)} of 806")
print(f"Se2V subset of CrRh3?    : {len(sets['Se2V'] - sets['CrRh3'])} donors of Se2V's are NOT reachable at CrRh3")

# is the reachable set nested by r?  (each higher-r cell a subset of every lower-r cell)
print("\nnesting test -- donors reachable at cell i but NOT at the next-lower-r cell:")
for i in range(1, len(order)):
    hi, lo = order[i], order[i-1]
    print(f"  {hi:>8} (n={len(sets[hi]):3d}) \ {lo:<8} = {len(sets[hi]-sets[lo]):3d}")

# r-composition of the common set vs each cell's own set
rd = d.drop_duplicates('key').set_index('key').donor_r
print(f"\ncommon-set donor r: median {rd[list(inter)].median():.3f}  "
      f"IQR [{rd[list(inter)].quantile(.25):.3f}, {rd[list(inter)].quantile(.75):.3f}]")
print(f"{'cell':>9}{'n':>6}{'own r med':>11}{'common r med':>14}")
for t in order:
    own = rd[list(sets[t])]
    print(f"{t:>9}{len(sets[t]):6d}{own.median():11.3f}{rd[list(inter)].median():14.3f}")
