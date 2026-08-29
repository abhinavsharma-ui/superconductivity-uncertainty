# physics_dataset.csv provenance

**Rebuilt at `SOLVER_TOL = 1e-4` on 2026-08-30**, 806 materials, 33 min at 3
workers. The previous file carried `eliashberg_tc`'s signature default of
`tol = 2e-3` and is recoverable at `git show 64b75c9:data/processed/physics_dataset.csv`.

What the rebuild changed, old -> new:

    distinct Tc_ME/Tc_AD ratios          255 -> 539   (of 583 defined)
    repeated ad_error values             107 -> 40
    rows sitting on a repeat             435 -> 85
    floor population on a repeat        94.4% -> 19.4%
    median |relative Tc shift|                 3.03e-04
    rows moving more than the old grid step   320 of 583

What it did not change. Every headline number is stable to the third decimal:

    population    med|e| old -> new    RMS old -> new     mean old -> new
    is_sc         0.0355 -> 0.0356     0.0993 -> 0.0993   +0.0341 -> +0.0341
    Tc_AD > 1     0.0205 -> 0.0204     0.0379 -> 0.0379   -0.0142 -> -0.0142

    functional lever RMS/median         2.79x -> 2.79x
    is_sc count                           520 -> 520
    floor population count                304 -> 304

So the quantisation was real, it moved 320 of 583 rows by more than the old grid
step, and it never touched a conclusion. What it did corrupt was the left tail
of `ln|delta ad_error|` in the pair regressions, where exact ties made the
choice of estimator look like a modelling decision -- see `src/resolution_check.py`.

What this does and does not affect:

- Does NOT move the conditional-spread bracket. Rebuilt at 1e-4 on the 304-row
  floor population, sigma is 0.0173 / 0.0301 / 0.0278 / 0.0296 against 0.0172 /
  0.0301 / 0.0278 / 0.0295 before -- a ratio of 1.00 at every matching
  tolerance, because a variance is dominated by differences far above the grid
  step.
- Does NOT move the r exponent, stable across tolerances and estimators.
- DOES dissolve the OLS/L1 disagreement in the pair regression: the lambda
  coefficient gap goes 1.44x -> 1.03x at 10% matching tolerance. The
  disagreement was the quantisation, not a choice of estimator.

Rebuild cost, measured across the whole Tc_AD range rather than extrapolated
from the cheap end:

    Tc_AD band        n      s/mat 2e-3   s/mat 1e-4   ratio
    [0.005, 0.05)    72        19.25        19.99      1.04
    [0.05,  0.30)    97         2.86         2.80      0.98
    [0.30,  1.00)   110         0.58         0.95      1.65
    [1.00,  5.00)   182         0.26         0.27      1.02
    [5.00,   inf)   122         0.19         0.22      1.15

Tightening is essentially free per material, but the full dataset is ~30 min
single core for the primary solve and about the same again for the mu13 solve,
because 72 materials below 0.05 K cost 19 s each and carry three quarters of
the total. A per-material figure taken from the Tc_AD > 1 K population
understates the full rebuild by roughly forty-fold.
