# physics_dataset.csv provenance

The committed `physics_dataset.csv` was built BEFORE `SOLVER_TOL` was made
explicit, so its `Tc_ME` and `Tc_ME_mu13` carry `eliashberg_tc`'s signature
default of `tol = 2e-3`, not the `1e-4` the code now passes.

Consequence, measured in `src/resolution_check.py`: `Tc_ME/Tc_AD` takes 255
distinct values over 583 materials; 435 rows share 107 repeated `ad_error`
values; 94.4% of the `is_sc & Tc_AD > 1.0` population sits on a repeated value.

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
