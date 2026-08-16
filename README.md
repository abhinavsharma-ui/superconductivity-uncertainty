# Does Model Uncertainty Track McMillan-Allen-Dynes Breakdown?

Evidence from conventional superconductors.

**Status:** steps 1–3 built and verified. Steps 4–6 (calibrated uncertainty,
density control, hypothesis test) not yet written.

Start with **`docs/theory_notes.md`** — it explains what every object in this
codebase is and why. The code will not make sense without it.

---

## The one thing that changed from the original plan

The handoff proposed training on JARVIS-DFT `supercon_3d` (1,058 materials with
λ, ω_log, μ*, Tc). **That dataset's Tc column is computed from λ and ω_log by
the Allen-Dynes formula itself.** Training a model on it to study Allen-Dynes
breakdown is circular — the labels *are* the formula, so there is nothing for
the formula to be wrong about.

This project instead uses the **BETE-NET** dataset: 806 DFT electron-phonon
calculations that include the full Eliashberg spectral function α²F(ω). Having
α²F means the true Tc can be computed by numerically solving the Eliashberg
equations, which gives a genuine reference that Allen-Dynes can be wrong
*relative to*. Breakdown becomes a directly measured per-material quantity,

    ad_error = log(Tc_Eliashberg / Tc_AllenDynes)

rather than a proxy for λ.

**Known limitation, stated up front:** only 10 of 806 materials have λ ≥ 1.5.
Any threshold test at λ ≈ 1.5 is a test on ten points. The continuous
`ad_error` measure across all 806 is the statistically serious version.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/fetch_data.py           # ~110 MB into data/raw/
```

## Run

```bash
# verify the Eliashberg solver before trusting anything it produces (~2 min)
python src/verify_eliashberg.py

# build the physics dataset  (~2 min single core, faster with --workers)
python src/build_physics_dataset.py --workers 8

# the mu* convention diagnostic
python src/diagnose_mustar.py

# step 2, the composition-only sanity baseline (~3 min)
python src/baseline_uci.py
```

---

## What's here

| file | what it does |
|---|---|
| `docs/theory_notes.md` | **read first** — the physics, and how it maps to the code |
| `src/eliashberg.py` | α²F moments, Allen-Dynes, McMillan, and the numerical linearized Migdal-Eliashberg Tc solver |
| `src/verify_eliashberg.py` | 7 correctness checks on the solver, incl. the analytic strong-coupling limit |
| `src/diagnose_mustar.py` | is the ME-vs-AD gap a μ* convention mismatch or real formula error? |
| `src/build_physics_dataset.py` | step 3 — builds `data/processed/physics_dataset.csv` |
| `src/baseline_uci.py` | step 2 — composition→Tc baseline, with the leakage check |
| `src/fetch_data.py` | downloads BETE-NET + UCI |

### Not yet written

- `src/uncertainty_model.py` — step 4, quantile regression forest / GP with
  calibration curves. **Not plain RF ensemble variance.**
- `src/density.py` — step 5, k-NN density in feature space
- `src/hypothesis_test.py` — step 6, uncertainty vs λ controlling for density

---

## Verification

`verify_eliashberg.py` is not a "does it run" test suite. Each check has a
source of truth independent of this code:

| check | what it establishes |
|---|---|
| T1 | α²F integration reproduces the database's stored λ, ω_log, ω₂ (max rel. dev. 1.6e-3 over 200 materials) |
| T2 | power iteration matches exact eigendecomposition (1.5e-11) |
| T3 | Tc converged w.r.t. Matsubara cutoff (1.5% from 20×→40×) |
| **T4** | **reproduces the analytic λ→∞ limit k_BTc → 0.1827√(λ⟨ω²⟩) to 0.2% at λ=100** |
| T5 | ρ(T) monotone, so the bisection is well posed |
| T6 | agreement with Allen-Dynes at weak coupling — see μ* note below |
| T7 | Tc stable under μ* cutoff rescaling (0.59%) |

**T4 is the one that matters.** It's derived independently of any fitted
formula and contains no free parameters. Passing it to 0.2% is the reason to
trust the solver.

**T6 currently fails** at 18%, and that is expected until the μ* convention is
pinned down — see `diagnose_mustar.py` and §4 of the theory notes. At μ* = 0 the
solver agrees with Allen-Dynes to 2–6%; the disagreement appears only when μ*
is turned on, and is worst at low λ. That is the signature of a cutoff
convention mismatch, not a solver bug.

---

## Data

**BETE-NET** (806 materials, 85 MB) — DFT electron-phonon calculations with full
α²F(ω), λ, ω_log, ω₂. https://github.com/henniggroup/BETE-NET

**UCI Superconductivity** (21,263 compounds, 81 composition features,
experimental Tc) — used only for the step-2 baseline. Contains 5,721 duplicate
formulas, so `baseline_uci.py` reports both a naive random split and a
formula-grouped split to make the leakage inflation visible.

**JARVIS-DFT `supercon_3d`** — optional, `python src/fetch_data.py --jarvis`.
Use for cross-checking λ against an independent DFT pipeline. **Do not use its
Tc column as a training label** (see above).

---

## Conventions

- Energies in **meV**, temperatures in **kelvin**
- μ* = 0.10 primary, 0.13 sensitivity column
- Eliashberg cutoff at 10 × ω_max
- Tc < 0.05 K is treated as non-superconducting (`TC_FLOOR_K`)
