# Theory notes: what this project is actually working with

Written for Track A, but it's really the bridge between Track A and Track B.
Read it alongside Kittel Ch.10 — the connections are marked.

---

## 1. The one object that matters: α²F(ω)

Everything here is downstream of a single function. Not a number — a *function*.

You know the BCS result:

$$T_c = 1.14\,\theta_D\,e^{-1/N(0)V}$$

Look at what it assumes. There is **one** coupling number, N(0)V, and the
attraction is constant inside an energy shell of width ω_D around the Fermi
surface and exactly zero outside. A step function.

> **Kittel link.** That shell is the one the entropy argument in Ch.10 (p.264)
> told you about — the ~10⁻⁴ k_B per atom implying only a thin Fermi-surface
> shell participates. BCS treats that shell as sharp-edged with uniform
> coupling inside.

Real materials aren't like that. Electrons couple to 20 meV phonons differently
than to 5 meV phonons, and there are different *numbers* of phonons at each
frequency. **α²F(ω)** is the honest version: the phonon density of states F(ω),
weighted at each frequency by how strongly Fermi-surface electrons actually
couple to those phonons, α²(ω).

It is dimensionless, and it is the entire electron-phonon story of a material
compressed into one curve. In our dataset Nb has 276 points from 0 to 27.8 meV;
PdH has 895 points out to 89.7 meV.

**α²F is the input to everything below.** λ is not an input. Neither is ω_log.

---

## 2. λ, ω_log, ω₂ are three moments of that curve

This is the point the whole paper hinges on.

$$\lambda = 2\int_0^\infty \frac{\alpha^2F(\omega)}{\omega}d\omega$$

$$\omega_{\log} = \exp\left[\frac{2}{\lambda}\int \frac{\alpha^2F(\omega)\ln\omega}{\omega}d\omega\right]
\qquad
\omega_2 = \left[\frac{2}{\lambda}\int \alpha^2F(\omega)\,\omega\,d\omega\right]^{1/2}$$

Three summary statistics of a whole function.

**Note the 1/ω weighting in λ. Soft phonons count more.** A material with lots
of low-frequency spectral weight gets a large λ even with unremarkable coupling
strength — which is why the strong-coupling materials in our dataset tend to
have very soft modes, often sitting near a lattice instability. AuInPdY has
λ = 2.39 with ω_log = 2.2 meV: enormous coupling, almost no energy scale. Its
Tc is only ~6 K, because Tc needs both.

**λ has a second meaning that is measurable without any superconductivity.**
It's the mass enhancement: electrons dragging a lattice distortion behind them
are effectively heavier,

$$m^* = m(1+\lambda)$$

which shows up in the electronic specific heat coefficient as
γ = γ_band(1+λ).

> **Kittel link.** This is the *normal-state* heat capacity — the linear-in-T
> electronic term, not the exponential e^(−Δ/k_BT) superconducting term from
> p.264. Same chapter, two different measurements, and λ connects them.

---

## 3. Why Tc is an eigenvalue problem

The bit of the code that looks like magic if nobody tells you.

The Eliashberg gap equation is **nonlinear** in the gap function Δ. But we don't
want the gap — we want the temperature at which it first appears. At exactly
T = T_c, Δ → 0. So drop every term beyond first order in Δ. What survives is
linear and homogeneous:

$$\Delta = \mathcal{K}(T)\,\Delta$$

A linear homogeneous equation has a nonzero solution **only if** 𝒦(T) has an
eigenvalue exactly equal to 1. So:

| ρ_max(T) | meaning |
|---|---|
| < 1 | only Δ = 0 solves it → normal metal |
| > 1 | normal state is unstable → a gap opens |
| = 1 | **this defines T_c** |

That's why `_me_eigenvalue()` returns a number and `eliashberg_tc()` bisects
until it hits 1. Same logic as the Stoner criterion for ferromagnetism: find
where the normal state goes unstable.

### Where the matrix comes from

Finite-temperature field theory. Work in imaginary time τ ∈ [0, ℏ/k_BT];
fermions are antiperiodic on that interval, so their Fourier components exist
only at the **Matsubara frequencies**

$$\omega_n = \pi T(2n+1)$$

Δ becomes a vector indexed by n, 𝒦 becomes a matrix, numpy takes it from there.

The full linearized equations, as implemented:

$$\lambda(n-m) = 2\int_0^\infty \frac{\omega\,\alpha^2F(\omega)}{\omega^2 + (\omega_n-\omega_m)^2}d\omega$$

$$Z_n = 1 + \frac{\pi T}{\omega_n}\sum_m \lambda(n-m)\,\mathrm{sgn}(\omega_m)$$

$$Z_n\Delta_n = \pi T\sum_m\left[\lambda(n-m) - \mu^*\right]\frac{\Delta_m}{|\omega_m|}$$

Two things to notice:

**λ(n−m) depends only on the index difference.** That's why the code computes it
once as a 1-D array and indexes it into a matrix, instead of doing N² integrals.

**Z_n fights superconductivity.** It's the same (1+λ) mass enhancement from §2,
and it sits on the left-hand side suppressing Δ. Heavier, slower electrons pair
less effectively. *This is why T_c doesn't grow linearly with λ* — strong
coupling buys you more glue and more sluggishness simultaneously.

### Practical consequence

The number of Matsubara frequencies scales as ω_c/T. Halving T doubles the
matrix dimension and quadruples the cost. That's why `t_floor` exists in the
code: without it, a non-superconducting material sends the bisection toward
T → 0, building 6000×6000 matrices to establish "no". Adding that floor made
the pipeline 24× faster.

---

## 4. μ*, and the trap in it

Track B already gave you the qualitative puzzle: Coulomb repulsion is eV-scale,
phonon attraction is meV-scale. Repulsion should crush pairing by a factor of a
thousand. It doesn't. Why?

**Retardation.** Coulomb repulsion is instantaneous. The phonon attraction is
slow — the lattice takes ~1/ω_D to respond, by which time the first electron is
long gone. Anderson and Morel showed you can exploit the separation of scales:
integrate out the electronic states between the phonon cutoff ω_c and E_F, and
the repulsion surviving at the pairing scale is *logarithmically* reduced:

$$\mu^* = \frac{\mu}{1 + \mu\ln(E_F/\omega_c)}$$

With bare μ ≈ 0.5 and E_F/ω_D ≈ 10²–10³, you get μ* ≈ 0.1. That's where the
famous "μ* = 0.1" comes from. **It is not a measured constant.** It's a screened
parameter everyone assumes.

### The trap

Look at that formula again: **μ\* depends on the cutoff ω_c it's defined at**,
and it *grows* with the cutoff. Quoting "μ* = 0.1" without stating ω_c is like
quoting a voltage without a reference point.

We solve with a cutoff at 10·ω_max and plug in 0.1. Allen and Dynes fitted their
formula against solutions using their own convention. Same symbol, different
physics. And because T_c depends on μ* through an exponential denominator,

$$\lambda - \mu^*(1 + 0.62\lambda)$$

a small mismatch is amplified — worst where λ is small, because there the μ*
term is a larger fraction of the denominator.

**This is exactly the pattern we measured:** at μ* = 0 the Eliashberg solver
agrees with Allen-Dynes to 2–6%; turn μ* on and the gap grows to ~18%, worst at
low λ. `src/diagnose_mustar.py` settles whether a single constant μ* explains
all of it (pure convention → adopt it) or whether it drifts with λ (real
formula error → that's the paper's subject).

---

## 5. What Allen-Dynes actually *is*

This reframes the whole project, so be precise about it.

McMillan (1968) took Nb's measured α²F, solved the Eliashberg equations
numerically, scaled the spectrum around, and **fitted an analytic formula to his
own numerical output**. Allen and Dynes (1975) redid it with many more spectra
and found McMillan's form fails badly at large λ — it *saturates* near
ω_log/1.45 while the true T_c keeps climbing. They patched it with two
correction factors:

$$T_c = \frac{f_1f_2\,\omega_{\log}}{1.2}\exp\left[\frac{-1.04(1+\lambda)}{\lambda - \mu^*(1+0.62\lambda)}\right]$$

- **f₁** — strong-coupling correction, depends on λ
- **f₂** — spectral-shape correction, depends on ω₂/ω_log

So the Allen-Dynes formula is **an interpolation fit, not a theory.** The
Eliashberg equations are the theory. That is precisely why "Allen-Dynes
breakdown" is a meaningful and *measurable* thing: it's a fit failing outside
the region it was fitted in.

### The exact result that anchors everything

As λ → ∞ the Eliashberg equations have an analytic asymptotic solution:

$$k_BT_c \to 0.1827\sqrt{\lambda\langle\omega^2\rangle}$$

T_c grows as **√λ**, without bound. McMillan's exponential saturates instead.
That's the qualitative failure in one line.

It's also the strongest test of our solver, because it's derived independently
and contains no fitted parameters. Ours reproduces it to **0.2% at λ = 100**
(test T4 in `verify_eliashberg.py`). That's the main reason to trust the
machinery even with the μ* convention question still open.

---

## 6. Where this leaves the paper

Allen-Dynes claims three numbers (λ, ω_log, ω₂) determine T_c. But those three
numbers are *moments of a function*. Infinitely many different α²F curves share
the same three moments — and they have **different true T_c**.

That spread is **irreducible**. No model, however good, can predict T_c exactly
from those three inputs, because the inputs genuinely don't contain the answer.
A properly calibrated uncertainty estimate should recover exactly that spread,
and it should be larger wherever the formula is more shape-sensitive.

This is a physical prediction with a mechanism, which is far more defensible
than "uncertainty correlates with λ". It also dissolves the density confound as
the *primary* worry: you're no longer claiming uncertainty is high because data
is sparse, you're claiming it's high because information was discarded — and you
can measure what was discarded, directly from α²F.

### The honest problems

1. **10 of 806 materials have λ ≥ 1.5.** Any threshold test at λ ≈ 1.5 is a test
   on ten points. The continuous breakdown measure, log(T_c^ME/T_c^AD) across
   all 806, is the statistically serious version.

2. **The density confound hasn't gone away, it's changed shape.** Shape-sensitivity
   and data sparsity may still be correlated. Still needs controlling for.

3. **JARVIS supercon_3d cannot serve as ground truth.** Its Tc column is computed
   *from* λ and ω_log by the Allen-Dynes formula. Training on it to study
   Allen-Dynes breakdown is circular — the labels are the formula. Use it to
   cross-check λ against an independent DFT pipeline, nothing more.

---

## 7. Reading queue, reordered

Given where Track A now is, the original Track B order isn't optimal.
Suggested:

1. **Energy Gap (Kittel p.266)** — you need Δ before the gap equation means
   anything. Currently next in your queue anyway. Good.
2. **Allen & Dynes 1975** — read §I and §V (the asymptotic limit). Skip the
   numerical tables. This is the single most relevant paper to what you're
   building and it's readable.
3. **Coherence length** — you flagged it as your weakest area. It matters less
   for this paper than the gap does, so it can wait, but don't skip it.
4. **BCS ground state** — heaviest, still fine to save for last. Eliashberg is
   the generalization of it, so having built the Eliashberg solver first will
   make the BCS wavefunction feel like a special case rather than a mountain.

---

## References

- W.L. McMillan, *Transition Temperature of Strong-Coupled Superconductors*,
  Phys. Rev. **167**, 331 (1968).
- P.B. Allen and R.C. Dynes, *Transition temperature of strong-coupled
  superconductors reanalyzed*, Phys. Rev. B **12**, 905 (1975).
- P.B. Allen and B. Mitrović, *Theory of Superconducting T_c*, Solid State
  Physics **37**, 1 (1982). — the careful derivation
- P. Morel and P.W. Anderson, Phys. Rev. **125**, 1263 (1962). — origin of μ*
- Hennig group, BETE-NET dataset: https://github.com/henniggroup/BETE-NET
