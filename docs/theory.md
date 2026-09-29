# Replica implementation notes

The theory module follows main-text Eqs. (19), (21), (25), and (26), and supplementary Eqs. (67), (100), and (109) of the [paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/31917677a66c6eddd3ab1f68b0679e2f-Paper.pdf) and [supplement](https://proceedings.neurips.cc/paper_files/paper/2021/file/31917677a66c6eddd3ab1f68b0679e2f-Supplemental.pdf).

## Spectrum and finite-size surrogate

`Spectrum` integrates the Kesten–McKay adjacency spectrum and maps adjacency eigenvalues to covariance eigenvalues using Eq. (109). The default quadrature is 192-point Gauss–Legendre in adjacency angle. The grid theory uses d=4 in this RR/tree construction, not the exact covariance spectrum or Gibbs marginals of the periodic grid.

`LocalLinear` implements finite-M star samples for Eq. (25). After the gauge transformation xⱼ=s₀sⱼ, the d local spins are independent with mean tanh K₀. Instead of explicitly constructing all M rows, it samples multinomial counts over the 2ᵈ local configurations and conditional Gaussian sums with variance equal to the counts. This is an exact sufficient-statistic sampling scheme for the surrogate distribution; it does not replace a finite-sample Gram matrix by its population expectation.

`LocalLogistic` samples the finite-M star explicitly and uses `logistic.cpp` to optimize the Moreau envelope in Eq. (100). Both methods solve d-dimensional local surrogate problems. They are separate from the N−1-dimensional regressions in `experiments.py`.

## H-update discrepancy in the printed algorithms

Algorithm 2, line 16, and Algorithm 3, line 21, print R/E² and a negative final term in the H update. The formal EOS in main-text Eq. (19) and supplementary Eq. (67) instead gives

\[
H=\frac{R}{\eta^2}+F\widetilde\Lambda
 +\frac{(-ER+F\eta)E}{\int \rho(\gamma)/(\widetilde\Lambda-\gamma)^2\,d\gamma}.
\]

The implementation follows this formal EOS: **R/η² and the positive final term**. This choice is explicit and independently checked by substituting the stabilized equations into the original spectral relations.

## Stable parametrization

Let τ=λM/√(HN), p=erfc(τ/√2), and

\[
h_2=(1+\tau^2)p-\sqrt{2/\pi}\,\tau e^{-\tau^2/2}.
\]

Write Γ=−1/Λ̃ and solve

\[
\left\langle\frac{\Gamma\gamma}{1+\Gamma\gamma}\right\rangle_\rho=p.
\]

For f=1/(1+Γγ), define A=⟨f⟩, D=⟨f²⟩,

\[
k=\frac{\langle\gamma f\rangle}{A},\quad
B=\frac{\langle\gamma f^2\rangle}{D},\quad
V=\frac{\operatorname{Var}(\gamma f)}{A^2D},\quad
C=\frac{\Gamma^2\operatorname{Var}(\gamma f)}{D}.
\]

Then

\[
K=Ek,\quad\chi=p/E,\quad\eta=p/K,\quad R=Hh_2/K^2,
\]

\[
F=H(1-h_2V/k^2)/B,\qquad Q=RB+(F/E^2)C.
\]

This avoids subtracting nearly equal terms when p is small. LinR uses E=α−p and solves a scalar root for log H. LogR solves two equations for log E and log H. Local random samples are held fixed throughout each self-consistency solve.

## LogR score and Stein variance reduction

For each local sample, aᵤ=xᵤᵀJ+√Q zᵤ, and the scalar proximal solution obeys

\[
y_\mu-a_\mu=\chi(1-\tanh y_\mu).
\]

The raw score in Algorithm 3 becomes noisy as Q approaches zero. Define

\[
h_\mu=\frac{1-\tanh^2 y_\mu}{1+\chi(1-\tanh^2 y_\mu)},\qquad
A_S=X_S^T\operatorname{diag}(h)X_S,
\]

where S is the optimized active set. Gaussian integration by parts, including the response of the optimized coefficients, gives

\[
E/\alpha=\mathbb E\left[h_\mu-h_\mu^2x_{\mu,S}^TA_S^{-1}x_{\mu,S}\right].
\]

The correction term is included; using only hᵤ would omit the coefficient response. Tests compare this total derivative with finite differences and compare local optimization with an independent SciPy solver. The reference `stein_check.json` also records an independent raw-score/Stein statistical comparison. LinR retains the paper's E update α/(1+χ).

## Metrics and precision conventions

RSS is the mean active-set squared error plus R, without an additional (N−1−d)/N factor. Recall is E[T]/d. Conditional on the fitted order parameters, the inactive false-positive count is

\[
B_{\rm FP}\sim\operatorname{Binomial}(N-1-d,p).
\]

For the primary precision, the code sums over this binomial count exactly and averages over the simulated active count T:

\[
P_{26}=\mathbb E\left[\frac{T}{T+B_{\rm FP}}\right].
\]

Empty predictions have precision 1. The alternative is

\[
P_{\rm ratio}=\frac{\mathbb E[T]}{\mathbb E[T]+\mathbb E[B_{\rm FP}]}.
\]

They can differ substantially at finite N. Visual proximity of an alternative theoretical curve to a published curve does not establish which statistic the original experiment code used. The full-graph experiments use the per-run mean as their primary statistic.

## Reference accuracy

The reference theory uses seed 202109, adjusted by M and graph type in `run_figure2.py`, and 30,000/8,000 local MC repetitions for LinR/LogR. Maximum EOS log-relative residual: 2.685×10⁻¹¹. Maximum local KKT residual: 2.636×10⁻¹¹.

Seven theory tests cover spectrum normalization/moments, substitution into the original EOS, independent local optimizers, precision averaging, total Stein derivatives, and quadrature convergence. CSV Monte Carlo standard errors are conditional on the shared fitted order parameters; they do not include full self-consistency uncertainty.
