# Full-graph experiment protocol

The full experiment comprises two graphs × two losses × eight α values × 1,000 repetitions = **32,000 fits**. A separate 64-repeat audit per graph/point compares snapshot gaps of 5 and 20 MCS, adding 2,048 fits. The implementation estimates one center node's neighborhood per dataset; it does not apply an AND/OR symmetrization over all nodewise fits.

## Graphs, data and random streams

RR graphs have N=200, d=3 and independent equiprobable ±0.4 edge couplings. The complete configuration pairing is rejected if it contains a loop or multiple edge. Conditional on simplicity, this samples the uniform labeled regular graph ensemble. Signs agree at both ends of each edge.

The grid has N=225, degree 4, periodic boundaries and +0.2 couplings. Regularization parameters are 0.3 and 0.15 respectively. M=round(αN), with α ∈ {0.2, 0.4, 0.8, 1, 2, 3, 4, 5}.

The root seed is 20260915. `SeedSequence([seed, graph_id, repeat]).spawn(4)` supplies independent streams for the graph, MCMC seed, sample subsets and center nodes. The sampler uses C++ `mt19937_64`. The same graph and pool are reused across α within a repetition. Centers and subsets are selected separately for each α. Both losses use identical data. Thus repeats within each point are independent, but α points and paired losses are correlated.

## Metropolis sampling

The random-site update accepts a spin flip with probability

\[
\min\{1,\exp[-2s_i\sum_jJ^*_{ij}s_j]\}.
\]

One MCS consists of N random-site flip attempts. Each chain starts randomly, burns in for 100,000 MCS and records a snapshot every 5 MCS. The pool contains 5× the largest M: 5,000 configurations for RR and 5,625 for grid. Each α uses M configurations chosen uniformly without replacement.

The paper cites [Abbara et al., Learning performance in inverse Ising problems with sparse teacher couplings](https://arxiv.org/pdf/1912.11591). Appendix C describes 10⁵ MCS burn-in, at least a fivefold pool, and snapshots every few MCS. **The precise gap of 5, reuse of a maximal pool across α, and all seeds are explicit choices of this reimplementation.** They are not claimed to be recovered original settings.

Finite MCMC draws are not exactly iid. The diagnostic evidence is:

| Diagnostic | RR | Grid |
|---|---:|---:|
| Largest absolute lag-1 correlation in gap-5 pool | 0.1007 | 0.0841 |
| After randomly retaining one fifth of snapshots | 0.0405 | 0.0229 |
| Largest classical split-R̂ over both sets of traces | <1.0007 | <1.0007 |

These numbers use four chains initialized randomly, all positive, all negative, and randomly, with 10,000 retained pool snapshots each. Observables include signed bond averages, magnetization, four spins and four bond products. The RR audit uses one saved graph; it is not an audit of every graph in the experiment. Small exact-enumeration tests compare second moments with the complete Gibbs distribution on an 8-node signed regular graph and a 9-node periodic grid.

The gap-20 audit uses the same seeds and graph pairing as the first 64 gap-5 repeats. Across 96 point/metric comparisons, the largest absolute paired mean difference divided by its standard error is 2.64. This is a limited-power diagnostic with multiple comparisons, not proof that thinning has no effect.

## Estimators and optimization

For y=s₀ and X=s∖₀, both objectives are optimized over all N−1 coefficients:

\[
\frac{1}{2M}\|y-XJ\|^2+\lambda\|J\|_1,
\qquad
\frac1M\sum_\mu\log(1+e^{-2y_\mu(XJ)_\mu})+\lambda\|J\|_1.
\]

There is no intercept, centering or standardization. The computational gauge transformation Xᵤⱼ←yᵤXᵤⱼ makes the response one without changing J.

LinR uses coordinate soft-thresholding. LogR uses coordinate Newton updates and line searches. Rare line-search stalls at the floating-point objective floor are polished by solving the active smooth KKT equations and checking all inactive coordinates. Such repeats are not dropped. Every reference fit satisfies a global KKT tolerance of 2×10⁻¹⁰; the largest stored residual is 1.9991×10⁻¹⁰.

The complete checkpoint audit reconstructs every metric from coefficients and true couplings. It also rebuilds training subsets from the first saved sample pool of each graph and recomputes gradients in the original ungauged coordinates. The latter check is independent of the optimizer's stored residual calculation, but is not repeated over every sample pool because only the first pools are bundled.

## Metrics and uncertainty

- RSS is the sum of squared errors against the true signed coupling row.
- The selected support is |Ĵ|>10⁻⁸; the center coefficient is exactly zero.
- Precision is the per-run TP/(TP+FP), with 1 assigned to an empty predicted set.
- Recall is TP/d. Empty predictions and near-zero coefficients are separately recorded.
- Error bars are one standard error of the mean over 1,000 repetitions, not the standard deviation or a 95% confidence interval.
- The alternate ratio-of-means precision uses a delta-method standard error.

Changing the support threshold to 10⁻¹⁰ or 10⁻⁶ changes the selected support in 5 of 32,000 fits. The largest change in a point's mean precision is 0.00033334; recall is unchanged. A zero empirical standard error, such as precision=1 across all repeats, does not establish zero population uncertainty.

## Agreement and limits

The new experiments reproduce the overall published trends, including the grid discrepancy from tree-based replica predictions. At α=5, grid LinR RSS is 0.09020±0.00043 and recall is 0.97050±0.00265. RR LinR at α=0.4 has mean precision 0.78615±0.00768. Some individual points still differ visibly from the published figure.

Original seeds and full output arrays are unavailable. This is an independent reproduction, not a claim of pointwise identity. `wall_seconds_this_invocation` in the reference metadata describes the last resume/aggregation call, not total simulation runtime. Different C++ standard libraries may map RNG outputs to distributions differently even with the same seed.
