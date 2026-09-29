# Reference data dictionary

`data/reference/` is the immutable snapshot used in the README figure. `MANIFEST.json` hashes every other file in that directory with SHA-256. `provenance.json` records the reference archive digest, original solver hashes, and the relationship between those solvers and this public release. Reference validation files retain hashes of the original calculation sources; new checks write current hashes under `outputs/`.

## Theory

`theory/theory.csv` and `theory.json` contain one row per `(graph, loss, alpha)`.

| Field | Meaning |
|---|---|
| `graph`, `loss` | `rr`/`grid`, `lin`/`log` |
| `M`, `alpha`, `mc`, `seed` | Training size, M/N, local MC count, RNG seed |
| `rss`, `precision`, `recall` | Primary predicted metrics |
| `precision_ratio_of_means` | Alternate precision aggregation |
| `*_mc_se` | Conditional local-MC standard error |
| `tp_mean`, `fp_mean`, `empty_probability` | Expected counts and empty-prediction probability |
| `E,F,H,K,Q,R,chi,eta,Gamma,tau,p` | Replica order parameters and stable parametrization |
| `eos_residual`, `kkt_residual` | EOS log-relative error and local optimization residual |

The JSON adds the TP histogram and quadrature settings. `validation.json` and `stein_check.json` contain the original numerical checks.

## Experiments

`experiments/runs.csv` contains 32,000 rows. `summary.csv`/`.json` contain 32 point summaries. `diagnostics/thinning20/` uses the same formats for 64 repeats per point.

| Field | Meaning |
|---|---|
| `graph`, `repeat`, `alpha`, `loss` | Unique key for one fit |
| `seed`, `mcmc_seed`, `center`, `M` | Reproduction and dataset identifiers |
| `rss`, `tp`, `fp`, `precision`, `recall` | Per-fit metrics |
| `empty`, `near_zero` | Empty support and coefficients between 10⁻¹⁰ and 10⁻⁶ |
| `kkt`, `objective`, `sweeps` | Global optimization residual, objective, coordinate sweep count |
| `acceptance`, `attempts` | MCMC acceptance fraction and total attempted flips |

Summary columns contain means and `*_se`, plus `precision_ratio_of_means`, its delta-method SE, `repeats`, and maximum residual/sweep counts. Newton root-polishing steps are not included in the coordinate sweep count.

Each `checkpoints/{graph}_{repeat:04d}.npz` is readable with `numpy.load(..., allow_pickle=False)`:

| Array | Shape / meaning |
|---|---|
| `coefficients` | `[8, 2, N]`, ordered by α then loss (`lin`, `log`) |
| `neighbors` | `[N, d]`, integer neighbor indices |
| `signs` | `[N, d]`, coupling signs; multiply by the graph's coupling magnitude |
| `centers` | `[8]`, selected center per α |
| `rows` | Scalar Unicode JSON containing the 16 per-fit records |
| `graph_pairing_attempts` | Number of full configuration pairings attempted |
| `sample_pool` | `[pool_size, N]`, present only for repeat 0 of each graph |

All coefficients and graphs are included. Other sample pools can be regenerated from seeds on the original software/compiler configuration. The audit also writes a consolidated NPZ under its output directory; this duplicate archive is not stored in Git.

Example:

```python
import json
import numpy as np

with np.load("data/reference/experiments/checkpoints/rr_0000.npz", allow_pickle=False) as z:
    coefficients = z["coefficients"]
    records = json.loads(str(z["rows"]))
    print(coefficients.shape, records[0]["rss"])
```

`config.json` contains solver tolerances, RNG seed, sampling settings and original numerical source hashes. `run_metadata.json` records completion status and the last invocation's time; it is not a benchmark for the total run.

## Diagnostics

`sampling_audit.json` stores initialization choices, observable names, classical split-R̂ and lag-1/2/5/10 correlations. `thinning_sensitivity.csv`/`.json` store gap-20 minus gap-5 paired differences, their standard errors and descriptive z-scores. Interpret them with the limitations in [experiments.md](experiments.md).
