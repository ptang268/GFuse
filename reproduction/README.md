# Reproducing the manuscript results

These scripts reproduce the figures and tables in the accompanying paper using
the `GFuse` package. They are kept out of the installed package (via
`.Rbuildignore`) but ship with the GitHub repository for reviewers.

## Setup

```r
# install the package (from the repository root, or via:)
# devtools::install_github("ptang268/GFuse")

# additional packages used by the reproduction scripts:
install.packages(c("ggplot2", "dplyr", "tidyr", "MASS", "patchwork", "gridExtra"))
```

Run each script from within this `reproduction/` directory (figures are written
to a local `images/` folder):

```r
setwd("reproduction")
source("plot_solutionpath_2d.R")
```

The 2020 US county election script downloads its data from a public mirror, so
it requires an internet connection.

## Scripts and what they produce

| Script | Manuscript output |
|---|---|
| `plot_solutionpath_2d.R` | Gaussian solution paths (atom trajectories + K̂(λ)) |
| `plot_solutionpath_t.R` | Student-t solution paths |
| `plot_solutionpath_multinom.R` | Multinomial solution paths |
| `regen_t_nu5.R` | Student-t (ν=5) order-selection bar panels (linear / symmetric) |
| `real_data_crabs.R` | Rock-crab morphometrics: solution paths, K̂(λ), cluster scatter |
| `crabs_diag_fulldim.R` | Crabs per-group normality diagnostics + full-dimensional refit |
| `real_data_election.R` | 2020 US county election: multinomial typology, simplex plots |
| `aic_bic_comparison.R` | Classical AIC/BIC order-selection baselines (simulations + real data) |
| `adaptive_realdata.R` | Adaptive m-NN order selection on the real datasets |

## Notes

* The graph estimators (`normalLocOrder`, `tLocOrder`, `multinomialOrder`) are
  provided by `GFuse`; the scripts call `library(GFuse)`.
* `aic_bic_comparison.R` implements the classical finite-mixture AIC/BIC
  baselines independently (no `GFuse` dependency) for the head-to-head
  comparison.
* The full replicated bar-chart simulations in the manuscript use 100
  replications; `regen_t_nu5.R` uses 50 for tractability. Increase `nrep` in the
  script to match the paper exactly.
* Random seeds are set in each script so figures are reproducible.
