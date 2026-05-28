# aic_bic_comparison.R
#
# Traditional AIC / BIC baselines for finite-mixture order selection,
# matched to the emission model of the proposed graph-fused estimator.
#
# Comparison principle:
#   The proposed method (normalLocOrder / tLocOrder / multinomialOrder) assumes a
#   COMMON covariance matrix shared by all components and selects the order along
#   a fusion-regularisation path by BIC.  The classical baseline here uses the
#   same emission family (Gaussian / Student-t / multinomial) with a common
#   covariance ESTIMATED by EM (the strongest, standard finite-mixture baseline),
#   but DROPS the fusion penalty: it fits an independent K-component finite
#   mixture by EM for each K = 1..K_max and selects the order by AIC or BIC.
#   Thus the difference is the order-selection mechanism (independent EM + IC
#   vs. regularisation path), with covariance estimated properly in both.
#
#   df (to match bicTuning in the gsf package):
#     location (Gaussian / Student-t), common Sigma:
#         p = K*(D+1) - 1 + D*(D+1)/2
#     multinomial (D categories):
#         p = K*D - 1
#   (The Sigma term D*(D+1)/2 is constant in K and does not affect the argmin;
#    it is included for exact reporting.)
#
# Output: console summary tables + aic_bic_results.rds.

suppressPackageStartupMessages({
  library(MASS)
})

# =========================================================
# 0.  EM helpers (FIXED common Sigma = cov(data))
# =========================================================
log_sum_exp <- function(lp) { m <- max(lp); m + log(sum(exp(lp - m))) }

# ---- Location mixture (Gaussian or Student-t), common ESTIMATED Sigma ----
#   family = "gauss" or "t" (nu used only for t).  A single covariance matrix
#   is shared across components and re-estimated each EM iteration (matching the
#   common-covariance assumption of the proposed estimator).
em_loc_mix <- function(X, K, family = c("gauss", "t"), nu = 5,
                       max_iter = 200, tol = 1e-6, n_init = 5) {
  family <- match.arg(family)
  n <- nrow(X); D <- ncol(X)

  logdens <- function(dm, logdetS) {
    if (family == "gauss") {
      -0.5 * (D * log(2 * pi) + logdetS + dm)
    } else {
      lgamma((nu + D) / 2) - lgamma(nu / 2) -
        0.5 * (D * log(nu * pi) + logdetS) -
        ((nu + D) / 2) * log1p(dm / nu)
    }
  }

  best_ll <- -Inf; best_fit <- NULL
  for (rp in seq_len(n_init)) {
    set.seed(100 + rp)
    mu  <- if (K == 1) matrix(colMeans(X), D, 1) else t(X[sample(n, K), , drop = FALSE])
    Sigma <- cov(X)                       # initialise at full-data covariance
    pii <- rep(1 / K, K)
    ll_prev <- -Inf; ll <- -Inf

    for (it in seq_len(max_iter)) {
      Sinv    <- solve(Sigma)
      logdetS <- as.numeric(determinant(Sigma, logarithm = TRUE)$modulus)
      DM <- vapply(seq_len(K), function(k) {
        Xc <- sweep(X, 2, mu[, k], "-"); rowSums((Xc %*% Sinv) * Xc)
      }, numeric(n))                                          # n x K Mahalanobis
      lpdf <- vapply(seq_len(K), function(k) logdens(DM[, k], logdetS), numeric(n))
      lr   <- sweep(lpdf, 2, log(pmax(pii, 1e-12)), "+")
      lse  <- apply(lr, 1, log_sum_exp)
      ll   <- sum(lse)
      resp <- exp(sweep(lr, 1, lse, "-"))

      pii <- colMeans(resp)
      U   <- if (family == "gauss") matrix(1, n, K) else (nu + D) / (nu + DM)
      W   <- resp * U
      for (k in seq_len(K)) {
        sw <- sum(W[, k]); if (sw > 1e-10) mu[, k] <- colSums(W[, k] * X) / sw
      }
      # common covariance update (pooled, weighted)
      S <- matrix(0, D, D)
      for (k in seq_len(K)) {
        Xc <- sweep(X, 2, mu[, k], "-")
        S  <- S + crossprod(Xc * W[, k], Xc)
      }
      Sigma <- S / n
      Sigma <- Sigma + diag(1e-6, D)       # numerical floor

      if (abs(ll - ll_prev) < tol) break
      ll_prev <- ll
    }
    if (is.finite(ll) && ll > best_ll) { best_ll <- ll; best_fit <- list(mu = mu, pii = pii, ll = ll) }
  }
  best_fit
}

# ---- Multinomial mixture (M known per row, theta_k unknown) ----
em_multinom_mix <- function(Y, K, max_iter = 200, tol = 1e-6, n_init = 5) {
  n <- nrow(Y); D <- ncol(Y); M_i <- rowSums(Y)
  best_ll <- -Inf; best_fit <- NULL
  for (rp in seq_len(n_init)) {
    set.seed(100 + rp)
    theta <- matrix(rgamma(D * K, 1), D, K); theta <- sweep(theta, 2, colSums(theta), "/")
    pii <- rep(1 / K, K); ll_prev <- -Inf
    for (it in seq_len(max_iter)) {
      log_pdf <- vapply(seq_len(K), function(k)
        as.vector(Y %*% log(pmax(theta[, k], 1e-12))), numeric(n))
      lr  <- sweep(log_pdf, 2, log(pmax(pii, 1e-12)), "+")
      lse <- apply(lr, 1, log_sum_exp)
      ll  <- sum(lse) + sum(lgamma(M_i + 1)) - sum(lgamma(Y + 1))
      resp <- exp(sweep(lr, 1, lse, "-"))
      pii <- colMeans(resp)
      for (k in seq_len(K)) {
        num <- colSums(resp[, k] * Y); den <- sum(resp[, k] * M_i)
        if (den > 1e-10) theta[, k] <- num / den
        theta[, k] <- pmax(theta[, k], 1e-12); theta[, k] <- theta[, k] / sum(theta[, k])
      }
      if (abs(ll - ll_prev) < tol) break
      ll_prev <- ll
    }
    if (is.finite(ll) && ll > best_ll) { best_ll <- ll; best_fit <- list(theta = theta, pii = pii, ll = ll) }
  }
  best_fit
}

# =========================================================
# 1.  AIC / BIC, df matched to gsf::bicTuning
# =========================================================
aic_bic <- function(ll, K, D, n, family = c("loc", "multinom")) {
  family <- match.arg(family)
  p <- if (family == "loc") K * (D + 1) - 1 + D * (D + 1) / 2 else K * D - 1
  list(aic = -2 * ll + 2 * p, bic = -2 * ll + p * log(n), df = p)
}

scan_K <- function(fit_fun, X, K_seq, D, n, family) {
  res <- lapply(K_seq, function(K) {
    fit <- tryCatch(fit_fun(X, K),
                    error = function(e) { warning("K=", K, ": ", e$message); NULL })
    if (is.null(fit)) return(c(K = K, ll = NA, aic = NA, bic = NA))
    ic <- aic_bic(fit$ll, K, D, n, family)
    c(K = K, ll = fit$ll, aic = ic$aic, bic = ic$bic)
  })
  tab <- as.data.frame(do.call(rbind, res))
  list(table = tab,
       K_aic = tab$K[which.min(tab$aic)],
       K_bic = tab$K[which.min(tab$bic)])
}

# =========================================================
# 2.  Data generators (identical to the solution-path scripts)
# =========================================================
rmix <- function(n, mu_mat, Sigma, pi_vec) {
  K <- ncol(mu_mat); D <- nrow(mu_mat)
  z <- sample.int(K, n, replace = TRUE, prob = pi_vec)
  noise <- matrix(rnorm(n * D), n, D) %*% chol(Sigma)
  for (k in seq_len(K)) { idx <- which(z == k)
    if (length(idx)) noise[idx, ] <- sweep(noise[idx, , drop = FALSE], 2, mu_mat[, k], "+") }
  noise
}
rmix_t <- function(n, mu_mat, Sigma, pi_vec, nu = 5) {
  K <- ncol(mu_mat); D <- nrow(mu_mat)
  z <- sample.int(K, n, replace = TRUE, prob = pi_vec)
  g <- matrix(rnorm(n * D), n, D) %*% chol(Sigma)
  w <- sqrt(rchisq(n, df = nu) / nu); Y <- g / w
  for (k in seq_len(K)) { idx <- which(z == k)
    if (length(idx)) Y[idx, ] <- sweep(Y[idx, , drop = FALSE], 2, mu_mat[, k], "+") }
  Y
}
rmix_multinom <- function(n, theta_mat, pi_vec, M) {
  K <- ncol(theta_mat); z <- sample.int(K, n, replace = TRUE, prob = pi_vec)
  t(sapply(seq_len(n), function(i) as.integer(rmultinom(1, M, theta_mat[, z[i]]))))
}

n_sim <- 400; K_max <- 9
results_sim <- list()

run_loc <- function(tag, X, K_true, family, nu = 5) {
  sc <- scan_K(function(X, K) em_loc_mix(X, K, family = family, nu = nu),
               X, 1:K_max, D = ncol(X), n = nrow(X), family = "loc")
  results_sim[[tag]] <<- list(K_true = K_true, K_aic = sc$K_aic, K_bic = sc$K_bic)
  cat(sprintf("%-10s (K_true=%d): AIC -> K=%d, BIC -> K=%d\n",
              tag, K_true, sc$K_aic, sc$K_bic))
}

cat("\n========== SIMULATION SCENARIOS (common ESTIMATED Sigma) ==========\n")
muA <- cbind(c(-2, -2), c(-2, 2), c(2, -2), c(2, 2))   # symmetric, K=4
muB <- cbind(c(-3, 0), c(0, 0), c(3, 0))               # linear, K=3

set.seed(2025); run_loc("Gauss_A", rmix(n_sim, muA, diag(2), rep(1/4, 4)), 4, "gauss")
set.seed(12);   run_loc("Gauss_B", rmix(n_sim, muB, diag(2), rep(1/3, 3)), 3, "gauss")
set.seed(2025); run_loc("t_A", rmix_t(n_sim, muA, diag(2), rep(1/4, 4), 5), 4, "t")
set.seed(12);   run_loc("t_B", rmix_t(n_sim, muB, diag(2), rep(1/3, 3), 5), 3, "t")

thetaA <- cbind(c(0.20, 0.20, 0.60), c(0.20, 0.60, 0.20),
                c(0.60, 0.20, 0.20), c(0.45, 0.10, 0.45))
thetaB <- cbind(c(0.10, 0.70, 0.20), c(0.40, 0.40, 0.20), c(0.70, 0.10, 0.20))

set.seed(2025); YA <- rmix_multinom(n_sim, thetaA, rep(1/4, 4), 50)
scA <- scan_K(em_multinom_mix, YA, 1:K_max, D = 3, n = n_sim, family = "multinom")
results_sim[["M_A"]] <- list(K_true = 4, K_aic = scA$K_aic, K_bic = scA$K_bic)
cat(sprintf("%-10s (K_true=4): AIC -> K=%d, BIC -> K=%d\n", "M_A", scA$K_aic, scA$K_bic))

set.seed(12); YB <- rmix_multinom(n_sim, thetaB, rep(1/3, 3), 50)
scB <- scan_K(em_multinom_mix, YB, 1:K_max, D = 3, n = n_sim, family = "multinom")
results_sim[["M_B"]] <- list(K_true = 3, K_aic = scB$K_aic, K_bic = scB$K_bic)
cat(sprintf("%-10s (K_true=3): AIC -> K=%d, BIC -> K=%d\n", "M_B", scB$K_aic, scB$K_bic))

# =========================================================
# 3.  Real data
# =========================================================
cat("\n========== REAL DATA (common ESTIMATED Sigma) ==========\n")
results_rd <- list()

# --- Crabs: PC2, PC3 ---
data(crabs)
Xcrab <- prcomp(log(crabs[, 4:8]), scale = TRUE)$x[, 2:3]
sc <- scan_K(function(X, K) em_loc_mix(X, K, family = "gauss"),
             Xcrab, 1:K_max, D = 2, n = nrow(Xcrab), family = "loc")
results_rd[["Crabs_Gauss"]] <- list(K_true = 4, K_aic = sc$K_aic, K_bic = sc$K_bic)
cat(sprintf("Crabs Gauss (K_true=4): AIC -> K=%d, BIC -> K=%d\n", sc$K_aic, sc$K_bic))

sc <- scan_K(function(X, K) em_loc_mix(X, K, family = "t", nu = 5),
             Xcrab, 1:K_max, D = 2, n = nrow(Xcrab), family = "loc")
results_rd[["Crabs_t"]] <- list(K_true = 4, K_aic = sc$K_aic, K_bic = sc$K_bic)
cat(sprintf("Crabs t     (K_true=4): AIC -> K=%d, BIC -> K=%d\n", sc$K_aic, sc$K_bic))

# --- Election (multinomial) — same preprocessing as real_data_election.R ---
url <- paste0("https://raw.githubusercontent.com/",
              "xsarinix/election2020-data/main/countypres_2000-2020.csv")
suppressPackageStartupMessages({ library(dplyr); library(tidyr) })
cat("Loading election data...\n")
raw <- read.csv(url, stringsAsFactors = FALSE)
elec2020 <- raw %>%
  filter(year == 2020, office == "US PRESIDENT") %>%
  mutate(party3 = case_when(party == "DEMOCRAT" ~ "D",
                            party == "REPUBLICAN" ~ "R", TRUE ~ "O")) %>%
  group_by(county_fips, county_name, state_po, party3) %>%
  summarise(votes = sum(candidatevotes, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = party3, values_from = votes, values_fill = 0) %>%
  mutate(total = D + R + O) %>% filter(total >= 500, !is.na(county_fips))
set.seed(2025)
elec2020 <- elec2020[sample(nrow(elec2020), 1000), ]
Y_raw <- as.matrix(elec2020[, c("D", "R", "O")])
props <- sweep(Y_raw, 1, rowSums(Y_raw), "/")
Y_50 <- round(props * 50)
for (i in seq_len(nrow(Y_50))) {
  d <- 50L - sum(Y_50[i, ]); Y_50[i, which.max(Y_50[i, ])] <- Y_50[i, which.max(Y_50[i, ])] + d
}
Y_elec <- Y_50 + 1L
sc <- scan_K(em_multinom_mix, Y_elec, 1:K_max, D = 3, n = nrow(Y_elec), family = "multinom")
results_rd[["Elec_Mult"]] <- list(K_true = NA, K_aic = sc$K_aic, K_bic = sc$K_bic)
cat(sprintf("Election Multinom: AIC -> K=%d, BIC -> K=%d\n", sc$K_aic, sc$K_bic))

# =========================================================
# 4.  Summary
# =========================================================
cat("\n========== FINAL SUMMARY ==========\n--- Simulation ---\n")
cat(sprintf("%-10s %7s %5s %5s\n", "Scenario", "K_true", "AIC", "BIC"))
for (nm in names(results_sim)) { r <- results_sim[[nm]]
  cat(sprintf("%-10s %7d %5d %5d\n", nm, r$K_true, r$K_aic, r$K_bic)) }
cat("\n--- Real data ---\n")
cat(sprintf("%-14s %7s %5s %5s\n", "Dataset", "K_true", "AIC", "BIC"))
for (nm in names(results_rd)) { r <- results_rd[[nm]]
  kt <- if (is.na(r$K_true)) "?" else as.character(r$K_true)
  cat(sprintf("%-14s %7s %5d %5d\n", nm, kt, r$K_aic, r$K_bic)) }

saveRDS(list(sim = results_sim, rd = results_rd), "aic_bic_results.rds")
cat("\nSaved: aic_bic_results.rds\n")
