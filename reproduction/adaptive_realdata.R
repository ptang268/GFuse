# adaptive_realdata.R
# Compute the adaptive m-NN selection (argmin modified-BIC over m=1,2,3)
# for the crabs (Gaussian & Student-t) and election (multinomial) datasets,
# and print the per-method mBIC values so the adaptive choice is explicit.

suppressPackageStartupMessages({ library(GFuse); library(MASS); library(dplyr); library(tidyr) })
set.seed(2025)

# helper: minimum BIC value + selected order over the lambda path
bic_summary <- function(fit, y) {
  bt <- bicTuning(y, fit)
  # bicTuning minimises BIC = -2 loglik + df log n; expose the value
  val <- tryCatch(bt$result$bic, error = function(e) NA)
  if (is.null(val) || is.na(val)) {
    # fall back: recompute is not trivial; just report order/lambda
    val <- NA
  }
  list(order = bt$result$order, lambda = bt$result$lambda, bic = val,
       full = bt)
}

cat("================ CRABS ================\n")
data(crabs)
Xc <- prcomp(log(crabs[, 4:8]), scale = TRUE)$x[, 2:3]
lam_c <- c(0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.8,1.2,1.8)

for (emi in c("gauss","t")) {
  cat(sprintf("--- emission: %s ---\n", emi))
  best_bic <- Inf; best_m <- NA; best_K <- NA
  for (m in 1:3) {
    fit <- if (emi=="gauss")
      normalLocOrder(Xc, m=m, lambdas=lam_c, K=8, graphtype="MNN", verbose=FALSE, maxMem=100, maxadmm=80)
    else
      tLocOrder(Xc, m=m, lambdas=lam_c, K=8, df=5, graphtype="MNN", verbose=FALSE, maxMem=100, maxadmm=80)
    bt <- bicTuning(Xc, fit)
    cat(sprintf("  %d-NN: K=%d lambda=%.3f | bicfields: %s\n",
                m, bt$result$order, bt$result$lambda, paste(names(bt$result), collapse=",")))
    print(bt$result)
    bv <- bt$result$bic
    if (!is.null(bv) && is.finite(bv) && bv < best_bic) { best_bic <- bv; best_m <- m; best_K <- bt$result$order }
  }
  cat(sprintf("  >> adaptive m-NN: m=%s, K=%s (min BIC=%.2f)\n", best_m, best_K, best_bic))
}

cat("\n================ ELECTION ================\n")
url <- paste0("https://raw.githubusercontent.com/",
              "xsarinix/election2020-data/main/countypres_2000-2020.csv")
raw <- read.csv(url, stringsAsFactors = FALSE)
elec <- raw %>% filter(year==2020, office=="US PRESIDENT") %>%
  mutate(party3=case_when(party=="DEMOCRAT"~"D", party=="REPUBLICAN"~"R", TRUE~"O")) %>%
  group_by(county_fips, county_name, state_po, party3) %>%
  summarise(votes=sum(candidatevotes, na.rm=TRUE), .groups="drop") %>%
  pivot_wider(names_from=party3, values_from=votes, values_fill=0) %>%
  mutate(total=D+R+O) %>% filter(total>=500, !is.na(county_fips))
set.seed(2025); elec <- elec[sample(nrow(elec), 1000), ]
Yr <- as.matrix(elec[, c("D","R","O")]); pr <- sweep(Yr,1,rowSums(Yr),"/")
Y50 <- round(pr*50)
for (i in seq_len(nrow(Y50))) { d <- 50L-sum(Y50[i,]); Y50[i,which.max(Y50[i,])] <- Y50[i,which.max(Y50[i,])]+d }
Y <- Y50 + 1L
lam_e <- c(0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.8,1.2)
t1 <- c(0.70,0.50,0.38,0.28,0.16,0.08); t2 <- c(0.25,0.44,0.56,0.67,0.79,0.88)
th0 <- rbind(t1,t2); pii0 <- rep(1/6,6)

best_bic <- Inf; best_m <- NA; best_K <- NA
for (m in 1:3) {
  fit <- multinomialOrder(Y, m=m, lambdas=lam_e, theta=th0, pii=pii0,
                          graphtype="MNN", verbose=FALSE, maxMem=150, maxadmm=100)
  bt <- bicTuning(Y, fit)
  cat(sprintf("  %d-NN: K=%d lambda=%.3f\n", m, bt$result$order, bt$result$lambda))
  print(bt$result)
  bv <- bt$result$bic
  if (!is.null(bv) && is.finite(bv) && bv < best_bic) { best_bic <- bv; best_m <- m; best_K <- bt$result$order }
}
cat(sprintf("  >> adaptive m-NN: m=%s, K=%s (min BIC=%.2f)\n", best_m, best_K, best_bic))
cat("\nDONE\n")
