# crabs_diag_fulldim.R
# (#2a) Two analyses requested by reviewers for the crabs real-data section:
#  (1) Per-group multivariate-normality diagnostics (chi-square QQ + Shapiro)
#      to check whether the species x sex CLASSES are actually elliptical CLUSTERS.
#  (2) Full-dimensional refit (all 5 PCs) of AIC/BIC and the graph estimators,
#      to answer the AE's question of why the main analysis is limited to PC2/PC3.

suppressPackageStartupMessages({ library(GFuse); library(MASS); library(ggplot2) })
set.seed(2025)

data(crabs)
group <- interaction(crabs$sp, crabs$sex)            # B.F, O.F, B.M, O.M
Ylog  <- log(crabs[, 4:8])
pc    <- prcomp(Ylog, scale = TRUE)
X2    <- pc$x[, 2:3]; colnames(X2) <- c("PC2", "PC3")  # main analysis space
X5    <- pc$x[, 1:5]                                    # full-dimensional space

# ============================================================
# 1.  Per-group shape diagnostics in (PC2, PC3)
# ============================================================
cat("=== Per-group multivariate-normality diagnostics in (PC2,PC3) ===\n")
qq_list <- list()
for (g in levels(group)) {
  Xg <- X2[group == g, , drop = FALSE]; ng <- nrow(Xg)
  D2 <- mahalanobis(Xg, colMeans(Xg), cov(Xg))
  qx <- qchisq((seq_len(ng) - 0.5) / ng, df = 2)
  D2s <- sort(D2)
  rqq <- cor(D2s, qx)
  sw1 <- shapiro.test(Xg[, 1])$p.value
  sw2 <- shapiro.test(Xg[, 2])$p.value
  qq_list[[g]] <- data.frame(theo = qx, samp = D2s, group = g)
  cat(sprintf("  %-5s n=%d  QQcor=%.3f  Shapiro-PC2 p=%.3f  Shapiro-PC3 p=%.3f\n",
              g, ng, rqq, sw1, sw2))
}
qq_all <- do.call(rbind, qq_list)
fig <- ggplot(qq_all, aes(theo, samp)) +
  geom_abline(slope = 1, intercept = 0, colour = "firebrick2", linetype = "dashed") +
  geom_point(size = 1, alpha = 0.75, colour = "#2171b5") +
  facet_wrap(~group, scales = "free", ncol = 2) +
  labs(title = "Crabs: per-group chi-square QQ plots in (PC2, PC3) space",
       subtitle = "Points near the dashed line indicate approximate bivariate normality",
       x = expression("Theoretical "*chi[2]^2*" quantile"),
       y = "Ordered squared Mahalanobis distance") +
  theme_bw() + theme(strip.text = element_text(face = "bold"))
if (!dir.exists("images")) dir.create("images")
ggsave("images/rd_crabs_shape_qq.png", fig, width = 8, height = 6.5, dpi = 150)
cat("Saved: images/rd_crabs_shape_qq.png\n\n")

# ============================================================
# 2.  Full-dimensional refit (all 5 PCs)
# ============================================================
log_sum_exp <- function(lp) { m <- max(lp); m + log(sum(exp(lp - m))) }
em_loc_mix <- function(X, K, family = c("gauss","t"), nu = 5,
                       max_iter = 200, tol = 1e-6, n_init = 5) {
  family <- match.arg(family); n <- nrow(X); D <- ncol(X)
  logdens <- function(dm, lds) {
    if (family == "gauss") -0.5*(D*log(2*pi)+lds+dm)
    else lgamma((nu+D)/2)-lgamma(nu/2)-0.5*(D*log(nu*pi)+lds)-((nu+D)/2)*log1p(dm/nu)
  }
  best_ll <- -Inf; best <- NULL
  for (rp in 1:n_init) {
    set.seed(100 + rp)
    mu <- if (K == 1) matrix(colMeans(X), D, 1) else t(X[sample(n, K), , drop = FALSE])
    Sig <- cov(X); pii <- rep(1/K, K); llp <- -Inf; ll <- -Inf
    for (it in 1:max_iter) {
      Si <- solve(Sig); lds <- as.numeric(determinant(Sig, logarithm = TRUE)$modulus)
      DM <- vapply(1:K, function(k){ Xc <- sweep(X,2,mu[,k],"-"); rowSums((Xc%*%Si)*Xc) }, numeric(n))
      lp <- vapply(1:K, function(k) logdens(DM[,k], lds), numeric(n))
      lr <- sweep(lp, 2, log(pmax(pii,1e-12)), "+"); lse <- apply(lr,1,log_sum_exp); ll <- sum(lse)
      resp <- exp(sweep(lr,1,lse,"-")); pii <- colMeans(resp)
      U <- if (family=="gauss") matrix(1,n,K) else (nu+D)/(nu+DM); W <- resp*U
      for (k in 1:K){ sw <- sum(W[,k]); if (sw>1e-10) mu[,k] <- colSums(W[,k]*X)/sw }
      S <- matrix(0,D,D); for (k in 1:K){ Xc <- sweep(X,2,mu[,k],"-"); S <- S+crossprod(Xc*W[,k],Xc) }
      Sig <- S/n + diag(1e-6,D)
      if (abs(ll-llp) < tol) break; llp <- ll
    }
    if (is.finite(ll) && ll > best_ll) { best_ll <- ll; best <- list(ll=ll) }
  }
  best
}
ic_pick <- function(X, family, nu = 5, Kmax = 8) {
  n <- nrow(X); D <- ncol(X); p <- function(K) K*(D+1)-1+D*(D+1)/2
  aic <- bic <- rep(NA, Kmax)
  for (K in 1:Kmax) {
    f <- tryCatch(em_loc_mix(X,K,family,nu), error=function(e) NULL)
    if (!is.null(f)) { aic[K] <- -2*f$ll+2*p(K); bic[K] <- -2*f$ll+p(K)*log(n) }
  }
  c(AIC = which.min(aic), BIC = which.min(bic))
}

cat("=== Full-dimensional crabs analysis (all 5 PCs, d=5; K_true=4) ===\n")
lam <- c(0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.8,1.2,1.8)
for (family in c("gauss","t")) {
  ic <- ic_pick(X5, family)
  cat(sprintf("[%s] classical AIC K=%d, BIC K=%d\n", family, ic["AIC"], ic["BIC"]))
  best_bic <- Inf; best <- NULL
  for (m in 1:3) {
    fit <- if (family=="gauss")
      normalLocOrder(X5, m=m, lambdas=lam, K=8, graphtype="MNN", verbose=FALSE, maxMem=100, maxadmm=80)
    else
      tLocOrder(X5, m=m, lambdas=lam, K=8, df=5, graphtype="MNN", verbose=FALSE, maxMem=100, maxadmm=80)
    bt <- bicTuning(X5, fit)
    cat(sprintf("   %d-NN: K=%d\n", m, bt$result$order))
    if (is.finite(bt$result$bic) && bt$result$bic < best_bic) {
      best_bic <- bt$result$bic; best <- list(m=m, K=bt$result$order, fit=fit, lam=bt$result$lambda)
    }
  }
  for (gt in c("GSF","MST")) {
    fit <- if (family=="gauss")
      normalLocOrder(X5, m=1, lambdas=lam, K=8, graphtype=gt, verbose=FALSE, maxMem=100, maxadmm=80)
    else
      tLocOrder(X5, m=1, lambdas=lam, K=8, df=5, graphtype=gt, verbose=FALSE, maxMem=100, maxadmm=80)
    cat(sprintf("   %s: K=%d\n", gt, bicTuning(X5, fit)$result$order))
  }
  cat(sprintf("   >> adaptive m-NN: m=%s K=%s\n", best$m, best$K))
  if (family == "gauss") {
    idx <- which.min(abs(lam - best$lam)); mu_sel <- best$fit[[idx]]$mu
    cls <- apply(X5, 1, function(x) which.min(colSums((mu_sel - x)^2)))
    cat("   Contingency (adaptive Gaussian full-dim cluster vs true group):\n")
    print(table(Estimated = cls, True = group))
  }
}
cat("\nDONE\n")
