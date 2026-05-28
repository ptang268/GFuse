# regen_t_nu5.R
# Regenerate the two Student-t bar-chart panels (linear Model 4, symmetric Model 5)
# at nu=5 (heavier tails) for the consolidated simulation figure.
# 50 replications, n=400, all 8 methods; matched to the existing "type6" style.

suppressPackageStartupMessages({ library(GFuse); library(ggplot2) })

n <- 400; nu <- 5; Kstart <- 10; nrep <- 50
lambdas <- seq(1e-8, n^(-1/4) * log(n), length.out = 12)

rmix_t <- function(n, mu, Sigma, nu) {
  Kc <- ncol(mu); p <- nrow(mu); L <- chol(Sigma)
  z <- sample.int(Kc, n, TRUE, rep(1/Kc, Kc))
  t(vapply(1:n, function(i) mu[, z[i]] + drop(rnorm(p) %*% L) * sqrt(nu / rchisq(1, nu)), numeric(p)))
}

log_sum_exp <- function(lp) { m <- max(lp); m + log(sum(exp(lp - m))) }
em_t <- function(X, Kc, nu = 5, max_iter = 150, tol = 1e-6, n_init = 3) {
  n <- nrow(X); D <- ncol(X); best <- -Inf
  for (rp in 1:n_init) { set.seed(rp * 7 + Kc)
    mu <- if (Kc == 1) matrix(colMeans(X), D, 1) else t(X[sample(n, Kc), , drop = FALSE])
    Sig <- cov(X); pii <- rep(1/Kc, Kc); llp <- -Inf; ll <- -Inf
    for (it in 1:max_iter) {
      Si <- solve(Sig); lds <- as.numeric(determinant(Sig, logarithm = TRUE)$modulus)
      DM <- vapply(1:Kc, function(k){Xc<-sweep(X,2,mu[,k],"-"); rowSums((Xc%*%Si)*Xc)}, numeric(n))
      lp <- vapply(1:Kc, function(k) lgamma((nu+D)/2)-lgamma(nu/2)-0.5*(D*log(nu*pi)+lds)-((nu+D)/2)*log1p(DM[,k]/nu), numeric(n))
      lr <- sweep(lp,2,log(pmax(pii,1e-12)),"+"); lse <- apply(lr,1,log_sum_exp); ll <- sum(lse)
      resp <- exp(sweep(lr,1,lse,"-")); pii <- colMeans(resp)
      U <- (nu+D)/(nu+DM); W <- resp*U
      for (k in 1:Kc){ sw<-sum(W[,k]); if(sw>1e-10) mu[,k]<-colSums(W[,k]*X)/sw }
      S <- matrix(0,D,D); for(k in 1:Kc){Xc<-sweep(X,2,mu[,k],"-"); S<-S+crossprod(Xc*W[,k],Xc)}; Sig <- S/n+diag(1e-6,D)
      if (abs(ll-llp) < tol) break; llp <- ll
    }
    if (is.finite(ll) && ll > best) best <- ll
  }
  best
}
ic_pick <- function(X, nu, Kmax = 8) {
  n <- nrow(X); D <- ncol(X); p <- function(K) K*(D+1)-1+D*(D+1)/2
  a <- b <- rep(NA, Kmax)
  for (K in 1:Kmax) { f <- tryCatch(em_t(X,K,nu), error=function(e) NA)
    if (is.finite(f)) { a[K] <- -2*f+2*p(K); b[K] <- -2*f+p(K)*log(n) } }
  c(AIC = which.min(a), BIC = which.min(b))
}
graphK <- function(X, gt, m) {
  fit <- tryCatch(tLocOrder(X, m=m, lambdas=lambdas, df=nu, graphtype=gt, K=Kstart,
                            verbose=FALSE, maxMem=100, maxadmm=80), error=function(e) NULL)
  if (is.null(fit)) return(list(K=NA, bic=Inf))
  bt <- tryCatch(bicTuning(X, fit), error=function(e) NULL)
  if (is.null(bt)) return(list(K=NA, bic=Inf))
  list(K = bt$result$order, bic = bt$result$bic)
}

run_scenario <- function(label, mu, Sigma, fname) {
  Kstar <- ncol(mu)
  methods <- c("AIC","BIC","1NN","2NN","3NN","Adp-m","MST","GSF")
  M <- matrix(NA, nrep, length(methods), dimnames = list(NULL, methods))
  for (r in 1:nrep) {
    set.seed(1000 + r); X <- rmix_t(n, mu, Sigma, nu)
    ic <- ic_pick(X, nu); M[r,"AIC"] <- ic["AIC"]; M[r,"BIC"] <- ic["BIC"]
    g1 <- graphK(X,"MNN",1); g2 <- graphK(X,"MNN",2); g3 <- graphK(X,"MNN",3)
    M[r,"1NN"] <- g1$K; M[r,"2NN"] <- g2$K; M[r,"3NN"] <- g3$K
    bics <- c(g1$bic,g2$bic,g3$bic); M[r,"Adp-m"] <- c(g1$K,g2$K,g3$K)[which.min(bics)]
    M[r,"MST"] <- graphK(X,"MST",1)$K
    M[r,"GSF"] <- graphK(X,"GSF",1)$K
    cat(sprintf("  %s rep %d/%d done\n", label, r, nrep))
  }
  diff <- M - Kstar
  df <- data.frame(method = factor(methods, levels = methods),
                   mean = colMeans(diff, na.rm = TRUE),
                   sd   = apply(diff, 2, sd, na.rm = TRUE))
  cat(sprintf("== %s (K*=%d, nu=%d) ==\n", label, Kstar, nu))
  print(round(df[,2:3], 2))
  fig <- ggplot(df, aes(method, mean)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), color = "red", width = 0.18, linewidth = 0.7) +
    geom_point(shape = 95, size = 12, color = "blue") +
    labs(x = NULL, y = expression(hat(K) - K^"*")) +
    theme_bw(base_size = 14) + theme(panel.grid.minor = element_blank())
  if (!dir.exists("images")) dir.create("images")
  ggsave(paste0("images/", fname), fig, width = 6, height = 5, dpi = 150)
  saveRDS(M, paste0("images/", sub(".png", ".rds", fname)))
  cat("Saved: images/", fname, "\n", sep = "")
}

run_scenario("t-linear",    cbind(c(0,0),c(2,2),c(4,4),c(6,6)),          diag(2), "Tlin-nu5-type6.png")
run_scenario("t-symmetric", cbind(c(0,0,0),c(0,0,2),c(0,2,0),c(2,0,0)),  diag(3), "Tsym-nu5-type6.png")
cat("ALLDONE\n")
