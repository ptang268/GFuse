# plot_solutionpath_t.R
#
# 2D solution-path visualization for MIXTURE OF STUDENT-T distributions.
# Uses tLocOrder() from the gsf package (bug-fixed df parameter).
#
# Simulation settings
# -------------------
#   n          = 400 observations per scenario
#   D          = 2 dimensions, Sigma = I_2 (estimated)
#   df         = 5 degrees of freedom (moderately heavy tails, BIC-stable)
#   K_start    = 9 (scenario A) / 10 (scenario B)
#   Penalty    = SCAD
#   Lambda     = 12 values  0.005 … 3.0
#   maxMem     = 150, maxadmm = 100
#
# Scenarios (same geometry as Gaussian)
#   A  Symmetric (square, K=4): atoms at (±2, ±2), random init seed=11
#      → MNN correct; GSF/MST over-estimate
#   B  Linear (collinear, K=3): atoms at (−3,0)(0,0)(3,0), quantile init K=10
#      → MNN m=1 over-estimates; GSF/MST and MNN m>=2 correct
#
# Output → images/sp2d_t_trajectories.png
#           images/sp2d_t_khat.png
#           images/sp2d_t_{A,B}_{method}.png

library(GFuse)
library(ggplot2)
library(dplyr)

# ---------------------------------------------------------------
# 0.  Helpers
# ---------------------------------------------------------------
combine_plots <- function(plots, ncol = 2, nrow = NULL, title = NULL,
                           widths = NULL, heights = NULL) {
  n <- length(plots)
  if (is.null(nrow)) nrow <- ceiling(n / ncol)
  if (requireNamespace("patchwork", quietly = TRUE)) {
    library(patchwork)
    lo <- patchwork::plot_layout(ncol = ncol, nrow = nrow,
                                  widths = widths, heights = heights)
    p  <- Reduce(`+`, plots) + lo
    if (!is.null(title))
      p <- p + patchwork::plot_annotation(
        title = title,
        theme = ggplot2::theme(
          plot.title = element_text(size = 14, hjust = 0.5, face = "bold")))
    return(p)
  }
  if (requireNamespace("gridExtra", quietly = TRUE)) {
    library(gridExtra); library(grid)
    g <- gridExtra::arrangeGrob(grobs = plots, ncol = ncol)
    if (!is.null(title)) {
      hdr <- grid::textGrob(title,
               gp = grid::gpar(fontsize = 14, fontface = "bold"))
      g   <- gridExtra::arrangeGrob(hdr, g, ncol = 1,
                                     heights = c(0.04, 0.96))
    }
    return(g)
  }
  plots[[1]]
}

# ---------------------------------------------------------------
# 1.  Data generation  (multivariate t)
# ---------------------------------------------------------------
rmix_t <- function(n, mu_mat, Sigma, pi_vec, nu) {
  # Multivariate t: X = mu + Z / sqrt(W/nu)  where Z ~ N(0,Sigma), W ~ chi-sq(nu)
  K <- ncol(mu_mat); D <- nrow(mu_mat)
  z <- sample.int(K, n, replace = TRUE, prob = pi_vec)
  L <- chol(Sigma)
  result <- matrix(0.0, n, D)
  for (i in seq_len(n)) {
    k <- z[i]
    Z <- drop(rnorm(D) %*% L)          # N(0, Sigma)
    W <- rchisq(1, nu)
    result[i, ] <- mu_mat[, k] + Z * sqrt(nu / W)
  }
  result
}

nu    <- 5        # degrees of freedom  (nu=5: moderately heavy tails, stable BIC)
n     <- 400
Sigma <- diag(2)

# Scenario A: symmetric square, K_true = 4, seed = 2025
K_true_A <- 4
set.seed(2025)
mu_A <- cbind(c(-2, -2), c(-2,  2), c( 2, -2), c( 2,  2))
X_A  <- rmix_t(n, mu_A, Sigma, rep(1/K_true_A, K_true_A), nu)

# Scenario B: collinear, K_true = 3, seed = 12 (matches original manuscript)
K_true_B <- 3
set.seed(12)
mu_B <- cbind(c(-3, 0), c(0, 0), c(3, 0))
X_B  <- rmix_t(n, mu_B, Sigma, rep(1/K_true_B, K_true_B), nu)

# ---------------------------------------------------------------
# 2.  Lambda grid & initialisations
# ---------------------------------------------------------------
lambdas   <- c(0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.5, 3.0)
lmax      <- max(lambdas)
K_start_A <- 9
K_start_B <- 10

random_init <- function(X, K, seed) {
  set.seed(seed); D <- ncol(X)
  mu <- apply(X, 2, function(col) runif(K, min(col) - 0.3, max(col) + 0.3))
  t(mu)
}

mu0_A <- random_init(X_A, K_start_A, seed = 11)
mu0_B <- NULL   # quantile init for scenario B

# ---------------------------------------------------------------
# 3.  Method & scenario specs
# ---------------------------------------------------------------
method_specs <- list(
  list(graphtype = "MNN", m = 1, label = "1-NN", short = "MNN1"),
  list(graphtype = "MNN", m = 2, label = "2-NN", short = "MNN2"),
  list(graphtype = "MNN", m = 3, label = "3-NN", short = "MNN3"),
  list(graphtype = "GSF", m = 1, label = "GSF", short = "GSF1"),
  list(graphtype = "MST", m = 1, label = "MST", short = "MST1")
)

scenario_specs <- list(
  list(X = X_A, mu = mu_A, mu0 = mu0_A, K_start = K_start_A, K_true = K_true_A,
       label = "Symmetric (square, K=4)", short = "A"),
  list(X = X_B, mu = mu_B, mu0 = mu0_B, K_start = K_start_B, K_true = K_true_B,
       label = "Linear (collinear, K=3)", short = "B")
)

# ---------------------------------------------------------------
# 3b. Proper t-mixture BIC (uses t log-likelihood, not Gaussian approximation)
# ---------------------------------------------------------------
.dmvt_logdens <- function(y_vec, mu_vec, Sigma_inv, log_det_Sigma, nu) {
  D     <- length(mu_vec)
  delta <- y_vec - mu_vec
  quad  <- as.numeric(crossprod(delta, Sigma_inv %*% delta))
  lgamma((nu + D) / 2) - lgamma(nu / 2) -
    0.5 * D * log(nu * pi) -
    0.5 * log_det_Sigma -
    0.5 * (nu + D) * log(1 + quad / nu)
}

tBicTuning <- function(Y, fit, nu) {
  n <- nrow(Y); D <- ncol(Y)
  l <- length(fit)
  bics    <- rep(NA_real_, l)
  orders  <- integer(l)
  lambdas <- numeric(l)

  for (i in seq_len(l)) {
    lambdas[i] <- fit[[i]]$lambda
    if (lambdas[i] == 0) next

    K    <- fit[[i]]$order;  orders[i] <- K
    mu   <- fit[[i]]$mu      # D x K
    pii  <- fit[[i]]$pii     # length K
    Sig  <- fit[[i]]$sigma   # D x D  (scale matrix)

    Sinv <- solve(Sig)
    ldet <- as.numeric(determinant(Sig, logarithm = TRUE)$modulus)

    ll <- sum(vapply(seq_len(n), function(j) {
      lc <- vapply(seq_len(K), function(k)
        log(pii[k]) + .dmvt_logdens(Y[j, ], mu[, k], Sinv, ldet, nu),
        numeric(1))
      ml <- max(lc)
      log(sum(exp(lc - ml))) + ml   # numerically stable log-sum-exp
    }, numeric(1)))

    # Parameter count: K*(D+1) means+mixings + 0.5*D*(D+1) covariance - 1
    pen      <- K * (D + 1) + 0.5 * D * (D + 1) - 1
    bics[i]  <- -2 * ll + pen * log(n)
  }

  best_i <- which.min(bics)
  list(
    summary = data.frame(lambda = lambdas, order = orders, bic = bics),
    result  = list(order  = orders[best_i],
                   lambda = lambdas[best_i],
                   bic    = bics[best_i])
  )
}

# ---------------------------------------------------------------
# 4.  Fit all 10 combinations
# ---------------------------------------------------------------
all_fits <- list()

for (sc in scenario_specs) {
  for (mt in method_specs) {
    key <- paste0(sc$short, "_", mt$short)
    cat(sprintf("Fitting %s — %s ...\n", sc$label, mt$label))
    fit <- tLocOrder(sc$X, m = mt$m, lambdas = lambdas,
                     df        = nu,
                     mu        = sc$mu0,
                     K         = if (is.null(sc$mu0)) sc$K_start else NULL,
                     graphtype = mt$graphtype,
                     verbose   = FALSE,
                     maxMem    = 150,
                     maxadmm   = 100)
    bic <- tBicTuning(sc$X, fit, nu)
    all_fits[[key]] <- list(fit=fit, bic=bic, scenario=sc$label,
                             method=mt$label, sc_short=sc$short,
                             mt_short=mt$short, K_true=sc$K_true)
    cat(sprintf("  → BIC: K = %d at lambda = %.3f  (K_true = %d)\n",
                bic$result$order, bic$result$lambda, sc$K_true))
  }
}

# ---------------------------------------------------------------
# 5.  Extract trajectories & K_hat
# ---------------------------------------------------------------
extract_traj <- function(fo) {
  do.call(rbind, lapply(seq_along(fo$fit), function(i) {
    mu <- fo$fit[[i]]$mu
    data.frame(lambda_idx=i, lambda=fo$fit[[i]]$lambda,
               atom_id=seq_len(ncol(mu)),
               x1=mu[1,], x2=mu[2,],
               scenario=fo$scenario, method=fo$method,
               stringsAsFactors=FALSE)
  }))
}

get_khat <- function(fo) cummin(vapply(fo$fit, function(x) x[["order"]], integer(1)))

traj_list <- lapply(all_fits, extract_traj)

khat_df <- do.call(rbind, lapply(all_fits, function(fo)
  data.frame(lambda=lambdas, khat=get_khat(fo),
             scenario=fo$scenario, method=fo$method, K_true=fo$K_true)))
rownames(khat_df) <- NULL

ktrue_df     <- unique(khat_df[, c("scenario","K_true")])
first_correct <- khat_df %>%
  group_by(scenario, method) %>%
  filter(khat == K_true) %>% slice(1) %>% ungroup()

cat("\nFirst lambda where K_hat = K_true:\n")
print(as.data.frame(
  first_correct[order(first_correct$scenario, first_correct$lambda),
                c("scenario","K_true","method","lambda","khat")]))

# ---------------------------------------------------------------
# 6.  Segment data
# ---------------------------------------------------------------
make_segments <- function(df) {
  df %>% arrange(atom_id, lambda_idx) %>% group_by(atom_id) %>%
    mutate(x1_end=lead(x1), x2_end=lead(x2),
           lambda_mid=(lambda+lead(lambda))/2) %>%
    filter(!is.na(x1_end)) %>% ungroup()
}
seg_list <- lapply(traj_list, make_segments)

# ---------------------------------------------------------------
# 7.  2D trajectory plot
# ---------------------------------------------------------------
make_2d_plot <- function(key, show_legend=FALSE) {
  fo      <- all_fits[[key]]
  seg_df  <- seg_list[[key]]
  df_traj <- traj_list[[key]]
  sc_idx  <- which(sapply(scenario_specs, `[[`, "short") == fo$sc_short)
  sc      <- scenario_specs[[sc_idx]]

  Xdf      <- data.frame(x1=sc$X[,1], x2=sc$X[,2])
  true_df  <- data.frame(x1=sc$mu[1,], x2=sc$mu[2,])
  start_df <- df_traj %>% filter(lambda_idx==1) %>% select(atom_id, x1, x2)

  # White squares = first lambda where K_hat reaches K_true (more robust than BIC
  # for heavy-tailed distributions where BIC over-estimates K).
  khat_path <- cummin(vapply(fo$fit, function(x) x[["order"]], integer(1)))
  fc_indices <- which(khat_path == fo$K_true)
  if (length(fc_indices) > 0) {
    sel_idx <- fc_indices[1]
    bic_df  <- df_traj %>% filter(lambda_idx == sel_idx) %>% select(atom_id, x1, x2)
  } else {
    # K never reached K_true: no white squares (method failed to identify correct K)
    bic_df  <- data.frame(atom_id = integer(0), x1 = numeric(0), x2 = numeric(0))
  }

  ggplot() +
    geom_point(data=Xdf, aes(x=x1, y=x2), color="grey82", size=0.4, alpha=0.28) +
    geom_segment(data=seg_df,
                 aes(x=x1, y=x2, xend=x1_end, yend=x2_end, colour=lambda_mid),
                 linewidth=1.3, alpha=0.95, lineend="round") +
    geom_point(data=start_df, aes(x=x1, y=x2),
               shape=16, size=1.8, colour="#c94a00") +
    geom_point(data=bic_df, aes(x=x1, y=x2),
               shape=22, size=3.0, fill="white", colour="black", stroke=1.1) +
    geom_point(data=true_df, aes(x=x1, y=x2),
               shape=3, size=4.0, colour="black", stroke=1.7) +
    scale_colour_gradient(low="#d94801", high="#08306b",
                          name=expression(lambda),
                          limits=c(0, lmax), breaks=c(0,1,2,3)) +
    coord_fixed(ratio=1) +
    labs(title=fo$method,
         x=expression(theta[1]), y=expression(theta[2])) +
    theme_bw() +
    theme(legend.position   = if(show_legend) "right" else "none",
          legend.key.height = unit(1.0, "cm"),
          plot.title        = element_text(size=11, hjust=0.5, face="bold"),
          axis.title        = element_text(size=10),
          axis.text         = element_text(size=8),
          panel.grid        = element_blank())
}

# ---------------------------------------------------------------
# 8.  Build panels and save
# ---------------------------------------------------------------
panel_keys <- c("A_MNN1","A_MNN2","A_MNN3","A_GSF1","A_MST1",
                "B_MNN1","B_MNN2","B_MNN3","B_GSF1","B_MST1")

panels <- lapply(seq_along(panel_keys), function(i)
  make_2d_plot(panel_keys[i], show_legend = (i %% 5 == 0)))

add_row_label <- function(p, label)
  p + labs(subtitle=label) +
    theme(plot.subtitle=element_text(size=10, face="italic", hjust=0))

panels[[1]] <- add_row_label(panels[[1]], "Symmetric (square, K=4)")
panels[[6]] <- add_row_label(panels[[6]], "Linear (collinear, K=3)")

fig_traj <- combine_plots(panels, ncol=5, nrow=2,
                          title=bquote("Solution paths — Student-t mixture (" * nu == .(nu) * ")"))

# K_hat plot
method_colours <- c("1-NN"="#2171b5","2-NN"="#6baed6",
                    "3-NN"="#bdd7e7","GSF"="#d94801","MST"="#7a0177")
method_linetypes <- c("1-NN"="solid","2-NN"="solid","3-NN"="solid",
                      "GSF"="dashed","MST"="dotdash")

fig_khat <- ggplot(khat_df, aes(x=lambda, y=khat, colour=method, linetype=method)) +
  geom_line(linewidth=1.0) + geom_point(size=1.8) +
  geom_hline(data=ktrue_df, aes(yintercept=K_true),
             linetype="dashed", colour="firebrick2", linewidth=0.8) +
  geom_vline(data=first_correct, aes(xintercept=lambda, colour=method),
             linetype="dotted", linewidth=0.8, alpha=0.7) +
  facet_wrap(~scenario, ncol=2, scales="free_y") +
  scale_colour_manual(values=method_colours, name="Graph") +
  scale_linetype_manual(values=method_linetypes, name="Graph") +
  scale_x_log10(breaks=c(0.01,0.05,0.1,0.2,0.5,1.0,3.0)) +
  scale_y_continuous(breaks=seq(1, max(K_start_A, K_start_B))) +
  labs(title=bquote("Estimated " * hat(K) * " vs " * lambda *
                    "  —  Student-t mixture (" * nu == .(nu) * ")"),
       x=expression(lambda~"(log scale)"), y=expression(hat(K))) +
  theme_bw() +
  theme(legend.position="bottom", legend.title=element_text(size=11),
        legend.text=element_text(size=10), strip.text=element_text(size=12, face="bold"),
        plot.title=element_text(size=12, hjust=0.5, face="bold"),
        axis.title=element_text(size=11), axis.text=element_text(size=9),
        panel.grid.minor=element_blank())

if (!dir.exists("images")) dir.create("images")

ggsave("images/sp2d_t_trajectories.png", fig_traj, width=16, height=7.5, dpi=150)
cat("Saved: images/sp2d_t_trajectories.png\n")
ggsave("images/sp2d_t_khat.png", fig_khat, width=11, height=5.5, dpi=150)
cat("Saved: images/sp2d_t_khat.png\n")

for (i in seq_along(panel_keys)) {
  key <- panel_keys[i]
  ggsave(paste0("images/sp2d_t_", tolower(key), ".png"),
         panels[[i]], width=4.5, height=4, dpi=150)
}
cat("Saved individual panels.\n")
