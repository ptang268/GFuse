# plot_solutionpath_multinom.R
#
# 2D solution-path visualization for MIXTURE OF MULTINOMIAL distributions.
# Uses multinomialOrder() from the gsf package.
#
# Simulation settings
# -------------------
#   n          = 400 observations per scenario
#   D          = 3 categories (θ visualised in (θ_1, θ_2) simplex space)
#   M          = 50 trials per observation
#   K_start    = 9 (both scenarios, but different init strategy)
#   Penalty    = SCAD
#   Lambda     = 12 values  0.005 … 3.0
#   maxMem     = 150, maxadmm = 100
#
# Scenarios
#   A  Symmetric (interior simplex, K=4): M3 config, seed=2025
#      theta: (0.2,0.2,0.6), (0.2,0.6,0.2), (0.6,0.2,0.2), (0.45,0.1,0.45)
#      Init:  MCMC (K=9, mcmcIter=10), set.seed(11) before each call
#      → MNN correct (K=4); GSF/MST over-estimate
#   B  Linear (collinear, K=3): atoms on θ_3=0.2 line, seed=12
#      theta: (0.1,0.7,0.2), (0.4,0.4,0.2), (0.7,0.1,0.2)
#      Init:  Explicit "transition-atom" init mimicking Gaussian quantile init
#             (2 + 1 + 3 + 1 + 2 atoms spanning the collinear direction)
#      → MNN m=1 over-estimates; GSF/MST correct (K=3)
#
# Output → images/sp2d_multinom_trajectories.png
#           images/sp2d_multinom_khat.png
#           images/sp2d_multinom_{A,B}_{method}.png

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
# 1.  Data generation
# ---------------------------------------------------------------
rmix_multinom <- function(n, theta_mat, pi_vec, M) {
  K <- ncol(theta_mat)
  z <- sample.int(K, n, replace = TRUE, prob = pi_vec)
  t(sapply(seq_len(n), function(i)
    as.integer(rmultinom(1, M, theta_mat[, z[i]]))))
}

M <- 50; n <- 400

# Scenario A: symmetric interior, K_true = 4, seed = 2025
K_true_A <- 4
theta_A <- cbind(c(0.20, 0.20, 0.60),
                 c(0.20, 0.60, 0.20),
                 c(0.60, 0.20, 0.20),
                 c(0.45, 0.10, 0.45))
set.seed(2025)
X_A <- rmix_multinom(n, theta_A, rep(1/K_true_A, K_true_A), M)

# Scenario B: collinear (θ_3 = 0.20 line), K_true = 3, seed = 12
K_true_B <- 3
theta_B <- cbind(c(0.10, 0.70, 0.20),
                 c(0.40, 0.40, 0.20),
                 c(0.70, 0.10, 0.20))
set.seed(12)
X_B <- rmix_multinom(n, theta_B, rep(1/K_true_B, K_true_B), M)

# ---------------------------------------------------------------
# 2.  Lambda grid & initialisations
# ---------------------------------------------------------------
lambdas   <- c(0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.5, 3.0)
lmax      <- max(lambdas)
K_start_A <- 9    # MCMC init for scenario A
K_start_B <- 9    # explicit transition-atom init for scenario B

# Transition-atom init for Scenario B
# Places 2 atoms per true cluster + 1 "bridge" atom between each pair of clusters.
# Under MNN m=1, bridge atoms are isolated (no mutual NN edge) and persist as
# spurious components, just like the transition quantile atoms in the Gaussian case.
t1_init  <- c(0.08, 0.12,          # near cluster 1 (θ1=0.10)
              0.25,                 # bridge atom between clusters 1 & 2
              0.37, 0.40, 0.43,    # near cluster 2 (θ1=0.40)
              0.55,                 # bridge atom between clusters 2 & 3
              0.68, 0.72)           # near cluster 3 (θ1=0.70)
t2_init  <- 0.80 - t1_init         # θ_3 = 0.20 for all init atoms
theta0_B <- rbind(t1_init, t2_init) # 2×9 (first two categories; 3rd implied)
pii0_B   <- rep(1 / K_start_B, K_start_B)

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
  list(X = X_A, theta = theta_A[1:2, ], theta_true_all = theta_A,
       theta0 = NULL, pii0 = NULL, use_mcmc = TRUE,
       K_start = K_start_A, K_true = K_true_A,
       label = "Symmetric (interior, K=4)", short = "A"),
  list(X = X_B, theta = theta_B[1:2, ], theta_true_all = theta_B,
       theta0 = theta0_B, pii0 = pii0_B, use_mcmc = FALSE,
       K_start = K_start_B, K_true = K_true_B,
       label = "Linear (collinear, K=3)", short = "B")
)

# ---------------------------------------------------------------
# 4.  Fit all 10 combinations
# ---------------------------------------------------------------
all_fits <- list()

for (sc in scenario_specs) {
  for (mt in method_specs) {
    key <- paste0(sc$short, "_", mt$short)
    cat(sprintf("Fitting %s — %s ...\n", sc$label, mt$label))

    if (sc$use_mcmc) {
      set.seed(11)   # same MCMC init for all methods in this scenario
      fit <- multinomialOrder(sc$X, m = mt$m, lambdas = lambdas,
                              K         = sc$K_start,
                              graphtype = mt$graphtype,
                              verbose   = FALSE,
                              maxMem    = 150,
                              maxadmm   = 100,
                              mcmcIter  = 10)
    } else {
      fit <- multinomialOrder(sc$X, m = mt$m, lambdas = lambdas,
                              theta     = sc$theta0,
                              pii       = sc$pii0,
                              graphtype = mt$graphtype,
                              verbose   = FALSE,
                              maxMem    = 150,
                              maxadmm   = 100)
    }

    bic <- bicTuning(sc$X, fit)
    all_fits[[key]] <- list(fit      = fit,
                             bic      = bic,
                             scenario = sc$label,
                             method   = mt$label,
                             sc_short = sc$short,
                             mt_short = mt$short,
                             K_true   = sc$K_true)
    cat(sprintf("  → BIC: K = %d at lambda = %.3f  (K_true = %d)\n",
                bic$result$order, bic$result$lambda, sc$K_true))
  }
}

# ---------------------------------------------------------------
# 5.  Extract trajectories & K_hat
# ---------------------------------------------------------------
extract_traj <- function(fo) {
  do.call(rbind, lapply(seq_along(fo$fit), function(i) {
    th <- fo$fit[[i]]$theta   # 2×K (first 2 multinomial categories)
    data.frame(lambda_idx = i,
               lambda     = fo$fit[[i]]$lambda,
               atom_id    = seq_len(ncol(th)),
               x1 = th[1, ], x2 = th[2, ],
               scenario = fo$scenario, method = fo$method,
               stringsAsFactors = FALSE)
  }))
}

get_khat <- function(fo) cummin(vapply(fo$fit, function(x) x[["order"]], integer(1)))

traj_list <- lapply(all_fits, extract_traj)

khat_df <- do.call(rbind, lapply(all_fits, function(fo)
  data.frame(lambda  = lambdas,
             khat    = get_khat(fo),
             scenario = fo$scenario,
             method   = fo$method,
             K_true   = fo$K_true)))
rownames(khat_df) <- NULL

ktrue_df     <- unique(khat_df[, c("scenario", "K_true")])
first_correct <- khat_df %>%
  group_by(scenario, method) %>%
  filter(khat == K_true) %>% slice(1) %>% ungroup()

cat("\nFirst lambda where K_hat = K_true:\n")
print(as.data.frame(
  first_correct[order(first_correct$scenario, first_correct$lambda),
                c("scenario", "K_true", "method", "lambda", "khat")]))

# ---------------------------------------------------------------
# 6.  Segment data
# ---------------------------------------------------------------
make_segments <- function(df) {
  df %>% arrange(atom_id, lambda_idx) %>% group_by(atom_id) %>%
    mutate(x1_end = lead(x1), x2_end = lead(x2),
           lambda_mid = (lambda + lead(lambda)) / 2) %>%
    filter(!is.na(x1_end)) %>% ungroup()
}
seg_list <- lapply(traj_list, make_segments)

# ---------------------------------------------------------------
# 7.  2D trajectory plot (simplex space)
# ---------------------------------------------------------------
# Simplex hypotenuse: θ_1 + θ_2 = 1  (θ_3 = 0 boundary)
simplex_hyp <- data.frame(x = c(0, 1), y = c(1, 0))

make_2d_plot <- function(key, show_legend = FALSE) {
  fo      <- all_fits[[key]]
  seg_df  <- seg_list[[key]]
  df_traj <- traj_list[[key]]
  sc_idx  <- which(sapply(scenario_specs, `[[`, "short") == fo$sc_short)
  sc      <- scenario_specs[[sc_idx]]

  # Data scatter (empirical proportions, first two categories)
  Xdf     <- data.frame(x1 = sc$X[, 1] / M, x2 = sc$X[, 2] / M)
  # True component locations (first two coordinates)
  true_df <- data.frame(x1 = sc$theta_true_all[1, ],
                        x2 = sc$theta_true_all[2, ])
  # Initial atom positions
  start_df <- df_traj %>% filter(lambda_idx == 1) %>% select(atom_id, x1, x2)
  # White squares = first lambda where K_hat reaches K_true (more robust than BIC
  # for heavy-tailed or discrete distributions where BIC over-estimates K).
  khat_path  <- cummin(vapply(fo$fit, function(x) x[["order"]], integer(1)))
  fc_indices <- which(khat_path == fo$K_true)
  if (length(fc_indices) > 0) {
    sel_idx <- fc_indices[1]
    bic_df  <- df_traj %>% filter(lambda_idx == sel_idx) %>% select(atom_id, x1, x2)
  } else {
    # K never reached K_true: no white squares (method failed to identify correct K)
    bic_df  <- data.frame(atom_id = integer(0), x1 = numeric(0), x2 = numeric(0))
  }

  ggplot() +
    # Simplex boundary (dashed): axes are θ_1=0 and θ_2=0 (plot edges),
    # hypotenuse is θ_1+θ_2=1 (θ_3=0)
    geom_line(data = simplex_hyp, aes(x = x, y = y),
              colour = "grey50", linetype = "dashed", linewidth = 0.5) +
    # Data scatter
    geom_point(data = Xdf, aes(x = x1, y = x2),
               colour = "grey82", size = 0.4, alpha = 0.28) +
    # Trajectories
    geom_segment(data = seg_df,
                 aes(x = x1, y = x2, xend = x1_end, yend = x2_end,
                     colour = lambda_mid),
                 linewidth = 1.3, alpha = 0.95, lineend = "round") +
    # Initial atom positions (start of path)
    geom_point(data = start_df, aes(x = x1, y = x2),
               shape = 16, size = 1.8, colour = "#c94a00") +
    # BIC-selected model (white square)
    geom_point(data = bic_df, aes(x = x1, y = x2),
               shape = 22, size = 3.0, fill = "white", colour = "black",
               stroke = 1.1) +
    # True component locations (cross)
    geom_point(data = true_df, aes(x = x1, y = x2),
               shape = 3, size = 4.0, colour = "black", stroke = 1.7) +
    scale_colour_gradient(low = "#d94801", high = "#08306b",
                          name = expression(lambda),
                          limits = c(0, lmax), breaks = c(0, 1, 2, 3)) +
    coord_fixed(ratio = 1, xlim = c(-0.02, 0.87), ylim = c(-0.02, 0.87)) +
    labs(title = fo$method,
         x = expression(theta[1]), y = expression(theta[2])) +
    theme_bw() +
    theme(legend.position   = if (show_legend) "right" else "none",
          legend.key.height = unit(1.0, "cm"),
          plot.title        = element_text(size = 11, hjust = 0.5, face = "bold"),
          axis.title        = element_text(size = 10),
          axis.text         = element_text(size = 8),
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
  p + labs(subtitle = label) +
    theme(plot.subtitle = element_text(size = 10, face = "italic", hjust = 0))

panels[[1]] <- add_row_label(panels[[1]], "Symmetric (interior, K=4)")
panels[[6]] <- add_row_label(panels[[6]], "Linear (collinear, K=3)")

fig_traj <- combine_plots(panels, ncol = 5, nrow = 2,
                          title = "Solution paths — Multinomial mixture")

# K_hat plot
method_colours  <- c("1-NN" = "#2171b5", "2-NN" = "#6baed6",
                     "3-NN" = "#bdd7e7", "GSF" = "#d94801",
                     "MST" = "#7a0177")
method_linetypes <- c("1-NN" = "solid", "2-NN" = "solid",
                      "3-NN" = "solid", "GSF" = "dashed",
                      "MST" = "dotdash")

fig_khat <- ggplot(khat_df, aes(x = lambda, y = khat,
                                 colour = method, linetype = method)) +
  geom_line(linewidth = 1.0) + geom_point(size = 1.8) +
  geom_hline(data = ktrue_df, aes(yintercept = K_true),
             linetype = "dashed", colour = "firebrick2", linewidth = 0.8) +
  geom_vline(data = first_correct, aes(xintercept = lambda, colour = method),
             linetype = "dotted", linewidth = 0.8, alpha = 0.7) +
  facet_wrap(~scenario, ncol = 2, scales = "free_y") +
  scale_colour_manual(values = method_colours, name = "Graph") +
  scale_linetype_manual(values = method_linetypes, name = "Graph") +
  scale_x_log10(breaks = c(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0)) +
  scale_y_continuous(breaks = seq(1, max(K_start_A, K_start_B))) +
  labs(title = bquote("Estimated " * hat(K) * " vs " * lambda *
                      "  —  Multinomial mixture"),
       x = expression(lambda ~ "(log scale)"), y = expression(hat(K))) +
  theme_bw() +
  theme(legend.position  = "bottom",
        legend.title     = element_text(size = 11),
        legend.text      = element_text(size = 10),
        strip.text       = element_text(size = 12, face = "bold"),
        plot.title       = element_text(size = 12, hjust = 0.5, face = "bold"),
        axis.title       = element_text(size = 11),
        axis.text        = element_text(size = 9),
        panel.grid.minor = element_blank())

if (!dir.exists("images")) dir.create("images")

ggsave("images/sp2d_multinom_trajectories.png", fig_traj,
       width = 16, height = 7.5, dpi = 150)
cat("Saved: images/sp2d_multinom_trajectories.png\n")
ggsave("images/sp2d_multinom_khat.png", fig_khat,
       width = 11, height = 5.5, dpi = 150)
cat("Saved: images/sp2d_multinom_khat.png\n")

for (i in seq_along(panel_keys)) {
  key <- panel_keys[i]
  ggsave(paste0("images/sp2d_multinom_", tolower(key), ".png"),
         panels[[i]], width = 4.5, height = 4, dpi = 150)
}
cat("Saved individual panels.\n")
