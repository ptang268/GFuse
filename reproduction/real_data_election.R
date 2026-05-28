# real_data_election.R
#
# Real data analysis: 2020 US Presidential Election by County
# Source: MIT Election Data and Science Lab (via public GitHub mirror)
#
# Each county's vote totals are modeled as a multinomial draw over
# three categories: Democrat (D), Republican (R), Other/Third-party (O).
# theta_k = (theta_D, theta_R, theta_O) for cluster k.
# K = number of politically distinct "county types".
#
# Visualization: simplex (theta_D, theta_R) space.

library(GFuse)
library(ggplot2)
library(dplyr)

set.seed(2025)

# ---------------------------------------------------------------
# 1.  Load and preprocess
# ---------------------------------------------------------------
url <- paste0("https://raw.githubusercontent.com/",
              "xsarinix/election2020-data/main/countypres_2000-2020.csv")

cat("Downloading 2020 county election data from MIT Election Lab...\n")
raw <- tryCatch(
  read.csv(url, stringsAsFactors=FALSE),
  error=function(e) { cat("Download failed:", e$message, "\n"); NULL }
)

if (is.null(raw)) stop("Could not load election data.")

# Filter to 2020 presidential; aggregate to 3 parties
elec2020 <- raw %>%
  filter(year==2020, office=="US PRESIDENT") %>%
  mutate(party3 = case_when(
    party == "DEMOCRAT"   ~ "D",
    party == "REPUBLICAN" ~ "R",
    TRUE                  ~ "O"
  )) %>%
  group_by(county_fips, county_name, state_po, party3) %>%
  summarise(votes=sum(candidatevotes, na.rm=TRUE), .groups="drop") %>%
  tidyr::pivot_wider(names_from=party3, values_from=votes, values_fill=0) %>%
  mutate(total = D + R + O) %>%
  filter(total >= 500,           # drop tiny/incomplete counties
         !is.na(county_fips))    # drop missing FIPS

cat(sprintf("Counties after filtering: n = %d\n", nrow(elec2020)))

# Subsample for computational feasibility (stratified by D-share quantile)
set.seed(2025)
n_use <- 1000
elec2020 <- elec2020[sample(nrow(elec2020), n_use), ]
cat(sprintf("Subsampled to n = %d for analysis\n", nrow(elec2020)))

# Build count matrix (n x 3): columns = D, R, O
# Scale all counties to M=50 "pseudo-counts" (round proportions × 50)
# then add 1 to every category (Laplace smoothing) to ensure no zero
# probabilities.  This preserves relative vote shares while keeping
# counts numerically manageable.
Y_raw  <- as.matrix(elec2020[, c("D","R","O")])
props  <- sweep(Y_raw, 1, rowSums(Y_raw), "/")  # empirical proportions
Y_50   <- round(props * 50)
# Fix rounding: adjust largest category so row sum = 50
for (i in seq_len(nrow(Y_50))) {
  diff <- 50L - sum(Y_50[i,])
  Y_50[i, which.max(Y_50[i,])] <- Y_50[i, which.max(Y_50[i,])] + diff
}
Y <- Y_50 + 1L   # add 1 per category (Laplace); row sums = 53

cat("Total votes range (original):", range(rowSums(Y_raw)), "\n")
cat(sprintf("Counties with zero Other votes (original): %d (%.1f%%)\n",
            sum(Y_raw[,"O"]==0), 100*mean(Y_raw[,"O"]==0)))
cat("After scaling M=50 + Laplace: row sums =", unique(rowSums(Y)), "\n")
cat("Mean D share:", round(mean(props[,1]), 3), "\n")
cat("Mean R share:", round(mean(props[,2]), 3), "\n\n")

# ---------------------------------------------------------------
# 2.  Settings
# ---------------------------------------------------------------
lambdas <- c(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)
lmax    <- max(lambdas)
K_start <- 6

# Explicit theta initialisation — K_start atoms spread across the D-R axis
# Avoids MCMC sampling from near-degenerate probabilities.
t1_init <- c(0.70, 0.50, 0.38, 0.28, 0.16, 0.08)   # Dem share
t2_init <- c(0.25, 0.44, 0.56, 0.67, 0.79, 0.88)   # Rep share
theta0  <- rbind(t1_init, t2_init)   # 2 × K_start
pii0    <- rep(1/K_start, K_start)

method_specs <- list(
  list(graphtype="MNN", m=1, label="1-NN", short="MNN1"),
  list(graphtype="MNN", m=2, label="2-NN", short="MNN2"),
  list(graphtype="MNN", m=3, label="3-NN", short="MNN3"),
  list(graphtype="GSF", m=1, label="GSF",        short="GSF"),
  list(graphtype="MST", m=1, label="MST",         short="MST")
)

# ---------------------------------------------------------------
# 3.  Fit all 5 methods
# ---------------------------------------------------------------
all_fits <- list()

for (mt in method_specs) {
  key <- mt$short
  cat(sprintf("Fitting multinomialOrder — %s ...\n", mt$label))
  fit <- multinomialOrder(Y, m=mt$m, lambdas=lambdas,
                          theta=theta0, pii=pii0,
                          graphtype=mt$graphtype, verbose=FALSE,
                          maxMem=150, maxadmm=100)
  bic  <- bicTuning(Y, fit)
  khat <- cummin(vapply(fit, function(x) x[["order"]], integer(1)))

  all_fits[[key]] <- list(
    fit=fit, bic=bic, khat_path=khat,
    method=mt$label, mt_short=mt$short
  )
  cat(sprintf("  → BIC K = %d at lambda = %.3f\n",
              bic$result$order, bic$result$lambda))
}

# First-correct lambda table
khat_df <- do.call(rbind, lapply(names(all_fits), function(k) {
  fo <- all_fits[[k]]
  data.frame(lambda=lambdas, khat=fo$khat_path, method=fo$method)
}))

cat("\n=== BIC-selected K by method ===\n")
for (k in names(all_fits)) {
  fo <- all_fits[[k]]
  cat(sprintf("  %-12s: K = %d (lambda = %.3f)\n",
              fo$method, fo$bic$result$order, fo$bic$result$lambda))
}

# ---------------------------------------------------------------
# 4.  K_hat vs lambda plot
# ---------------------------------------------------------------
method_colours  <- c("1-NN"="#2171b5","2-NN"="#6baed6",
                     "3-NN"="#bdd7e7","GSF"="#d94801","MST"="#7a0177")
method_linetypes <- c("1-NN"="solid","2-NN"="solid","3-NN"="solid",
                      "GSF"="dashed","MST"="dotdash")

fig_khat <- ggplot(khat_df, aes(x=lambda, y=khat, colour=method, linetype=method)) +
  geom_line(linewidth=1.0) + geom_point(size=1.8) +
  scale_colour_manual(values=method_colours, name="Graph") +
  scale_linetype_manual(values=method_linetypes, name="Graph") +
  scale_x_log10(breaks=c(0.005,0.01,0.05,0.1,0.2,0.5,1.0)) +
  scale_y_continuous(breaks=1:K_start) +
  labs(title="2020 US county election — estimated K vs. λ",
       x=expression(lambda~"(log scale)"), y=expression(hat(K))) +
  theme_bw() +
  theme(legend.position="bottom", plot.title=element_text(size=11, hjust=0.5),
        axis.title=element_text(size=11), axis.text=element_text(size=9),
        panel.grid.minor=element_blank())

# ---------------------------------------------------------------
# 5.  Solution path in simplex (theta_D, theta_R)
# ---------------------------------------------------------------
simplex_hyp <- data.frame(x=c(0,1), y=c(1,0))  # theta_D + theta_R = 1

# Data scatter: empirical proportions (use raw counts for display)
total_raw <- rowSums(Y_raw)
Xdf <- data.frame(x1=Y_raw[,1]/total_raw, x2=Y_raw[,2]/total_raw)

make_multinom_panel <- function(mt_label, show_legend=FALSE) {
  key  <- c("1-NN"="MNN1","2-NN"="MNN2","3-NN"="MNN3",
            "GSF"="GSF","MST"="MST")[mt_label]
  fo   <- all_fits[[key]]

  # Trajectory data
  df_traj <- do.call(rbind, lapply(seq_along(fo$fit), function(i) {
    th <- fo$fit[[i]]$theta
    data.frame(lambda_idx=i, lambda=fo$fit[[i]]$lambda,
               atom_id=seq_len(ncol(th)), x1=th[1,], x2=th[2,])
  }))
  seg_df <- df_traj %>% arrange(atom_id, lambda_idx) %>% group_by(atom_id) %>%
    mutate(x1_end=lead(x1), x2_end=lead(x2), lambda_mid=(lambda+lead(lambda))/2) %>%
    filter(!is.na(x1_end)) %>% ungroup()

  # BIC-selected positions
  bic_lam <- fo$bic$result$lambda
  bic_idx <- which.min(abs(lambdas - bic_lam))
  bic_df  <- df_traj %>% filter(lambda_idx==bic_idx)
  start_df <- df_traj %>% filter(lambda_idx==1)

  ggplot() +
    geom_line(data=simplex_hyp, aes(x=x, y=y),
              colour="grey50", linetype="dashed", linewidth=0.5) +
    geom_point(data=Xdf, aes(x=x1, y=x2),
               colour="grey80", size=0.15, alpha=0.2) +
    geom_segment(data=seg_df,
                 aes(x=x1, y=x2, xend=x1_end, yend=x2_end, colour=lambda_mid),
                 linewidth=1.2, alpha=0.9, lineend="round") +
    geom_point(data=start_df, aes(x=x1, y=x2),
               shape=16, size=1.8, colour="#c94a00") +
    geom_point(data=bic_df, aes(x=x1, y=x2),
               shape=22, size=3.5, fill="white", colour="black", stroke=1.1) +
    scale_colour_gradient(low="#d94801", high="#08306b",
                          name=expression(lambda), limits=c(0,lmax),
                          breaks=c(0,0.3,0.6,0.9,1.2)) +
    coord_fixed(ratio=1, xlim=c(-0.02, 0.92), ylim=c(-0.02, 0.92)) +
    labs(title=mt_label,
         x=expression(theta[D]~"(Dem. share)"),
         y=expression(theta[R]~"(Rep. share)")) +
    theme_bw() +
    theme(legend.position=if(show_legend)"right" else "none",
          legend.key.height=unit(0.9,"cm"),
          plot.title=element_text(size=11, hjust=0.5, face="bold"),
          axis.title=element_text(size=10), axis.text=element_text(size=8),
          panel.grid=element_blank())
}

panel_labels <- c("1-NN","2-NN","3-NN","GSF","MST")
panels <- lapply(seq_along(panel_labels), function(i)
  make_multinom_panel(panel_labels[i], show_legend=(i==5)))

combine_row <- function(ps) {
  if (requireNamespace("patchwork", quietly=TRUE)) {
    library(patchwork)
    Reduce(`+`, ps) + patchwork::plot_layout(ncol=5)
  } else {
    library(gridExtra)
    gridExtra::arrangeGrob(grobs=ps, ncol=5)
  }
}
fig_traj <- combine_row(panels)

# ---------------------------------------------------------------
# 6.  Cluster assignment for best method (BIC)
# ---------------------------------------------------------------
best_key <- names(which.min(sapply(all_fits, function(fo) fo$bic$result$order)))
# Use consensus K across methods
K_votes <- table(sapply(all_fits, function(fo) fo$bic$result$order))
cat("\nBIC-selected K distribution across methods:\n"); print(K_votes)

# Use MNN(m=2) result
fo_best <- all_fits[["MNN2"]]
bic_best <- fo_best$bic$result
blam_idx <- which.min(abs(lambdas - bic_best$lambda))
theta_sel <- fo_best$fit[[blam_idx]]$theta   # 2 × K
pii_sel   <- fo_best$fit[[blam_idx]]$pii
K_sel     <- ncol(theta_sel)

# Hard cluster assignment
theta_full <- rbind(theta_sel, 1 - colSums(theta_sel))  # 3 × K
props <- sweep(Y, 1, rowSums(Y), "/")  # n × 3 proportion matrix
assign_cls <- function(x, theta) {
  which.max(vapply(seq_len(ncol(theta)), function(k)
    sum(x * log(pmax(theta[,k], 1e-9))), numeric(1)))
}
cls <- apply(props, 1, assign_cls, theta=theta_full)

cluster_cols <- c("#e41a1c","#377eb8","#4daf4a","#984ea3",
                  "#ff7f00","#a65628","#f781bf","#999999")
theta_df <- data.frame(x1=theta_sel[1,], x2=theta_sel[2,], cl=factor(1:K_sel))
# Use raw proportions for scatter in final figure
Xdf_cls  <- data.frame(x1=Y_raw[,1]/total_raw, x2=Y_raw[,2]/total_raw,
                       cluster=factor(cls))

cat(sprintf("\nMNN(m=2): K = %d political county types selected\n", K_sel))
cat("Cluster size distribution:\n"); print(table(cls))
cat("\nMNN(m=2) component means (theta_D, theta_R, theta_O):\n")
for (k in seq_len(K_sel)) {
  cat(sprintf("  Cluster %d: D=%.3f, R=%.3f, O=%.3f  (pi=%.3f)\n",
              k, theta_sel[1,k], theta_sel[2,k],
              1-sum(theta_sel[,k]), pii_sel[k]))
}

fig_scatter <- ggplot(Xdf_cls, aes(x=x1, y=x2, colour=cluster)) +
  geom_line(data=simplex_hyp, aes(x=x, y=y),
            colour="grey50", linetype="dashed", linewidth=0.5,
            inherit.aes=FALSE) +
  geom_point(size=0.4, alpha=0.4) +
  geom_point(data=theta_df, aes(x=x1, y=x2, colour=cl),
             shape=4, size=5, stroke=2.0, inherit.aes=FALSE) +
  scale_colour_manual(values=cluster_cols[1:K_sel], name="Cluster") +
  coord_fixed(ratio=1, xlim=c(-0.02,0.92), ylim=c(-0.02,0.92)) +
  labs(title=sprintf("2020 US counties: multinomial mixture, K=%d (2-NN)", K_sel),
       x=expression(theta[D]~"(Dem. fraction)"),
       y=expression(theta[R]~"(Rep. fraction)")) +
  theme_bw() +
  theme(legend.position="right", plot.title=element_text(size=11, hjust=0.5),
        axis.title=element_text(size=10), axis.text=element_text(size=8),
        panel.grid=element_blank())

# ---------------------------------------------------------------
# 7.  Save
# ---------------------------------------------------------------
if (!dir.exists("images")) dir.create("images")

ggsave("images/rd_election_trajectories.png", fig_traj, width=15, height=4.5, dpi=150)
cat("Saved: images/rd_election_trajectories.png\n")
ggsave("images/rd_election_khat.png", fig_khat, width=7, height=5, dpi=150)
cat("Saved: images/rd_election_khat.png\n")
ggsave("images/rd_election_scatter.png", fig_scatter, width=6.5, height=5.5, dpi=150)
cat("Saved: images/rd_election_scatter.png\n")
