# real_data_crabs.R
#
# Real data analysis: Leptograpsus variegatus rock-crab morphometrics
# Source: Campbell & Mahon (1974), cited in Venables & Ripley (1999)
# Available in the MASS R package as "crabs"
#
# Data: n=200 crabs, 5 morphological measurements (FL, RW, CL, CW, BD)
# Groups (K_true=4): Blue × Female, Blue × Male, Orange × Female, Orange × Male
#
# Preprocessing: log-transform → PCA → use PC2 and PC3 (shape, after removing
# the dominant size factor PC1).  PC2 captures the sex axis, PC3 the species axis.
#
# Applied method: normalLocOrder + tLocOrder (robust to outliers / skewness)

library(GFuse)
library(MASS)
library(ggplot2)
library(dplyr)

set.seed(2025)

# ---------------------------------------------------------------
# 1.  Load and preprocess
# ---------------------------------------------------------------
data(crabs)
K_true <- 4
group  <- interaction(crabs$sp, crabs$sex)  # factor: B.F, B.M, O.F, O.M

# log-transform → PCA (scale=TRUE; removes dominant size factor PC1)
Ylog <- log(crabs[, 4:8])
pc   <- prcomp(Ylog, scale=TRUE)

# Use PC2 and PC3 — the shape axes
X <- pc$x[, 2:3]
colnames(X) <- c("PC2","PC3")

cat("Crabs morphometric data: n=200, 4 groups (species × sex)\n")
cat("2D representation: PC2 (sex axis) vs PC3 (species axis)\n")
cat("True group centroids:\n")
centroids <- aggregate(X, by=list(Group=group), FUN=mean)
centroids[, -1] <- round(centroids[, -1], 3)
print(centroids)
cat("\n")

# ---------------------------------------------------------------
# 2.  Settings
# ---------------------------------------------------------------
lambdas  <- c(0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8)
lmax     <- max(lambdas)
K_start  <- 8
nu_t     <- 5

method_specs <- list(
  list(graphtype="MNN", m=1, label="1-NN", short="MNN1"),
  list(graphtype="MNN", m=2, label="2-NN", short="MNN2"),
  list(graphtype="MNN", m=3, label="3-NN", short="MNN3"),
  list(graphtype="GSF", m=1, label="GSF",        short="GSF"),
  list(graphtype="MST", m=1, label="MST",         short="MST")
)

model_specs <- list(
  list(type="gauss", label="Gaussian"),
  list(type="t",     label=sprintf("Student-t (nu=%d)", nu_t))
)

# ---------------------------------------------------------------
# 3.  Fit all 10 combinations
# ---------------------------------------------------------------
all_fits <- list()

for (md in model_specs) {
  for (mt in method_specs) {
    key <- paste0(md$type, "_", mt$short)
    cat(sprintf("Fitting %s — %s ...\n", md$label, mt$label))

    fit <- if (md$type == "gauss") {
      normalLocOrder(X, m=mt$m, lambdas=lambdas, K=K_start,
                     graphtype=mt$graphtype, verbose=FALSE,
                     maxMem=100, maxadmm=80)
    } else {
      tLocOrder(X, m=mt$m, lambdas=lambdas, K=K_start, df=nu_t,
                graphtype=mt$graphtype, verbose=FALSE,
                maxMem=100, maxadmm=80)
    }

    bic  <- bicTuning(X, fit)
    khat <- cummin(vapply(fit, function(x) x[["order"]], integer(1)))

    all_fits[[key]] <- list(
      fit=fit, bic=bic, khat_path=khat,
      model=md$label, model_short=md$type,
      method=mt$label, mt_short=mt$short,
      K_true=K_true
    )
    cat(sprintf("  → BIC: K = %d at lambda = %.3f  (K_true = %d)\n",
                bic$result$order, bic$result$lambda, K_true))
  }
}

# ---------------------------------------------------------------
# 4.  Summary table
# ---------------------------------------------------------------
cat("\n=== BIC-selected K summary ===\n")
tab <- do.call(rbind, lapply(names(all_fits), function(k) {
  fo <- all_fits[[k]]
  data.frame(Model=fo$model, Method=fo$method,
             K_BIC=fo$bic$result$order,
             lambda_BIC=round(fo$bic$result$lambda, 3),
             Correct=(fo$bic$result$order == K_true))
}))
print(tab)

# First-correct lambda
khat_df <- do.call(rbind, lapply(names(all_fits), function(k) {
  fo <- all_fits[[k]]
  data.frame(lambda=lambdas, khat=fo$khat_path,
             model=fo$model, method=fo$method, K_true=K_true)
}))
first_correct <- khat_df %>% group_by(model, method) %>%
  filter(khat==K_true) %>% slice(1) %>% ungroup()
cat("\nFirst lambda where K_hat = K_true:\n")
print(as.data.frame(first_correct[order(first_correct$model,first_correct$lambda),
                                  c("model","method","lambda","khat")]))

# ---------------------------------------------------------------
# 5.  K_hat vs lambda
# ---------------------------------------------------------------
ktrue_df <- data.frame(K_true=K_true)
method_cols <- c("1-NN"="#2171b5","2-NN"="#6baed6",
                 "3-NN"="#bdd7e7","GSF"="#d94801","MST"="#7a0177")
method_lty  <- c("1-NN"="solid","2-NN"="solid","3-NN"="solid",
                 "GSF"="dashed","MST"="dotdash")

fig_khat <- ggplot(khat_df, aes(x=lambda, y=khat, colour=method, linetype=method)) +
  geom_line(linewidth=0.9) + geom_point(size=1.6) +
  geom_hline(yintercept=K_true, linetype="dashed", colour="firebrick2", linewidth=0.8) +
  geom_vline(data=first_correct, aes(xintercept=lambda, colour=method),
             linetype="dotted", linewidth=0.8, alpha=0.7) +
  facet_wrap(~model, ncol=2) +
  scale_colour_manual(values=method_cols, name="Graph") +
  scale_linetype_manual(values=method_lty,  name="Graph") +
  scale_x_log10(breaks=c(0.01,0.05,0.1,0.2,0.5,1.0,2.0)) +
  scale_y_continuous(breaks=1:K_start) +
  labs(title="Crabs morphometrics — estimated K vs. λ (PC2 vs. PC3 space)",
       x=expression(lambda~"(log scale)"), y=expression(hat(K))) +
  theme_bw() +
  theme(legend.position="bottom", strip.text=element_text(size=12, face="bold"),
        plot.title=element_text(size=11, hjust=0.5, face="bold"),
        axis.title=element_text(size=10), panel.grid.minor=element_blank())

# ---------------------------------------------------------------
# 6.  2D solution path panels
# ---------------------------------------------------------------
traj_all <- do.call(rbind, lapply(names(all_fits), function(k) {
  fo <- all_fits[[k]]
  do.call(rbind, lapply(seq_along(fo$fit), function(i) {
    mu <- fo$fit[[i]]$mu
    data.frame(lambda_idx=i, lambda=fo$fit[[i]]$lambda,
               atom_id=seq_len(ncol(mu)),
               x1=mu[1,], x2=mu[2,],
               model=fo$model, method=fo$method,
               model_short=fo$model_short, mt_short=fo$mt_short)
  }))
}))

seg_all <- traj_all %>%
  arrange(model, method, atom_id, lambda_idx) %>%
  group_by(model, method, atom_id) %>%
  mutate(x1e=lead(x1), x2e=lead(x2), lmid=(lambda+lead(lambda))/2) %>%
  filter(!is.na(x1e)) %>% ungroup()

# True group centroids for overlay
true_df <- as.data.frame(aggregate(X, by=list(Group=group), FUN=mean))
names(true_df) <- c("group","x1","x2")
Xdf     <- data.frame(x1=X[,1], x2=X[,2])

make_panel <- function(model_label, method_label, show_legend=FALSE) {
  seg <- seg_all %>% filter(model==model_label, method==method_label)
  tra <- traj_all %>% filter(model==model_label, method==method_label)
  mk  <- tra$mt_short[1]; mdk <- tra$model_short[1]
  fo  <- all_fits[[paste0(mdk,"_",mk)]]

  # First-correct lambda for white square
  kp  <- fo$khat_path
  fci <- which(kp == fo$K_true)
  bic_df <- if (length(fci)>0) {
    tra %>% filter(lambda_idx == fci[1])
  } else {
    data.frame(atom_id=integer(0), x1=numeric(0), x2=numeric(0),
               lambda_idx=integer(0), lambda=numeric(0),
               model=character(0), method=character(0),
               model_short=character(0), mt_short=character(0))
  }

  ggplot() +
    geom_point(data=Xdf, aes(x=x1, y=x2),
               colour="grey82", size=0.4, alpha=0.35) +
    geom_segment(data=seg,
                 aes(x=x1, y=x2, xend=x1e, yend=x2e, colour=lmid),
                 linewidth=1.1, alpha=0.9, lineend="round") +
    geom_point(data=tra %>% filter(lambda_idx==1), aes(x=x1, y=x2),
               shape=16, size=1.8, colour="#c94a00") +
    geom_point(data=bic_df, aes(x=x1, y=x2),
               shape=22, size=2.8, fill="white", colour="black", stroke=1.1) +
    geom_point(data=true_df, aes(x=x1, y=x2),
               shape=3, size=4.0, colour="black", stroke=1.7) +
    scale_colour_gradient(low="#d94801", high="#08306b",
                          name=expression(lambda), limits=c(0,lmax),
                          breaks=c(0,0.5,1.0,1.5)) +
    coord_fixed(ratio=1) +
    labs(title=method_label, x="PC2 (sex axis)", y="PC3 (species axis)") +
    theme_bw() +
    theme(legend.position=if(show_legend)"right" else "none",
          legend.key.height=unit(0.9,"cm"),
          plot.title=element_text(size=10, hjust=0.5, face="bold"),
          axis.title=element_text(size=9), axis.text=element_text(size=7),
          panel.grid=element_blank())
}

method_labels <- c("1-NN","2-NN","3-NN","GSF","MST")

panels_gauss <- lapply(seq_along(method_labels), function(i)
  make_panel("Gaussian", method_labels[i], show_legend=(i==5)))
panels_t <- lapply(seq_along(method_labels), function(i)
  make_panel(sprintf("Student-t (nu=%d)",nu_t), method_labels[i], show_legend=(i==5)))

combine_rows <- function(p1, p2) {
  add_lab <- function(p, lab)
    p + labs(subtitle=lab) +
      theme(plot.subtitle=element_text(size=8, face="italic", hjust=0))
  panels_all <- c(lapply(seq_along(p1), function(i)
                    if(i==1) add_lab(p1[[i]],"Gaussian") else p1[[i]]),
                  lapply(seq_along(p2), function(i)
                    if(i==1) add_lab(p2[[i]],sprintf("Student-t (nu=%d)",nu_t)) else p2[[i]]))
  if (requireNamespace("patchwork", quietly=TRUE)) {
    library(patchwork)
    Reduce(`+`, panels_all) + patchwork::plot_layout(ncol=5, nrow=2)
  } else {
    library(gridExtra)
    gridExtra::arrangeGrob(grobs=panels_all, ncol=5, nrow=2)
  }
}

fig_traj <- combine_rows(panels_gauss, panels_t)

# ---------------------------------------------------------------
# 7.  Cluster scatter — Gaussian MNN(m=2)
# ---------------------------------------------------------------
fo_best <- all_fits[["gauss_MNN2"]]
blam    <- fo_best$bic$result$lambda
bidx    <- which.min(abs(lambdas - blam))
mu_sel  <- fo_best$fit[[bidx]]$mu
K_sel   <- ncol(mu_sel)
pii_sel <- fo_best$fit[[bidx]]$pii

cls <- apply(X, 1, function(x)
  which.min(apply(mu_sel, 2, function(m) sum((x-m)^2))))

cluster_cols <- c("#e41a1c","#377eb8","#4daf4a","#984ea3",
                  "#ff7f00","#a65628","#f781bf","#999999")
Xdf_cls <- data.frame(x1=X[,1], x2=X[,2],
                      cluster=factor(cls), group=group)
mu_df   <- data.frame(x1=mu_sel[1,], x2=mu_sel[2,], cluster=factor(1:K_sel))

cat(sprintf("\nGaussian MNN(m=2): K = %d selected\n", K_sel))
cat("Contingency of estimated cluster vs true group:\n")
print(table(Estimated=cls, True=group))

fig_scatter <- ggplot(Xdf_cls, aes(x=x1, y=x2, colour=cluster, shape=group)) +
  geom_point(size=2.0, alpha=0.7) +
  geom_point(data=mu_df, aes(x=x1, y=x2, colour=cluster),
             shape=4, size=5, stroke=2.0, inherit.aes=FALSE) +
  scale_colour_manual(values=cluster_cols[1:K_sel], name="Est. cluster") +
  scale_shape_manual(values=c(16,17,15,18), name="True group") +
  coord_fixed(ratio=1) +
  labs(title=sprintf("Crabs morphometrics — Gaussian 2-NN, K=%d", K_sel),
       subtitle="Shape = true species×sex group;  Color = estimated cluster",
       x="PC2 (sex axis)", y="PC3 (species axis)") +
  theme_bw() +
  theme(legend.position="right", plot.title=element_text(size=11, hjust=0.5, face="bold"),
        plot.subtitle=element_text(size=9, hjust=0.5),
        axis.title=element_text(size=10), panel.grid=element_blank())

# ---------------------------------------------------------------
# 8.  Save
# ---------------------------------------------------------------
if (!dir.exists("images")) dir.create("images")
ggsave("images/rd_crabs_trajectories.png", fig_traj, width=15, height=7.5, dpi=150)
cat("Saved: images/rd_crabs_trajectories.png\n")
ggsave("images/rd_crabs_khat.png", fig_khat, width=11, height=5, dpi=150)
cat("Saved: images/rd_crabs_khat.png\n")
ggsave("images/rd_crabs_scatter.png", fig_scatter, width=7, height=5.5, dpi=150)
cat("Saved: images/rd_crabs_scatter.png\n")
