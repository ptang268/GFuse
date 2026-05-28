## ---------------------------------------------------------------------------
## GFuse example: the solution path as a diagnostic for latent structure
##
## Run with:  source(system.file("examples", "solution_path.R", package = "GFuse"))
## Requires:  ggplot2  (install.packages("ggplot2"))
## ---------------------------------------------------------------------------

library(GFuse)
library(ggplot2)
set.seed(1)

## 1. Simulate a 4-component 2-D Gaussian mixture (square configuration) -------
mu <- cbind(c(-2, -2), c(-2, 2), c(2, -2), c(2, 2))
n  <- 400
z  <- sample(4, n, replace = TRUE)
y  <- t(vapply(seq_len(n), function(i) mu[, z[i]] + rnorm(2), numeric(2)))

## 2. Fit the graph-guided estimator along a path of lambda --------------------
##    Start from an over-specified K = 10; 2-NN graph; SCAD penalty.
lambdas <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)
out  <- normalLocOrder(y, m = 2, K = 10, lambdas = lambdas,
                       graphtype = "MNN", penalty = "SCAD")

## 3. Select an order by the modified BIC --------------------------------------
tune <- bicTuning(y, out)
message("Selected order K = ", tune$result$order,
        " at lambda = ", signif(tune$result$lambda, 3))

## 4a. Built-in coefficient-path plot (parameter vs lambda) --------------------
plot(out, gg = TRUE, eta = FALSE, vlines = TRUE, opt = tune$result$lambda)

## 4b. Parameter-space solution path: each atom's (mu1, mu2) trajectory --------
path <- do.call(rbind, lapply(seq_along(out), function(i) {
  M <- out[[i]]$mu
  data.frame(lambda = out[[i]]$lambda, atom = seq_len(ncol(M)),
             mu1 = M[1, ], mu2 = M[2, ])
}))

print(
  ggplot(path, aes(mu1, mu2, group = atom, colour = lambda)) +
    geom_path(linewidth = 1) + geom_point(size = 0.7) +
    geom_point(data = data.frame(mu1 = mu[1, ], mu2 = mu[2, ]),
               aes(mu1, mu2), inherit.aes = FALSE, shape = 3, size = 4) +
    scale_colour_viridis_c() +
    labs(title = sprintf("GFuse solution path (selected K = %d)", tune$result$order),
         subtitle = "crosses = true component means",
         x = expression(mu[1]), y = expression(mu[2])) +
    theme_bw()
)

## Reading the path from small to large lambda, atoms from the same
## subpopulation collapse onto a common location first, while well-separated
## subpopulations stay distinct until strong fusion -- a hierarchical, visual
## summary of the heterogeneity in the data.
