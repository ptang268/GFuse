% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gf.R
\name{tLocOrder}
\alias{tLocOrder}
\title{Estimate the Number of Components in a Multivariate t-distribution Location Mixture Model.}
\usage{
tLocOrder(
  y,
  m = 1,
  lambdas,
  df = 1,
  graphtype = "MNN",
  K = NULL,
  sigma = NULL,
  arbSigma = TRUE,
  ...
)
}
\arguments{
\item{y}{n by D matrix consisting of the data, where n is the sample size 
and D is the dimension.}

\item{m}{The number of nearest neighbors in "\code{MNN}"; the lower bound \code{K_0} in "\code{MST}" or "\code{GSF}" for the graph-trimming procedure. Default is 1.}

\item{lambdas}{Vector of tuning parameter values.}

\item{df}{The known degree of freedom in the t-distribution mixture models. Default is 1.}

\item{graphtype}{Choice of the atom graph, which may be "\code{MNN}","\code{MST}" or "\code{GSF}". Default is "\code{MNN}".}

\item{K}{Upper bound on the true number of components. 
If \code{K} is \code{NULL}, at least one of \code{theta} and
\code{pii} must be non-\code{NULL}, and K is inferred from their
number of columns.}

\item{sigma}{D by D matrix, which is the starting value for the common
variance-covariance matrix. If \code{NULL}, \code{arbSigma}
must be \code{TRUE}, and in this case the starting value is
set to be the sample variance-covariance of the data.}

\item{arbSigma}{Equals \code{TRUE} if the common variance-covariance matrix should
be estimated, and FALSE if it should stay fixed, and equal to
\code{sigma}.}

\item{...}{Additional control parameters. See the \strong{Details} section.}
}
\value{
An object with S3 classes \code{gf} and \code{tLocGf}, 
        consisting of a list with the estimates produced for every tuning
        parameter in \code{lambdas}.
}
\description{
Estimate the order of a finite mixture of multivariate t-distribution densities
with respect to the mean parameter, whose variance-covariance matrices
are common but potentially unknown.
}
\details{
The following is a list of additional control parameters.

 \describe{
  \item{\code{mu}}{D by K matrix of starting values where each column is the mean
                     vector for one component. If \code{theta=NULL}, the starting 
                     values are chosen using the procedure of Benaglia et al. (2009).}
  \item{\code{pii}}{Vector of size K whose elements must sum to 1, consisting of 
                     the starting values for the mixing proportions.  
                     If \code{NULL}, it will be set to a discrete 
                     uniform distribution with K support points.}
  \item{\code{penalty}}{Choice of penalty, which may be "\code{SCAD}", "\code{MCP}", 
                         "\code{SCAD-LLA}", "\code{MCP-LLA}" or "\code{ADAPTIVE-LASSO}". 
                         Default is "\code{SCAD}".}
  \item{\code{C}}{Tuning parameter for penalizing the mixing proportions.}
  \item{\code{a}}{Tuning parameter for the SCAD or MCP penalty. Default is \code{3.7}.}
  \item{\code{convMem}}{Convergence criterion for the modified EM algorithm.}
  \item{\code{convADMM}}{Convergence criterion for the ADMM algorithm.}
  \item{\code{maxMem}}{Maximum number of iterations of the modified EM algorithm.}
  \item{\code{maxadmm}}{Maximum number of iterations of the ADMM algorithm.}
  \item{\code{maxNR}}{Maximum number of iterations of the Newton-Ralghson method if needed.}
  \item{\code{dynamic}}{Whether to apply the Dynamic Cluster Merging update. Default is \code{TRUE}}
  \item{\code{verbose}}{If \code{TRUE}, print updates while the function is running.}}
}
\examples{
 # Example 1: Seeds Data.
   data(seeds) 
   y <- cbind(seeds[,2], seeds[,6])
   set.seed(1)
   out <- tLocOrder(y, K=12, lambdas=seq(0.1, 1.7, 0.3),  maxMem=500)
   tuning <- bicTuning(y, out)
   print(tuning)
  
 # Example 2: Old Faithful Data.
   data(faithful)
   set.seed(1)
   out <- tLocOrder(faithful, K=10, 
             lambdas=c(0.1, 0.25, 0.5, 0.75, 1.0, 2), penalty="MCP-LLA", a=2, maxMem=500)
   tuning <- bicTuning(y, out)
   print(tuning)
}
\references{
Tang, P., Khalili, A., Yang A. 2024 "Graph-guided Regularization for Order Estimation in Finite Mixture Models" 

 Manole, T., Khalili, A. 2019. "Estimating the Number of Components in Finite Mixture Models 
 via the Group-Sort-Fuse Procedure".
}
