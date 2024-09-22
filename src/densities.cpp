#include "gfuse.h"


/*** Multivariate Location Normal auxiliary functions begin. ***/

 
double densityNormalLoc (const Matrix<double, 1, Dynamic>& y,   
                         const Matrix<double, Dynamic, 1>& theta,   
                         const MatrixXd& sigma) {
  double d1 = y * theta;
  double d2 = theta.transpose() * sigma.transpose() * theta;
  double d3 = y * sigma.inverse() * y.transpose();

  return exp( d1 - 0.5 * d2 -0.5 * d3) / sqrt( fabs( (2 * M_PI * sigma).determinant())) ; 
}

MatrixXd gradBNormalLoc (const MatrixXd& theta, const MatrixXd& sigma) {
  return (sigma + sigma.transpose()) * theta / 2.0; 
}

double bNormalLoc (const VectorXd& theta, const MatrixXd& sigma) {
  return 0.5 * theta.transpose() * sigma * theta;
}

MatrixXd tNormalLoc (const MatrixXd& y) {
  return y;
}

MatrixXd transfNormalLoc (const MatrixXd& theta, const MatrixXd& sigma) {
  return sigma.inverse() * theta;
}

MatrixXd invTransfNormalLoc (const MatrixXd& theta, const MatrixXd& sigma) {
  return sigma * theta;
}

bool constrCheckNormalLoc (const MatrixXd& theta) {
  return true;
}

int dfNormalLoc (int k, bool arbSigma) {
  if (arbSigma) {
    return (k * (D+1) - 1 + D*(D+1)/2);
  }

  return (k * (D+1) -1);
}

/*** Multivariate Location T auxiliary functions begin. ***/


double densityT (const Matrix<double, 1, Dynamic>& y,   
                         const Matrix<double, Dynamic, 1>& theta,   
                         const MatrixXd& sigma, const double nu) {
  double d1 = y * theta;
  double d2 = theta.transpose() * sigma.transpose() * theta;
  double d3 = y * sigma.inverse() * y.transpose();
  double result;
  
  int D = theta.rows();
  double vp=0.5*(nu+D);
  
  result = std::tgamma(vp)*pow((1+( -2*d1 + d2 + d3)/nu),-vp) / (std::tgamma(nu/2) * sqrt( fabs( ( nu * M_PI * sigma).determinant()))) ; 
  //Rcpp::Rcout << "densityT" << result;
  return result;
  }

MatrixXd gradBT (const MatrixXd& theta, const MatrixXd& sigma) {
  return (sigma + sigma.transpose()) * theta / 2.0; 
}

double bT (const VectorXd& theta, const MatrixXd& sigma) {
  return 0.5 * theta.transpose() * sigma * theta;
}

MatrixXd tT (const MatrixXd& y) {
  return y;
}

MatrixXd transfT (const MatrixXd& theta, const MatrixXd& sigma) {
  return sigma.inverse() * theta;
}

MatrixXd invTransfT (const MatrixXd& theta, const MatrixXd& sigma) {
  return sigma * theta;
}

bool constrCheckT (const MatrixXd& theta) {
  return true;
}

int dfT (int k, bool arbSigma) {
  if (arbSigma) {
    return (k * (D+1) - 1 + D*(D+1)/2);
  }
  
  return (k * (D+1) -1);
}


/*** Multinomial auxiliary functions begin. ***/

double densityMultinomial(const Matrix<double, 1, Dynamic>& y,
                          const Matrix<double, Dynamic, 1>& theta,
                          const MatrixXd& sigma) {
  double logMFact = lgamma(M + 1);
  double logYSum = 0.0, logYDiff = 0.0, logThetaSum = 0.0, ySum = 0.0;
  double temp = 1;


  for(int l = 0; l < D; l++){
    logYSum      += lgamma(y(0, l) + 1);
    ySum         += y(0, l);
    temp         += exp(theta(l, 0));
    logThetaSum  += theta(l, 0);
  }

  logYDiff = lgamma(M - ySum + 1);

  return exp( logMFact - logYSum - logYDiff + y * theta /*+ (M - ySum) * (exp(1 - logThetaSum) / (temp)) */) * pow(temp, -M);
}

MatrixXd transfMultinomial (const MatrixXd& theta) {
  int K = theta.cols();
  MatrixXd out(D, K);
  int j, k;
  double temp;

  for (k = 0; k < K; k++) {
    temp = 1.0;

    for (j = 0; j < D; j++) {
      temp -= theta(j, k);
    }

    for(j = 0; j < D; j++) {
      out(j, k) = log(theta(j, k) / temp);
    }
  }

  return out;
}

MatrixXd invTransfMultinomial (const MatrixXd& theta, const MatrixXd& sigma) {
  int K = theta.cols();  // Number of mixture components. 
  int D = theta.rows();  // Dimension of the parameter space.
  
  MatrixXd out (D, K);
  int j, k;
  double temp;

  for(k = 0; k < K; k++) {
    temp = 1.0;

    for(j = 0; j < D; j++) {
      temp += exp(theta(j, k));
    }

    for(j = 0; j < D; j++) {
      out(j, k) = exp(theta(j, k)) / temp;
    }
  }

  return out;
}

MatrixXd tMultinomial(const MatrixXd& y) {
  return y;
}

MatrixXd gradBMultinomial (const MatrixXd& theta, const MatrixXd& sigma) {
  int K = theta.cols();  // Number of mixture components. 
  int D = theta.rows();  // Dimension of the parameter space.
  
  MatrixXd out(D, K);
  int j, k;
  double temp;

  for (k = 0; k < K; k++) {
    temp = 1.0;

    for (j = 0; j < D; j++) {
      temp += exp(theta(j, k));
    }

    for(j = 0; j < D; j++) {
      out(j, k) = M * exp(theta(j, k)) / temp;
    }
  }

  return out;
}

MatrixXd hesBMultinomial (const VectorXd& theta, const MatrixXd& sigma) {
  int D = theta.rows();  // Dimension of the parameter space.
  
  MatrixXd out(D,D);
  int j;
  double temp;
  MatrixXd exptheta(D,1), diag = MatrixXd::Zero(D,D);

    temp = 1.0;
    
    for (j = 0; j < D; j++) {
      temp += exp(theta(j));
      diag(j,j) = exp(theta(j));
      exptheta(j,0) = exp(theta(j));
    }
    
    out= M * (1/temp * diag - 1/pow(temp,2) * exptheta * exptheta.transpose());
  
  return out;
}

double bMultinomial (const VectorXd& theta, const MatrixXd& sigma) {
  D = theta.rows();  // Dimension of the parameter space.
  
  double temp = 1.0;
  int j;

  for (j = 0; j < D; j++) {
    temp += exp(theta(j));
  }
  
  return M * log(temp);
}

bool constrCheckMultinomial (const MatrixXd& theta) {
  return true;
}

/*** Poisson auxiliary functions begin. ***/

// Helper.
int factorial (int n) { return (n == 1 || n == 0) ? 1 : n * factorial(n - 1);}

double densityPoisson (const Matrix<double, 1, Dynamic>& y, const Matrix<double, Dynamic, 1>& theta, const MatrixXd& sigma) {
  return exp(theta(0, 0) * y(0, 0) - exp(theta(0, 0))) / (factorial(y(0, 0)));
}

double bPoisson (const VectorXd& theta, const MatrixXd& sigma) {
  return exp(theta(0, 0));
}

MatrixXd gradBPoisson (const MatrixXd& theta, const MatrixXd& sigma) {
  return theta.array().exp();
}

MatrixXd transfPoisson (const MatrixXd& theta) {
  return theta.array().log();
}

MatrixXd invTransfPoisson (const MatrixXd& theta, const MatrixXd& sigma) {
  return theta.array().exp();
}

MatrixXd tPoisson (const MatrixXd& y) {
  return y;
}

bool constrCheckPoisson (const MatrixXd& theta) {
  return true;
}

/*** Exponential distribution auxiliary functions begin.  ***/
double densityExponential (const Matrix<double, 1, Dynamic>& y, const Matrix<double, Dynamic, 1>& theta, const MatrixXd& sigma) {
  return exp(theta(0, 0) * y(0, 0) + log(-theta(0, 0)));
}

double bExponential (const VectorXd& theta, const MatrixXd& sigma) {
  return - log( - theta(0, 0));
}

MatrixXd gradBExponential (const MatrixXd& theta, const MatrixXd& sigma) {
  return - theta.array().inverse();
}

MatrixXd transfExponential (const MatrixXd& theta) {
  return -theta;
}

MatrixXd invTransfExponential (const MatrixXd& theta, const MatrixXd& sigma) {
  return -theta;
}

MatrixXd tExponential (const MatrixXd& y) {
  return y;
}

bool constrCheckExponential (const MatrixXd& theta) {
  return true;
}



