#include "gsf.h"

MatrixXd normaltheta (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx, const MatrixXd& Eta, const MatrixXd& U){
  int K = psi.theta.cols();
  int k,d,i,j;
  MatrixXd Theta =  MatrixXd::Zero(D,K);
  MatrixXd A(D,D), 
  B(D,1),
  C(D,1);
  MatrixXd I = MatrixXd::Zero(D,D);
  
  VectorXd wMtxSums(K);
  
  for(k = 0; k < K; k++) {
    wMtxSums(k) = wMtx.col(k).sum();
  }
  
  for(d=0 ; d < D; d++){
    I(d,d)=1 ;
  }
  for(k = 0; k < K; k++) {
    A = wMtxSums[k]*psi.sigma+graph.col(k).sum()*I;
    //Rcpp::Rcout << "A" << A <<"\n";
    //Rcpp::Rcout << "A.inverse" << Ainv <<"\n";
    B = MatrixXd::Zero(D,1);
    //Rcpp::Rcout << "Bstart" << B <<"\n";
    for (i=0; i < n; i++){
      B = B + wMtx(i,k)*y.row(i).transpose();
      //Rcpp::Rcout << "B2" << B.row(1)<< "w" << wMtx(i,k) << "y" << y.row(i);
    }
    C = MatrixXd::Zero(D,1);
    for (j = 0 ; j < K; j++){
      if (graph(k,j)==1){
        //Rcpp::Rcout << "k=" << k << "j=" << j <<"\n";
        C = C + Eta.col(k+K*j)-U.col(k+K*j);
      }
    }
    Theta.col(k) = A.ldlt().solve(B + C) ;
    //Rcpp::Rcout << "B" << B <<"\n";
    //Rcpp::Rcout << "C" << C <<"\n";
    //Rcpp::Rcout << "Theta.col" << Theta.col(k) <<"\n";
  } 
  return Theta;
}

MatrixXd normaltheta0 (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx, const MatrixXd& Eta, const MatrixXd& U){
  int K = psi.theta.cols();
  int k,d,i,j;
  MatrixXd Theta =  MatrixXd::Zero(D,K);
  MatrixXd A(D,D), 
  B(D,1);
  MatrixXd I = MatrixXd::Zero(D,D);
  
  VectorXd wMtxSums(K);
  
  for(k = 0; k < K; k++) {
    wMtxSums(k) = wMtx.col(k).sum();
  }
  
  for(d=0 ; d < D; d++){
    I(d,d)=1 ;
  }
  for(k = 0; k < K; k++) {
    A = wMtxSums[k]*psi.sigma+graph.col(k).sum()*I;
    //Rcpp::Rcout << "A" << A <<"\n";
    //Rcpp::Rcout << "A.inverse" << Ainv <<"\n";
    B = MatrixXd::Zero(D,1);
    //Rcpp::Rcout << "Bstart" << B <<"\n";
    for (i=0; i < n; i++){
      B = B + wMtx(i,k)*y.row(i).transpose();
      //Rcpp::Rcout << "B2" << B.row(1)<< "w" << wMtx(i,k) << "y" << y.row(i);
    }
    Theta.col(k) = A.ldlt().solve(B) ;
    //Rcpp::Rcout << "B" << B <<"\n";
    //Rcpp::Rcout << "Theta.col" << Theta.col(k) <<"\n";
  } 
  return Theta;
}

MatrixXd Ttheta (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx1, const MatrixXd& wMtx2, const MatrixXd& Eta, const MatrixXd& U){
  int K = psi.theta.cols();
  int k,d,i,j;
  MatrixXd Theta =  MatrixXd::Zero(D,K);
  MatrixXd A(D,D), 
  B(D,1),
  C(D,1);
  MatrixXd I = MatrixXd::Zero(D,D);
  
  VectorXd wMtxSums(K);
  MatrixXd wMtx(n,K);
  for(k = 0; k < K; k++) {
    for(i=0 ; i < n; i++){
      wMtx(i,k) = wMtx1(i,k) * wMtx2(i,k);
    }}
  
  
  for(k = 0; k < K; k++) {
    wMtxSums(k) = wMtx.col(k).sum();
  }
  
  for(d=0 ; d < D; d++){
    I(d,d)=1 ;
  }
  for(k = 0; k < K; k++) {
    A = wMtxSums[k]*psi.sigma+graph.col(k).sum()*I;
    //Rcpp::Rcout << "A" << A <<"\n";
    //Rcpp::Rcout << "A.inverse" << Ainv <<"\n";
    B = MatrixXd::Zero(D,1);
    //Rcpp::Rcout << "Bstart" << B <<"\n";
    for (i=0; i < n; i++){
      B = B + wMtx(i,k)*y.row(i).transpose();
      //Rcpp::Rcout << "B2" << B.row(1)<< "w" << wMtx(i,k) << "y" << y.row(i);
    }
    C = MatrixXd::Zero(D,1);
    for (j = 0 ; j < K; j++){
      if (graph(k,j)==1){
        //Rcpp::Rcout << "k=" << k << "j=" << j <<"\n";
        C = C + Eta.col(k+K*j)-U.col(k+K*j);
      }
    }
    Theta.col(k) = A.ldlt().solve(B + C) ;
    Rcpp::Rcout << "B" << B <<"\n";
    Rcpp::Rcout << "C" << C <<"\n";
    Rcpp::Rcout << "Theta.col" << Theta.col(k) <<"\n";
  } 
  return Theta;
}

MatrixXd Ttheta0 (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx1,const MatrixXd& wMtx2, const MatrixXd& Eta, const MatrixXd& U){
  int K = psi.theta.cols();
  int k,d,i,j;
  MatrixXd Theta =  MatrixXd::Zero(D,K);
  MatrixXd A(D,D), 
  B(D,1);
  MatrixXd I = MatrixXd::Zero(D,D);
  
  VectorXd wMtxSums(K);
  MatrixXd wMtx(n,K);
  for(k = 0; k < K; k++) {
    for(i=0 ; i < n; i++){
      wMtx(i,k) = wMtx1(i,k) * wMtx2(i,k);
    }}
  
  for(k = 0; k < K; k++) {
    wMtxSums(k) = wMtx.col(k).sum();
  }
  
  for(d=0 ; d < D; d++){
    I(d,d)=1 ;
  }
  for(k = 0; k < K; k++) {
    A = wMtxSums[k]*psi.sigma+graph.col(k).sum()*I;
    //Rcpp::Rcout << "A" << A <<"\n";
    //Rcpp::Rcout << "A.inverse" << Ainv <<"\n";
    B = MatrixXd::Zero(D,1);
    //Rcpp::Rcout << "Bstart" << B <<"\n";
    for (i=0; i < n; i++){
      B = B + wMtx(i,k)*y.row(i).transpose();
      //Rcpp::Rcout << "B2" << B.row(1)<< "w" << wMtx(i,k) << "y" << y.row(i);
    }
    Theta.col(k) = A.ldlt().solve(B) ;
    //Rcpp::Rcout << "B" << B <<"\n";
    //Rcpp::Rcout << "Theta.col" << Theta.col(k) <<"\n";
  } 
  return Theta;
}

MatrixXd multinomialtheta (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx, const MatrixXd& Eta, const MatrixXd& U){
  int K = psi.theta.cols();
  int i,k,j, counter = 0;
  MatrixXd newTheta = MatrixXd::Zero(D, K), 
    oldTheta= psi.theta;
  MatrixXd gradB(D,K), gradh(D,1), hesh(D,D);
  MatrixXd A(D,D),
  B(D,1),
  C(D,1);
  MatrixXd I = MatrixXd::Zero(D,D);
  
  VectorXd wMtxSums(K);
  
  for(k = 0; k < K; k++) {
    wMtxSums(k) = wMtx.col(k).sum();
  }
  
  for(j=0 ; j < D; j++){
    I(j,j)=1 ;
  }
  
  //Rcpp::Rcout << "oldTheta" << oldTheta.row(0);
  
  do {
    oldTheta = newTheta ;   
    gradB = gradBMultinomial(oldTheta, psi.sigma);
  
  for(k = 0; k < K; k++) {
    gradh = MatrixXd::Zero(D,1);
    hesh = MatrixXd::Zero(D,D);
    
    A = wMtxSums[k]* gradB.col(k);
    //Rcpp::Rcout << "A" << A <<"\n";
    B = MatrixXd::Zero(D,1);
    
    for (i=0; i < n; i++){
      B = B + wMtx(i,k)*y.row(i).transpose();
      //Rcpp::Rcout << "B2" << B.row(1)<< "w" << wMtx(i,k) << "y" << y.row(i);
    }
    //Rcpp::Rcout << "B" << B <<"\n";
    C = MatrixXd::Zero(D,1);
    for (j = 0 ; j < K; j++){
      if (graph(k,j)==1){
        //Rcpp::Rcout << "k=" << k << "j=" << j <<"\n";
        C = C + oldTheta.col(k) - Eta.col(k+K*j) + U.col(k+K*j);
      }
    }
    //Rcpp::Rcout << "B" << B <<"\n";
    gradh = A - B + C ;
    hesh = wMtxSums[k]* hesBMultinomial(oldTheta.col(k), psi.sigma) + graph.col(k).sum()*I;
    //Rcpp::Rcout << "hessian" << hesh <<"\n";
  
  newTheta.col(k) = oldTheta.col(k) - hesh.ldlt().solve(gradh);
  }
  //Rcpp::Rcout << "diffintheta" << (oldTheta - newTheta).norm()<< "\n";
  } while (counter++ < maxNR && (oldTheta - newTheta).norm() > delta);
  //Rcpp::Rcout << "counterNR" << counter<< "\n";
  
  //Rcpp::Rcout << "newTheta" << newTheta.row(0);
  
  return newTheta;
  }

MatrixXd multinomialtheta0 (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx, const MatrixXd& Eta, const MatrixXd& U){
  int K = psi.theta.cols();
  int i,k,j, counter = 0;
  MatrixXd newTheta = MatrixXd::Zero(D, K), 
    oldTheta= psi.theta;
  MatrixXd gradB(D,K), gradh(D,1), hesh(D,D);
  MatrixXd A(D,D),
  B(D,1),
  C(D,1);
  MatrixXd I = MatrixXd::Zero(D,D);
  
  VectorXd wMtxSums(K);
  
  for(k = 0; k < K; k++) {
    wMtxSums(k) = wMtx.col(k).sum();
  }
  
  for(j=0 ; j < D; j++){
    I(j,j)=1 ;
  }
  
  //Rcpp::Rcout << "oldTheta" << oldTheta.row(0);
    
    for(k = 0; k < K; k++) {
      gradh = MatrixXd::Zero(D,1);
      hesh = MatrixXd::Zero(D,D);
      
      A = wMtxSums[k]* gradB.col(k);
      //Rcpp::Rcout << "A" << A <<"\n";
      B = MatrixXd::Zero(D,1);
      
      for (i=0; i < n; i++){
        B = B + wMtx(i,k)*y.row(i).transpose();
        //Rcpp::Rcout << "B2" << B.row(1)<< "w" << wMtx(i,k) << "y" << y.row(i);
      }
      newTheta.col(k) = A.ldlt().solve(B) ;
    }
  
  return newTheta;
}


double etamax(const Matrix<double, 1, Dynamic>& z, double lambda){
  double normZ = z.norm(), u;
  //Rcpp::Rcout << "normZ" << normZ <<"\n";
  //Rcpp::Rcout << "lambda" << lambda <<"\n";
  //Rcpp::Rcout << "1/normZ" << (1.0/normZ) <<"\n";
  u = 1-(1.0/normZ) * lambda;
  //Rcpp::Rcout << "valuephi" << u <<"\n";
  if( u>0.5) {
    return u;
  } else {
    return 0.5;
  }
}

VectorXd softThresholding(const VectorXd& z, double lambda){
  double c = 1 - (lambda / z.norm());

  if (c > 0) return c * z;
  else return VectorXd::Zero(D);
}

VectorXd scadUpdate(double u, const Matrix<double, 1, Dynamic>& z, double lambda, double a){
  double normZ = z.norm();

  if(normZ <= (u+1) * lambda){
    return softThresholding(z, u * lambda);

  } else if( (u + 1) * lambda <= normZ && normZ < a * lambda) {
    return ((a - 1)/(a - u - 1)) * softThresholding(z, (a * u * lambda)/(a - 1));

  } else {
    return z;
  }
}

VectorXd mcpUpdate(double u, const Matrix<double, 1, Dynamic>& z, double lambda, double a){
  double normZ = z.norm();

  if(normZ <= a * lambda){
    return (a/(a - u)) * softThresholding(z, u * lambda);

  } else {
    return z;
  }
}

// Note: in this case, "a" is the Adaptive Lasso weight. 
VectorXd adaptiveLassoUpdate(double u, const Matrix<double, 1, Dynamic>& z, double lambda, double a){
  return softThresholding(z, u * lambda * a);
}

VectorXd scadLLAUpdate(double u, const Matrix<double, 1, Dynamic>& z, const Matrix<double, 1, Dynamic>& eta, double lambda, double a){
  double scad, normEta = eta.norm();

  if (normEta <= lambda) {
    scad = lambda;

  } else if (lambda < normEta && normEta < a * lambda) {
    scad = (a * lambda - normEta) / (a - 1);

  } else {
    scad = 0;
  }

  return softThresholding(z, u * scad);
}

VectorXd mcpLLAUpdate(double u, const Matrix<double, 1, Dynamic>& z, const Matrix<double, 1, Dynamic>& eta, double lambda, double a){
  double mcp, normEta = eta.norm();

  if (normEta <= lambda * a) {
    mcp = (lambda - (normEta / a));

  } else {
    mcp = 0;
  }

  return softThresholding(z, u * mcp);
}

double GetSCAD(const Matrix<double, 1, Dynamic>& eta, double lambda, double a){
  double scad, normEta = eta.norm();
  
  if (normEta <= lambda) {
    scad = lambda;
    
  } else if (lambda < normEta && normEta < a * lambda) {
    scad = (a * lambda - normEta) / (a - 1);
    
  } else {
    scad = 0;
  }
  
  return scad;
}

double GetMCP(const Matrix<double, 1, Dynamic>& eta, double lambda, double a){
  double mcp, normEta = eta.norm();
  
  if (normEta <= lambda * a) {
    mcp = (lambda - (normEta / a));
    
  } else {
    mcp = 0;
  }
  
  return mcp;
}

double GetUnweight(const Matrix<double, 1, Dynamic>& eta, double lambda, double a){
  return lambda;
}

double GetALasso(const Matrix<double, 1, Dynamic>& eta, double lambda, double a){
  return lambda*a;
}
