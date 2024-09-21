#include "gsf.h"

int m, N, D, n, M, Graphtype, maxadmm, maxNR,maxPgd, maxRep, lambdaIter, modelIndex, penalty;
double epsilon, u, ck, tau1, a, delta,  
lambdaScale, uBound, H, alStart;

bool arbSigma, verbose, dynamic;

MatrixXd upTransf, invEtaTransf, ones, transformedData;
std::vector<MatrixXd> yOuterProd; 
VectorXd lambdaVals, adaptiveLassoWeights;

MatrixXd (*gradB)(const MatrixXd&, const MatrixXd&);
double   (*b)(const VectorXd&, const MatrixXd&);
MatrixXd (*T)(const MatrixXd&);
double   (*density)(const Matrix<double, 1, Dynamic>&, const Matrix<double, Dynamic, 1>&, const MatrixXd&);
double   (*density2)(const Matrix<double, 1, Dynamic>&, const Matrix<double, Dynamic, 1>&, const MatrixXd&, const double);
MatrixXd (*invTransf)(const MatrixXd&, const MatrixXd&);
MatrixXd (*Graphmat)(const MatrixXd&, int m);
MatrixXd (*updateTheta)(const MatrixXd& , const Psi& psi, const MatrixXd&, const MatrixXd&, const MatrixXd&, const MatrixXd& );
MatrixXd (*updateTheta0)(const MatrixXd& , const Psi& psi, const MatrixXd&, const MatrixXd&, const MatrixXd&, const MatrixXd& );
//MatrixXd (*updateTheta2)(const MatrixXd& , const Psi& psi, const MatrixXd&, const MatrixXd&, const MatrixXd&, const MatrixXd&, const MatrixXd& );
//MatrixXd (*updateTheta02)(const MatrixXd& , const Psi& psi, const MatrixXd&, const MatrixXd&, const MatrixXd&, const MatrixXd&, const MatrixXd& );
VectorXd (*updateEta)(double, const Matrix<double, 1, Dynamic>&, double, double);
VectorXd (*updateEtaLLA)(double, const Matrix<double, 1, Dynamic>&, const Matrix<double, 1, Dynamic>&, double, double);
bool     (*constrCheck)(const MatrixXd&);
double (*updateLambda)(const Matrix<double, 1, Dynamic>& eta, double lambda, double a);

Rcpp::List estimateSequence(const MatrixXd& y, const Psi& startingVals, const VectorXd& lambdaList);
Rcpp::List estimateSequence2(const MatrixXd& y, const Psi& startingVals, const VectorXd& lambdaList,const VectorXd&);
Rcpp::List rbic(const MatrixXd&, const Psi&, const VectorXd&);
Psi mem(const MatrixXd&, const Psi&, double);
Matrix<double, Dynamic, Dynamic> wMatrix(const MatrixXd& y, const Psi&);
Psi mStep(const MatrixXd& , const Psi& , const MatrixXd& ,  const MatrixXd& , double );
MatrixXd admm(const MatrixXd& , const Psi&, const MatrixXd&, const MatrixXd&, double );
MatrixXd graphrep(const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx, double lambda);
Psi mem2(const MatrixXd&, const Psi&, double, const VectorXd&);
Matrix<double, Dynamic, Dynamic> wMatrix1(const MatrixXd& y, const Psi&, const VectorXd&);
Matrix<double, Dynamic, Dynamic> wMatrix2(const MatrixXd& y, const Psi&, const VectorXd&);
Psi mStep2(const MatrixXd& , const Psi& , const MatrixXd& ,  const MatrixXd& ,const MatrixXd& , double);
//MatrixXd admm2(const MatrixXd& , const Psi&, const MatrixXd&, const MatrixXd&,const MatrixXd&, double);
MatrixXd graphrep2(const MatrixXd&, const Psi&, const MatrixXd& , const MatrixXd&,const MatrixXd&, double , const VectorXd&);
//MatrixXd pgd(const MatrixXd&, const Psi&, const MatrixXd&, double lambda) ;
//bool uCheck(double u, const MatrixXd& y, const MatrixXd& oldEta, const MatrixXd& newEta, const MatrixXd& sigma, const VectorXd& wMtxSums);


// [[Rcpp::export(.myEm)]]
extern "C" SEXP myEm(SEXP argY, SEXP argGraphtype, SEXP argm, SEXP argTheta, SEXP argSigma, 
                          SEXP argPii, SEXP argArbSigma, SEXP argM, 
                          SEXP argIndex, SEXP argMaxAdmm, SEXP argMaxNR, SEXP argCk, SEXP argA, 
                          SEXP argPenalty, SEXP argLambdaVals, SEXP argEpsilon, 
                          SEXP argDelta, SEXP argMaxRep, 
                          SEXP argVerbose, SEXP argDynamic){
  MatrixXd theta  = Rcpp::as<MatrixXd> (argTheta);      // Size D x K.
  //MatrixXd Graph  = Rcpp::as<MatrixXd> (argGraph);      // Size K x K.
  MatrixXd y      = Rcpp::as<MatrixXd> (argY);          // Size n x N.
  MatrixXd sigma  = Rcpp::as<MatrixXd> (argSigma);      // Size N x N.
  VectorXd pii    = Rcpp::as<VectorXd> (argPii);        // Size 1 x K.
  
  Graphtype       = Rcpp::as<int> (argGraphtype);    
  m       = Rcpp::as<int> (argm);       //the graph to use in penalty
  arbSigma        = Rcpp::as<bool> (argArbSigma);       // Common unknown structure parameter check.
  ck              = Rcpp::as<double> (argCk);           // Parameter for penalty on the pi_k.
  epsilon         = Rcpp::as<double> (argEpsilon);      // Convergence criterion for EM algorithm.
  maxadmm         = Rcpp::as<int> (argMaxAdmm);
  maxNR           = Rcpp::as<int> (argMaxNR);
  //  maxPgd          = Rcpp::as<int> (argMaxPgd);          // Maximum number of repetetitions of the PGD algorithm.
  maxRep          = Rcpp::as<int> (argMaxRep);          // Maximum EM iterations.
  a               = Rcpp::as<double> (argA);            // Parameter for the SCAD or MCP penalty.  
  delta           = Rcpp::as<double> (argDelta);        // Convergence criterion for PGD algorithm.
  lambdaVals      = Rcpp::as<VectorXd> (argLambdaVals); // Values of lambda to be used.

  //  uBound          = Rcpp::as<double> (argUBound);       // Upper bound on the parameter u for the PGD algorithm. 
  verbose         = Rcpp::as<bool> (argVerbose);        // 
  dynamic         = Rcpp::as<bool> (argDynamic);   
  M               = Rcpp::as<int> (argM);               // Number of trials for multinomial mixtures. 
  modelIndex      = Rcpp::as<int> (argIndex);           // Model index -- see below.
  //  H               = Rcpp::as<double> (argH);            // Hartigan lower bound for location-scale normal mixtures.
  penalty         = Rcpp::as<int> (argPenalty);                                   // H = 0 for other models. 
  
  // Set important constants.
  //K = theta.cols();  // Number of mixture components. 
  D = theta.rows();  // Dimension of the parameter space.
  N = y.cols();      // Dimension of the sample space. 
  n = y.rows();      // Sample size.
  
  switch(Graphtype){
  case 1:
    Graphmat = &graphmnn;
    break;
    
  case 2:
    Graphmat = &graphmGSF;
    break;
    
  case 3:
    Graphmat = &graphmMST;
    break;
    
  }
  
  switch(penalty) {
  case 1: 
    //updateEta = &scadUpdate;
    updateLambda = &GetSCAD; 
    lambdaScale = sqrt(n); 
    break;
    
  case 2: 
    //updateEta = &mcpUpdate;
    updateLambda = &GetMCP; 
    lambdaScale = sqrt(n);
    break;
    
  case 3: 
    //updateEta = &adaptiveLassoUpdate; 
    updateLambda = &GetALasso;
    lambdaScale = n; //pow(n, 0.5);
    
    if (lambdaVals(0) != 0) {
      VectorXd temp = VectorXd::Zero(lambdaVals.size() + 1);
      temp(0) = 0;
      temp.tail(lambdaVals.size()) = lambdaVals;
      lambdaVals = temp;
    }
    
    break;
    
  case 4: //unweight
    //updateEtaLLA = &scadLLAUpdate; 
    updateLambda = &GetUnweight;
    lambdaScale = sqrt(n);
    break;
    
  case 5: 
    //updateEtaLLA = &mcpLLAUpdate; 
    lambdaScale = sqrt(n);
    break;
    
  default: 
    throw "Unknown penalty.";
  }
  
  // Initialize the set of parameters.
  Psi psi;
  psi.pii   = pii;
  psi.sigma = sigma;
  
  alStart = 0;
  
  switch(modelIndex) {
  
  // User selected a multivariate normal mixture in location. 
  // The covariance matrix is assumed to be common, but if hasFixedMatrix is false, 
  // it is unknown and estimated by the EM algorithm.
  case 1:   updateTheta = &normaltheta;
    updateTheta0 = &normaltheta0;
    gradB       = &gradBNormalLoc;
    b           = &bNormalLoc;
    T           = &tNormalLoc;
    density     = &densityNormalLoc;
    invTransf   = &invTransfNormalLoc;
    constrCheck = &constrCheckNormalLoc;            
    
    psi.theta   = transfNormalLoc(theta, sigma);
    //Rcpp::Rcout << "transformed theta" << psi.theta << "\n";
    
    for (int i = 0; i < n; i++) {
      yOuterProd.push_back((y.row(i).transpose())*(y.row(i)));
    }
    
    break;
    // User selected a multinomial mixture.
  case 3: updateTheta = &multinomialtheta;
    updateTheta0 = &multinomialtheta0;
    gradB       = &gradBMultinomial;
    b           = &bMultinomial; 
    T           = &tMultinomial;
    density     = &densityMultinomial;
    invTransf   = &invTransfMultinomial;
    constrCheck = &constrCheckMultinomial;
    
    psi.theta   = transfMultinomial(theta);
    
    break;
    
    // User selected a Poisson mixture.
  case 5: gradB       = &gradBPoisson; 
    b           = &bPoisson;
    T           = &tPoisson;
    density     = &densityPoisson;
    invTransf   = &invTransfPoisson;
    constrCheck = &constrCheckPoisson;
    
    psi.theta   = transfPoisson(theta);

    break;
    
    // User selected a mixture of exponential distributions.
  case 6: gradB       = &gradBExponential; 
    b           = &bExponential;
    T           = &tExponential;
    density     = &densityExponential;
    invTransf   = &invTransfExponential;
    constrCheck = &constrCheckExponential;
    
    psi.theta   = transfExponential(theta);
    
    break;
  }
  
  transformedData = (*T)(y).transpose();
  Rcpp::List out = estimateSequence(y,psi, lambdaVals);
  
  return Rcpp::wrap(out); 
}

// The Modified EM (MEM) Algorithm wrapper.
Psi mem(const MatrixXd& y, const Psi& psi, double lambda) {
  //Rcpp::Rcout << "lambda in mem" << lambda << "\n";
  MatrixXd wMtx, th, graph;
  Psi oldEstimate, newEstimate = psi;
  int counter = 0;
  
  do {
    graph = (*Graphmat)(newEstimate.theta, m);
    //Rcpp::Rcout << "graph" << graph << "\n";
    oldEstimate = newEstimate;
    newEstimate = mStep(y, oldEstimate, graph, wMatrix(y, oldEstimate), lambda);
  } while (counter++ < maxRep && oldEstimate.distance(newEstimate) >= epsilon);
  
  
  if (verbose) {
    //Rcpp::Rcout << "Total MEM iterations: " << counter << ".\n";
  }
  
  return newEstimate;
}

// Creates a matrix of w_ik values.
MatrixXd wMatrix(const MatrixXd& y, const Psi& psi) {
  int K = psi.theta.cols();  // Number of mixture components. 
  MatrixXd result(n, K);
  double acc;

  for (int i = 0; i < n; i++) {
    acc = 0.0;

    for (int k = 0; k < K; k++) {
      result(i, k) = psi.pii(k) * (*density)(y.row(i), psi.theta.col(k), psi.sigma);
      acc += result(i, k);
    }

    result.row(i) /= acc;
  }
  //Rcpp::Rcout << "weights" << result <<"\n";
  return result;
}

// M-Step of the Modified EM Algorithm.
Psi mStep(const MatrixXd& y, const Psi& psi, const MatrixXd& graph,  const MatrixXd& wMtx, double lambda) {
  int K = psi.theta.cols();  // Number of mixture components. 
  Psi result;
  int i, k;
  result.pii = VectorXd::Zero(K);
  
  if(!arbSigma) {
    double acc;
    
    for (k = 0; k < K; k++) {
      acc = 0.0;
      
      for (i = 0; i < n; i++) {
        acc += wMtx(i, k);
      }
      
      result.pii(k) = (acc + ck) / (n + K * ck);
    }
    
    result.sigma = psi.sigma;
    
    // The following is currently hard-coded for updating 
    // the common structure parameter ("sigma")
    // in multivariate location Normal mixtures.
    // The notation is that of McLachlan and Peel 
    // (Finite Mixture Models, Chapter 3). 
  } else {   
    double Tk1Sum;      
    MatrixXd Tk2Sum(N, 1);
    MatrixXd Tk3Sum(N, N);
    
    result.sigma = MatrixXd::Zero(N, N);
    
    for (k = 0; k < K; k++) {
      Tk1Sum = wMtx.col(k).sum();
      Tk2Sum = MatrixXd::Zero(N, 1); //(wMtx.col(k).transpose() * y).transpose();
      Tk3Sum = MatrixXd::Zero(N, N);
      
      result.pii(k) = (Tk1Sum + ck) / (n + K * ck);
      
      for (i = 0; i < n; i++) {
        Tk2Sum += wMtx(i, k) * y.row(i).transpose();
        Tk3Sum += wMtx(i, k) * /*yOuterProd[i]; */ y.row(i).transpose() * y.row(i);
      }
      //Rcpp::Rcout << "Tk2sum" << Tk2Sum <<"\n";
      //Rcpp::Rcout << "Tk3sum" << Tk3Sum <<"\n";
      
      result.sigma += Tk3Sum - (1.0 / Tk1Sum) * Tk2Sum * Tk2Sum.transpose();
    }
    
    result.sigma /= n;
  }
  
  //Rcpp::Rcout << "sigma" << result.sigma <<"\n";
  
  result.theta = psi.theta;
  
  result.theta = admm(y, result, graph, wMtx, lambda);
  
  return result;
}

//Revised ADMM algorithm
MatrixXd admm(const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx, double lambda){
  int K = psi.theta.cols();  // Number of mixture components. 
  int D = psi.theta.rows();  // Dimension of the parameter space.
  int k,j, counter = 0;
  MatrixXd newTheta = MatrixXd::Zero(D, K), 
    oldTheta(D, K), 
    phi = MatrixXd::Zero(K,K), 
    Eta = MatrixXd::Zero(D,K*K), 
    oldU = MatrixXd::Zero(D,K*K),
    newU = MatrixXd::Zero(D,K*K);
  Psi psitemp;
  psitemp = psi;
  
  //Initialize theta
  
  oldTheta = psi.theta;
  for(k = 0; k < K; k++) {
    for(j = 0; j < K; j++){
      if (graph(k,j)==1){
      Eta.col(k+K*j) = psi.theta.col(k);
      }}}
  
  if (lambda > 0 && (penalty == 1 || penalty == 2)){
  do {
    oldTheta = newTheta ; 
    psitemp.theta = oldTheta;
    newTheta = (*updateTheta)(y, psitemp, graph, wMtx, Eta, oldU);
    
    //update Eta
    for(k = 0; k < K; k++) {
      for(j = 0; j < K; j++){
        if (graph(k,j)==1){
          phi(k,j) = etamax(newTheta.col(k)+oldU.col(k+K*j)-newTheta.col(j)-oldU.col(j+K*k), (*updateLambda)(newTheta.col(k)-newTheta.col(j), lambda, a));
          Eta.col(k+K*j) = phi(k,j)*(newTheta.col(k)+oldU.col(k+K*j))+(1-phi(k,j))*(newTheta.col(j)+oldU.col(j+K*k));
        }}}
    
    //update U
    for(k = 0; k < K; k++) {
      for(j = 0; j < K; j++){
        if (graph(k,j)==1){
          newU.col(k+K*j)=oldU.col(k+K*j)+newTheta.col(k)-Eta.col(k+K*j);
        }}}

    oldU=newU;
  } while (counter++ < maxadmm && (oldTheta - newTheta).norm() > delta);
  } else if (lambda > 0 && penalty == 3){
    do {
      oldTheta = newTheta ; 
      psitemp.theta = oldTheta;
      newTheta = (*updateTheta)(y, psitemp, graph, wMtx, Eta, oldU);
      
      //update Eta
      for(k = 0; k < K; k++) {
        for(j = 0; j < K; j++){
          if (graph(k,j)==1){
            phi(k,j) = etamax(newTheta.col(k)+oldU.col(k+K*j)-newTheta.col(j)-oldU.col(j+K*k), (*updateLambda)(newTheta.col(k)-newTheta.col(j), lambda, 1/adaptiveLassoWeights(k,j)));
            Eta.col(k+K*j) = phi(k,j)*(newTheta.col(k)+oldU.col(k+K*j))+(1-phi(k,j))*(newTheta.col(j)+oldU.col(j+K*k));
          }}}
      
      //update U
      for(k = 0; k < K; k++) {
        for(j = 0; j < K; j++){
          if (graph(k,j)==1){
            newU.col(k+K*j)=oldU.col(k+K*j)+newTheta.col(k)-Eta.col(k+K*j);
          }}}
      
      oldU=newU;
    } while (counter++ < maxadmm && (oldTheta - newTheta).norm() > delta);
  }else { //lambda = 0
    newTheta = (*updateTheta0)(y, psitemp, graph, wMtx, Eta, oldU);
  }
  return newTheta;
}

MatrixXd graphrep(const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx, double lambda){
  int K = psi.theta.cols();  // Number of mixture components. 
  int D = psi.theta.rows();  // Dimension of the parameter space.
  int k,j, counter = 0;
  MatrixXd newTheta = MatrixXd::Zero(D, K), 
    oldTheta(D, K), 
    phi = MatrixXd::Zero(K,K), 
    Eta = MatrixXd::Zero(D,K*K), 
    oldU = MatrixXd::Zero(D,K*K),
    newU = MatrixXd::Zero(D,K*K);
  
  //Initialize theta
  
  oldTheta = psi.theta;
  for(k = 0; k < K; k++) {
    for(j = 0; j < K; j++){
      if (graph(k,j)==1){
      Eta.col(k+K*j) = psi.theta.col(k);
      }}}
  
  do {
    oldTheta = newTheta ; 
    newTheta = (*updateTheta)(y, psi, graph, wMtx, Eta, oldU);
    //Rcpp::Rcout << "Theta" << newTheta.row(1) << ".\n";
    
    //update Eta
    for(k = 0; k < K; k++) {
      for(j = 0; j < K; j++){
        if (graph(k,j)==1){
          phi(k,j) = etamax(newTheta.col(k)+oldU.col(k+K*j)-newTheta.col(j)-oldU.col(j+K*k), (*updateLambda)(newTheta.col(k)-newTheta.col(j), lambda, a));
          Eta.col(k+K*j) = phi(k,j)*(newTheta.col(k)+oldU.col(k+K*j))+(1-phi(k,j))*(newTheta.col(j)+oldU.col(j+K*k));
        }}}
    //Rcpp::Rcout << "Eta" << Eta.row(1) << ".\n";
    
    
    //update U
    for(k = 0; k < K; k++) {
      for(j = 0; j < K; j++){
        if (graph(k,j)==1){
          newU.col(k+K*j)=oldU.col(k+K*j)+newTheta.col(k)-Eta.col(k+K*j);
        }}}
    //Rcpp::Rcout << "oldU" << oldU.row(1) << ".\n";
    //Rcpp::Rcout << "newU" << newU.row(1) << ".\n";
    oldU=newU;
    //Rcpp::Rcout << "thetadiff" << (oldTheta - newTheta).norm() << ".\n";
  } while (counter++ < maxadmm && (oldTheta - newTheta).norm() > delta);
  
  for(k = 0; k < K; k++) {
    for(j = 0; j < K; j++){
      if (phi(k,j) == 0.5){
        phi(k,j) = 1;
      }else {
        phi(k,j) = 0;
      }
      }}
  return phi;
}

// Log-likelihood function. 
double fullLogLikFunction(const MatrixXd& y, const MatrixXd& theta, const VectorXd& pii, const MatrixXd& sigma){
  int K = theta.cols();  // Number of mixture components. 
  double temp, loglikSum = 0.0;

  for (int i = 0; i < n; i++) {
    temp = 0.0;

    for (int k = 0; k < K; k++) {
      temp += pii(k) * (*density)(y.row(i), theta.col(k), sigma);
    }

    loglikSum += log(temp);
  }

  return loglikSum;
}

double logLikFunction(const MatrixXd& y, const Psi& psi){
  return fullLogLikFunction(y, psi.theta, psi.pii, psi.sigma);
}

Rcpp::List estimateSequence(const MatrixXd& y, const Psi& startingVals, const VectorXd& lambdaList){
  Psi newpsi, psi = startingVals;
  int i, k, numComponents;
  
  
  Rcpp::List estimates;
  Rcpp::NumericVector rbicVals, orders, loglikVals;
  
  
  for (i = 0; i < lambdaList.size(); i++) {
    int K = psi.theta.cols();
    int D = psi.theta.rows();
    MatrixXd transfTheta(D,K),finalgraph(D,K);
    Rcpp::List thisEstimate;
    Rcpp::NumericVector pii;
    Rcpp::CharacterVector names(K);

    if(penalty == 3) {
      adaptiveLassoWeights = MatrixXd::Zero(K,K);
      psi = mem(y, psi, alStart);
      adaptiveLassoWeights = getDistanceMatrix(psi.theta);
    }
    
    if (verbose) 
      //Rcpp::Rcout << "Lambda " << lambdaList(i) << ".\n";
    
    if(K >=2){
    try {
      //if (verbose) 
        //Rcpp::Rcout << "Estimate: \n" << invTransf(psi.theta, psi.sigma) << "\n\n";
     
     psi = mem(y, psi, lambdaScale * lambdaList(i));
   
    } catch (const char* error) {
      throw error;
    }
    
    finalgraph = graphrep(y,psi,(*Graphmat)(psi.theta, m),wMatrix(y,psi), lambdaScale * lambdaList(i));
    numComponents = countClusters(finalgraph);
    newpsi = equalizeThetaInClusters(psi, finalgraph);
    //psi = newpsi;

    if (dynamic) {
      newpsi= mergeComponents(psi, finalgraph);
      psi.pii = newpsi.pii;
      psi.theta = newpsi.theta;
    }}
    
    pii = Rcpp::wrap(psi.pii);

    transfTheta = invTransf(newpsi.theta, newpsi.sigma);
    
    
    thisEstimate["lambda"] = lambdaList(i); 
    thisEstimate["graph"]  = finalgraph;
    thisEstimate["order"]  = numComponents;
    thisEstimate["pii"]    = pii;

    switch (modelIndex) {
      case 1: thisEstimate["mu"]    = transfTheta;
              thisEstimate["sigma"] = psi.sigma;
              break;
        
      case 2: thisEstimate["mu"]    = transfTheta.row(0);
              thisEstimate["sigma"] = transfTheta.row(1);
              break;

      default: thisEstimate["theta"] = transfTheta;
    }
    
    estimates.push_back(thisEstimate);
   
  }

  return estimates;
}


///EM2

// [[Rcpp::export(.myEm2)]]
SEXP myEm2(SEXP argY, SEXP argnu, SEXP argGraphtype, SEXP argm, SEXP argTheta, SEXP argSigma, 
           SEXP argPii, SEXP argArbSigma, SEXP argM, 
           SEXP argIndex, SEXP argMaxAdmm, SEXP argMaxNR, SEXP argCk, SEXP argA, 
           SEXP argPenalty, SEXP argLambdaVals, SEXP argEpsilon, 
           SEXP argDelta, SEXP argMaxRep, 
           SEXP argVerbose, SEXP argDynamic){
  MatrixXd theta  = Rcpp::as<MatrixXd> (argTheta);      // Size D x K.
  //MatrixXd Graph  = Rcpp::as<MatrixXd> (argGraph);      // Size K x K.
  MatrixXd y      = Rcpp::as<MatrixXd> (argY);          // Size n x N.
  VectorXd nu     = Rcpp::as<VectorXd> (argnu);    // Size 1 x K
  MatrixXd sigma  = Rcpp::as<MatrixXd> (argSigma);      // Size N x N.
  VectorXd pii    = Rcpp::as<VectorXd> (argPii);        // Size 1 x K.
  
  Graphtype       = Rcpp::as<int> (argGraphtype);    
  m       = Rcpp::as<int> (argm);       //the graph to use in penalty
  arbSigma        = Rcpp::as<bool> (argArbSigma);       // Common unknown structure parameter check.
  ck              = Rcpp::as<double> (argCk);           // Parameter for penalty on the pi_k.
  epsilon         = Rcpp::as<double> (argEpsilon);      // Convergence criterion for EM algorithm.
  maxadmm         = Rcpp::as<int> (argMaxAdmm);
  maxNR           = Rcpp::as<int> (argMaxNR);
  //  maxPgd          = Rcpp::as<int> (argMaxPgd);          // Maximum number of repetitions of the PGD algorithm.
  maxRep          = Rcpp::as<int> (argMaxRep);          // Maximum EM iterations.
  a               = Rcpp::as<double> (argA);            // Parameter for the SCAD or MCP penalty.  
  delta           = Rcpp::as<double> (argDelta);        // Convergence criterion for PGD algorithm.
  lambdaVals      = Rcpp::as<VectorXd> (argLambdaVals); // Values of lambda to be used.
  
  //  uBound          = Rcpp::as<double> (argUBound);       // Upper bound on the parameter u for the PGD algorithm. 
  verbose         = Rcpp::as<bool> (argVerbose);        // 
  dynamic         = Rcpp::as<bool> (argDynamic);   
  M               = Rcpp::as<int> (argM);               // Number of trials for multinomial mixtures. 
  modelIndex      = Rcpp::as<int> (argIndex);           // Model index -- see below.
  //  H               = Rcpp::as<double> (argH);            // Hartigan lower bound for location-scale normal mixtures.
  penalty         = Rcpp::as<int> (argPenalty);                                   // H = 0 for other models. 
  
  // Set important constants.
  //K = theta.cols();  // Number of mixture components. 
  D = theta.rows();  // Dimension of the parameter space.
  N = y.cols();      // Dimension of the sample space. 
  n = y.rows();      // Sample size.
  
  switch(Graphtype){
  case 1:
    Graphmat = &graphmnn;
    break;
    
  case 2:
    Graphmat = &graphmGSF;
    break;
    
  case 3:
    Graphmat = &graphmMST;
    break;
    
  }
  
  switch(penalty) {
  case 1: 
    //updateEta = &scadUpdate;
    updateLambda = &GetSCAD; 
    lambdaScale = sqrt(n); 
    break;
    
  case 2: 
    //updateEta = &mcpUpdate;
    updateLambda = &GetMCP; 
    lambdaScale = sqrt(n);
    break;
    
  case 3: 
    //updateEta = &adaptiveLassoUpdate; 
    updateLambda = &GetALasso;
    lambdaScale = n; //pow(n, 0.5);
    
    if (lambdaVals(0) != 0) {
      VectorXd temp = VectorXd::Zero(lambdaVals.size() + 1);
      temp(0) = 0;
      temp.tail(lambdaVals.size()) = lambdaVals;
      lambdaVals = temp;
    }
    
    break;
    
  case 4: //unweight
    //updateEtaLLA = &scadLLAUpdate; 
    updateLambda = &GetUnweight;
    lambdaScale = sqrt(n);
    break;
    
  case 5: 
    //updateEtaLLA = &mcpLLAUpdate; 
    lambdaScale = sqrt(n);
    break;
    
  default: 
    throw "Unknown penalty.";
  }
  
  // Initialize the set of parameters.
  Psi psi;
  psi.pii   = pii;
  psi.sigma = sigma;
  
  alStart = 0;
  
  // User selected a T mixture.
  updateTheta = &normaltheta;
  updateTheta0 = &normaltheta0;
  gradB       = &gradBT;
  b           = &bT; 
  T           = &tT;
  density2     = &densityT;
  invTransf   = &invTransfT;
  constrCheck = &constrCheckT;
  
  psi.theta   = transfT(theta, sigma);
  
  
  transformedData = (*T)(y).transpose();
  Rcpp::List out = estimateSequence2(y,psi, lambdaVals,nu);
  
  return Rcpp::wrap(out); 
}

// The Modified EM (MEM) Algorithm wrapper.
Psi mem2(const MatrixXd& y, const Psi& psi, double lambda, const VectorXd& nu) {
  //Rcpp::Rcout << "lambda in mem" << lambda << "\n";
  MatrixXd wMtx, th, graph;
  Psi oldEstimate, newEstimate = psi;
  int counter = 0;
  
  do {
    graph = (*Graphmat)(newEstimate.theta, m);
    //Rcpp::Rcout << "graph" << graph << "\n";
    oldEstimate = newEstimate;
    newEstimate = mStep2(y, oldEstimate, graph, wMatrix1(y, oldEstimate, nu), wMatrix2(y, oldEstimate, nu), lambda);
    //Rcpp::Rcout << "distance" << oldEstimate.distance(newEstimate) << "\n";
  } while (counter++ < maxRep && oldEstimate.distance(newEstimate) >= epsilon);
  
  
  if (verbose) {
    //Rcpp::Rcout << "Total MEM iterations: " << counter << ".\n";
  }
  
  return newEstimate;
}

// Creates a matrix of w_ik values.
MatrixXd wMatrix1(const MatrixXd& y, const Psi& psi, const VectorXd& nu) {
  int K = psi.theta.cols();  // Number of mixture components. 
  MatrixXd result(n, K);
  double acc;
  
  for (int i = 0; i < n; i++) {
    acc = 0.0;
    
    for (int k = 0; k < K; k++) {
      result(i, k) = psi.pii(k) * (*density2)(y.row(i), psi.theta.col(k), psi.sigma, nu(k));
      acc += result(i, k);
    }
    
    result.row(i) /= acc;
  }
  //Rcpp::Rcout << "weights" << result.row(1) <<"\n";
  return result;
}

// Creates a matrix of w2_ik values.
MatrixXd wMatrix2(const MatrixXd& y, const Psi& psi, const VectorXd& nu) {
  int K = psi.theta.cols();  // Number of mixture components. 
  MatrixXd result(n, K);
  
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < K; k++) {
      result(i, k) = (nu(k)+D)/(nu(k)- 2* y.row(i) *  psi.theta.col(k) + psi.theta.col(k).transpose() * psi.sigma * psi.theta.col(k) + y.row(i) * psi.sigma.inverse() *y.row(i).transpose());
    } }
  //Rcpp::Rcout << "weights" << result.row(1) <<"\n";
  return result;
}

// M-Step of the Modified EM Algorithm.
Psi mStep2(const MatrixXd& y, const Psi& psi, const MatrixXd& graph,  const MatrixXd& wMtx1, const MatrixXd& wMtx2, double lambda) {
  int K = psi.theta.cols();  // Number of mixture components. 
  Psi result;
  int i, k;
  result.pii = VectorXd::Zero(K);
  
  MatrixXd wMtx(n,K);
  for(k = 0; k < K; k++) {
    for(i=0 ; i < n; i++){
      wMtx(i,k) = wMtx1(i,k) * wMtx2(i,k);
    }}
  
  if(!arbSigma) {
    double acc;
    
    for (k = 0; k < K; k++) {
      acc = 0.0;
      
      for (i = 0; i < n; i++) {
        acc += wMtx1(i, k);
      }
      
      result.pii(k) = (acc + ck) / (n + K * ck);
    }
    
    result.sigma = psi.sigma;
    
    // The following is currently hard-coded for updating 
    // the common structure parameter ("sigma")
    // in multivariate location Normal mixtures.
    // The notation is that of McLachlan and Peel 
    // (Finite Mixture Models, Chapter 3). 
  } else {   
    double Tk1Sum;      
    MatrixXd Tk2Sum(N, 1);
    MatrixXd Tk3Sum(N, N);
    
    result.sigma = MatrixXd::Zero(N, N);
    
    for (k = 0; k < K; k++) {
      Tk1Sum = wMtx.col(k).sum();
      Tk2Sum = MatrixXd::Zero(N, 1); //(wMtx.col(k).transpose() * y).transpose();
      Tk3Sum = MatrixXd::Zero(N, N);
      
      result.pii(k) = (Tk1Sum + ck) / (n + K * ck);
      
      for (i = 0; i < n; i++) {
        Tk2Sum += wMtx(i, k) * y.row(i).transpose();
        Tk3Sum += wMtx(i, k) * /*yOuterProd[i]; */ y.row(i).transpose() * y.row(i);
      }
      //Rcpp::Rcout << "Tk2sum" << Tk2Sum <<"\n";
      //Rcpp::Rcout << "Tk3sum" << Tk3Sum <<"\n";
      
      result.sigma += Tk3Sum - (1.0 / Tk1Sum) * Tk2Sum * Tk2Sum.transpose();
    }
    
    result.sigma /= n;
  }
  
  //Rcpp::Rcout << "sigma" << result.sigma <<"\n";
  
  result.theta = psi.theta;
  
  result.theta = admm(y, result, graph, wMtx, lambda);
  
  return result;
}

//Revised ADMM algorithm
MatrixXd admm2(const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx1,const MatrixXd& wMtx2, double lambda){
  int K = psi.theta.cols();  // Number of mixture components. 
  int D = psi.theta.rows();  // Dimension of the parameter space.
  int i,k,j, counter = 0;
  MatrixXd newTheta = MatrixXd::Zero(D, K), 
    oldTheta(D, K), 
    phi = MatrixXd::Zero(K,K), 
    Eta = MatrixXd::Zero(D,K*K), 
    oldU = MatrixXd::Zero(D,K*K),
    newU = MatrixXd::Zero(D,K*K);
  Psi psitemp;
  psitemp = psi;
  
  MatrixXd wMtx(n,K);
  for(k = 0; k < K; k++) {
    for(i=0 ; i < n; i++){
      wMtx(i,k) = wMtx1(i,k) * wMtx2(i,k);
    }}
  
  //Initialize theta
  
  oldTheta = psi.theta;
  for(k = 0; k < K; k++) {
    for(j = 0; j < K; j++){
      if (graph(k,j)==1){
        Eta.col(k+K*j) = psi.theta.col(k);
      }}}
  
  if (lambda > 0 && (penalty == 1 || penalty == 2)){
    do {
      oldTheta = newTheta ; 
      psitemp.theta = oldTheta;
      newTheta = (*updateTheta)(y, psitemp, graph, wMtx, Eta, oldU);
      
      //update Eta
      for(k = 0; k < K; k++) {
        for(j = 0; j < K; j++){
          if (graph(k,j)==1){
            phi(k,j) = etamax(newTheta.col(k)+oldU.col(k+K*j)-newTheta.col(j)-oldU.col(j+K*k), (*updateLambda)(newTheta.col(k)-newTheta.col(j), lambda, a));
            Eta.col(k+K*j) = phi(k,j)*(newTheta.col(k)+oldU.col(k+K*j))+(1-phi(k,j))*(newTheta.col(j)+oldU.col(j+K*k));
          }}}
      
      //update U
      for(k = 0; k < K; k++) {
        for(j = 0; j < K; j++){
          if (graph(k,j)==1){
            newU.col(k+K*j)=oldU.col(k+K*j)+newTheta.col(k)-Eta.col(k+K*j);
          }}}
      
      oldU=newU;
    } while (counter++ < maxadmm && (oldTheta - newTheta).norm() > delta);
  } else if (lambda > 0 && penalty == 3){
    do {
      oldTheta = newTheta ; 
      psitemp.theta = oldTheta;
      newTheta = (*updateTheta)(y, psitemp, graph, wMtx, Eta, oldU);
      
      //update Eta
      for(k = 0; k < K; k++) {
        for(j = 0; j < K; j++){
          if (graph(k,j)==1){
            phi(k,j) = etamax(newTheta.col(k)+oldU.col(k+K*j)-newTheta.col(j)-oldU.col(j+K*k), (*updateLambda)(newTheta.col(k)-newTheta.col(j), lambda, 1/adaptiveLassoWeights(k,j)));
            Eta.col(k+K*j) = phi(k,j)*(newTheta.col(k)+oldU.col(k+K*j))+(1-phi(k,j))*(newTheta.col(j)+oldU.col(j+K*k));
          }}}
      
      //update U
      for(k = 0; k < K; k++) {
        for(j = 0; j < K; j++){
          if (graph(k,j)==1){
            newU.col(k+K*j)=oldU.col(k+K*j)+newTheta.col(k)-Eta.col(k+K*j);
          }}}
      
      oldU=newU;
    } while (counter++ < maxadmm && (oldTheta - newTheta).norm() > delta);
  }else { //lambda = 0
    newTheta = (*updateTheta0)(y, psitemp, graph, wMtx, Eta, oldU);
  }
  return newTheta;
}

MatrixXd graphrep2(const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx1,const MatrixXd& wMtx2, double lambda){
  int K = psi.theta.cols();  // Number of mixture components. 
  int D = psi.theta.rows();  // Dimension of the parameter space.
  int i,k,j, counter = 0;
  MatrixXd newTheta = MatrixXd::Zero(D, K), 
    oldTheta(D, K), 
    phi = MatrixXd::Zero(K,K), 
    Eta = MatrixXd::Zero(D,K*K), 
    oldU = MatrixXd::Zero(D,K*K),
    newU = MatrixXd::Zero(D,K*K);
  
  MatrixXd wMtx(n,K);
  for(k = 0; k < K; k++) {
    for(i=0 ; i < n; i++){
      wMtx(i,k) = wMtx1(i,k) * wMtx2(i,k);
    }}
  
  //Initialize theta
  
  oldTheta = psi.theta;
  for(k = 0; k < K; k++) {
    for(j = 0; j < K; j++){
      if (graph(k,j)==1){
        Eta.col(k+K*j) = psi.theta.col(k);
      }}}
  
  do {
    oldTheta = newTheta ; 
    newTheta = (*updateTheta)(y, psi, graph, wMtx, Eta, oldU);
    //Rcpp::Rcout << "Theta" << newTheta.row(1) << ".\n";
    
    //update Eta
    for(k = 0; k < K; k++) {
      for(j = 0; j < K; j++){
        if (graph(k,j)==1){
          phi(k,j) = etamax(newTheta.col(k)+oldU.col(k+K*j)-newTheta.col(j)-oldU.col(j+K*k), (*updateLambda)(newTheta.col(k)-newTheta.col(j), lambda, a));
          Eta.col(k+K*j) = phi(k,j)*(newTheta.col(k)+oldU.col(k+K*j))+(1-phi(k,j))*(newTheta.col(j)+oldU.col(j+K*k));
        }}}
    //Rcpp::Rcout << "Eta" << Eta.row(1) << ".\n";
    
    
    //update U
    for(k = 0; k < K; k++) {
      for(j = 0; j < K; j++){
        if (graph(k,j)==1){
          newU.col(k+K*j)=oldU.col(k+K*j)+newTheta.col(k)-Eta.col(k+K*j);
        }}}
    //Rcpp::Rcout << "oldU" << oldU.row(1) << ".\n";
    //Rcpp::Rcout << "newU" << newU.row(1) << ".\n";
    oldU=newU;
    //Rcpp::Rcout << "thetadiff" << (oldTheta - newTheta).norm() << ".\n";
  } while (counter++ < maxadmm && (oldTheta - newTheta).norm() > delta);
  
  for(k = 0; k < K; k++) {
    for(j = 0; j < K; j++){
      if (phi(k,j) == 0.5){
        phi(k,j) = 1;
      }else {
        phi(k,j) = 0;
      }
    }}
  return phi;
}

// Log-likelihood function. 
double fullLogLikFunction2(const MatrixXd& y, const MatrixXd& theta, const VectorXd& pii, const MatrixXd& sigma, const VectorXd& nu){
  int K = theta.cols();  // Number of mixture components. 
  double temp, loglikSum = 0.0;
  
  for (int i = 0; i < n; i++) {
    temp = 0.0;
    
    for (int k = 0; k < K; k++) {
      temp += pii(k) * (*density2)(y.row(i), theta.col(k), sigma, nu(k));
    }
    
    loglikSum += log(temp);
  }
  
  return loglikSum;
}

double logLikFunction2(const MatrixXd& y, const Psi& psi, const VectorXd& nu){
  return fullLogLikFunction2(y, psi.theta, psi.pii, psi.sigma, nu);
}

Rcpp::List estimateSequence2(const MatrixXd& y, const Psi& startingVals, const VectorXd& lambdaList, const VectorXd& nu){
  Psi newpsi, psi = startingVals;
  int i, k, numComponents;
  
  
  Rcpp::List estimates;
  Rcpp::NumericVector rbicVals, orders, loglikVals;
  
  
  for (i = 0; i < lambdaList.size(); i++) {
    int K = psi.theta.cols();
    int D = psi.theta.rows();
    MatrixXd transfTheta(D,K),finalgraph(D,K);
    Rcpp::List thisEstimate;
    Rcpp::List th(K);
    Rcpp::NumericVector pii;
    Rcpp::CharacterVector names(K);
    
    if(penalty == 3) {
      adaptiveLassoWeights = MatrixXd::Zero(K,K);
      psi = mem2(y, psi, alStart,nu);
      adaptiveLassoWeights = getDistanceMatrix(psi.theta);
    }
    
    if (verbose) 
      //Rcpp::Rcout << "Lambda " << lambdaList(i) << ".\n";
    if(K >=2){
      try {
        //if (verbose) 
        //Rcpp::Rcout << "Estimate: \n" << invTransf(psi.theta, psi.sigma) << "\n\n";
        
        psi = mem2(y, psi, lambdaScale * lambdaList(i),nu);
        
      } catch (const char* error) {
        throw error;
      }
      
      finalgraph = graphrep2(y,psi,(*Graphmat)(psi.theta, m),wMatrix1(y,psi,nu),wMatrix2(y,psi,nu), lambdaScale * lambdaList(i));
      numComponents = countClusters(finalgraph);
      newpsi = equalizeThetaInClusters(psi, finalgraph);
      //psi = newpsi;
    }
    
    if (K >= 2 && dynamic) {
      newpsi= mergeComponents(psi, finalgraph);
      psi = newpsi;
    }
    
    pii = Rcpp::wrap(psi.pii);
    
    transfTheta = invTransf(newpsi.theta, newpsi.sigma);
    
    if (i == 0) {
      thisEstimate["ck"] = ck;
    }
    
    thisEstimate["lambda"] = lambdaList(i); 
    thisEstimate["graph"]  = finalgraph;
    thisEstimate["order"]  = numComponents;
    thisEstimate["pii"]    = pii;
    
    switch (modelIndex) {
    case 1: thisEstimate["mu"]    = transfTheta;
      thisEstimate["sigma"] = psi.sigma;
      break;

      
    default: thisEstimate["theta"] = transfTheta;
    }
    
    estimates.push_back(thisEstimate);
    
  }
  
  return estimates;
}

