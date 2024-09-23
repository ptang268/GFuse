#include <Rcpp.h>
#include <RcppEigen.h>
#include <iostream>
#include <float.h>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <sstream>

using namespace Eigen;

extern int D, N, n, M, m, graphtype, maxadmm, maxPgd, maxNR, maxRep, lambdaIter, modelIndex, penalty;
extern double epsilon, u, ck, tau1, a, delta, lambdaScale, alStart;
extern bool arbSigma, verbose;

typedef struct PsiStruct {
  VectorXd pii;
  MatrixXd theta;
  MatrixXd sigma;

  PsiStruct() {
    sigma = MatrixXd::Identity(N, N);
  }

  PsiStruct operator - (const PsiStruct psi) const {
    PsiStruct result;
    result.pii = pii - psi.pii;
    result.theta = theta - psi.theta;
    result.sigma = sigma - psi.sigma;

    return result;
  }

  double distance (PsiStruct psi) {
    return sqrt( (pii - psi.pii).squaredNorm() + (theta - psi.theta).squaredNorm() + (sigma - psi.sigma).squaredNorm() );
  }
} Psi;


/*** Functions for computing the permutation. ***/
bool isOrdered (int, int);
double thetaDist (const VectorXd&, const VectorXd&);
MatrixXd getDistanceMatrix (const MatrixXd&);
bool find (std::vector<int>, int);
MatrixXd graphmnn(const MatrixXd& theta, int m);
MatrixXd graphgsf(const MatrixXd& theta);
std::pair<int, int> findMaxDistanceIndices(const MatrixXd& distances);
int findNearestNeighbor(const VectorXi& visited, const MatrixXd& distances, int current);
std::pair<int, int> findLongestEdge(const MatrixXd& distances, const MatrixXd& graph);
MatrixXd graphmGSF(const Eigen::MatrixXd& theta,int m);
MatrixXd findMinimumSpanningTree(const Eigen::MatrixXd& distances);
MatrixXd graphmMST(const Eigen::MatrixXd& theta, int m);
void alpha(const MatrixXd& theta, std::vector<int>& perm);
MatrixXd reorderTheta(const MatrixXd& theta);
double fullLogLikFunction(const MatrixXd& y, const MatrixXd& theta, const VectorXd& pii, const MatrixXd& sigma);
double logLikFunction(const MatrixXd& y, const Psi& psi);
double fullLogLikFunction2(const MatrixXd& y, const MatrixXd& theta, const VectorXd& pii, const MatrixXd& sigma, const VectorXd&);
double logLikFunction2(const MatrixXd& y, const Psi& psi, const VectorXd&);
int countClusters(MatrixXd& graph);
void dfs(int node, const MatrixXd& finalgraph, std::vector<bool>& visited, std::vector<int>& component);
Psi mergeComponents(const Psi& psi, const MatrixXd& finalgraph);
void dfsEqualize(int node, const MatrixXd& finalgraph, std::vector<bool>& visited, std::vector<int>& component);
Psi equalizeThetaInClusters(const Psi& psi, const MatrixXd& finalgraph);

MatrixXd normaltheta (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx, const MatrixXd& Eta, const MatrixXd& U);
MatrixXd normaltheta0 (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx, const MatrixXd& Eta, const MatrixXd& U);
MatrixXd Ttheta (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx1, const MatrixXd& wMtx2, const MatrixXd& Eta, const MatrixXd& U);
MatrixXd Ttheta0 (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx1,const MatrixXd& wMtx2, const MatrixXd& Eta, const MatrixXd& U);
MatrixXd multinomialtheta (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx, const MatrixXd& Eta, const MatrixXd& U);
MatrixXd multinomialtheta0 (const MatrixXd& y, const Psi& psi, const MatrixXd& graph, const MatrixXd& wMtx, const MatrixXd& Eta, const MatrixXd& U);
double etamax(const Matrix<double, 1, Dynamic>& z, double lambada);
VectorXd softThresholding(const VectorXd& z, double lambda);
VectorXd adaptiveLassoUpdate(double u, const Matrix<double, 1, Dynamic>& z, double lambda, double a);
VectorXd scadLLAUpdate(double u, const Matrix<double, 1, Dynamic>& z, const Matrix<double, 1, Dynamic>& eta, double lambda, double a);
VectorXd mcpLLAUpdate(double u, const Matrix<double, 1, Dynamic>& z, const Matrix<double, 1, Dynamic>& eta, double lambda, double a);
double GetSCAD(const Matrix<double, 1, Dynamic>& eta, double lambda, double a);
double GetMCP(const Matrix<double, 1, Dynamic>& eta, double lambda, double a);
double GetUnweight(const Matrix<double, 1, Dynamic>& eta, double lambda, double a);
double GetALasso(const Matrix<double, 1, Dynamic>& eta, double lambda, double a);

/*** Auxiliary functions for Normal mixtures in location. ***/

double densityNormalLoc (const Matrix<double, 1, Dynamic>& y,   
                         const Matrix<double, Dynamic, 1>& theta,   
                         const MatrixXd& sigma);
MatrixXd gradBNormalLoc (const MatrixXd& theta, const MatrixXd& sigma);
double bNormalLoc (const VectorXd& theta, const MatrixXd& sigma);
MatrixXd tNormalLoc (const MatrixXd& y); 
MatrixXd transfNormalLoc (const MatrixXd& theta, const MatrixXd& sigma);
MatrixXd invTransfNormalLoc (const MatrixXd& theta, const MatrixXd& sigma);
bool constrCheckNormalLoc (const MatrixXd& theta);

/*** Multivariate Location T auxiliary functions begin. ***/


double densityT (const Matrix<double, 1, Dynamic>& y,   
                 const Matrix<double, Dynamic, 1>& theta,   
                 const MatrixXd& sigma, const double nu);
MatrixXd gradBT (const MatrixXd& theta, const MatrixXd& sigma);
double bT (const VectorXd& theta, const MatrixXd& sigma);
MatrixXd tT (const MatrixXd& y);
MatrixXd transfT (const MatrixXd& theta, const MatrixXd& sigma);
MatrixXd invTransfT (const MatrixXd& theta, const MatrixXd& sigma);
bool constrCheckT (const MatrixXd& theta);
int dfT (int k, bool arbSigma);


/*** Auxiliary functions for multinomial mixtures. ***/

double densityMultinomial(const Matrix<double, 1, Dynamic>& y,
                          const Matrix<double, Dynamic, 1>& theta,
                          const MatrixXd& sigma);
MatrixXd gradBMultinomial (const MatrixXd& theta, const MatrixXd& sigma);
double bMultinomial (const VectorXd& theta, const MatrixXd& sigma);
MatrixXd hesBMultinomial (const VectorXd& theta, const MatrixXd& sigma);
MatrixXd tMultinomial (const MatrixXd& y);
MatrixXd transfMultinomial (const MatrixXd&);
MatrixXd invTransfMultinomial (const MatrixXd&, const MatrixXd&);
bool constrCheckMultinomial (const MatrixXd& theta);

