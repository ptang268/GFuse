#include "gfuse.h"

bool isOrdered(int i, int j) {
  return (i < j);
}

double thetaDist(const VectorXd& theta1, const VectorXd& theta2) {
   double acc = 0.0;

   for (int i = 0; i < D; i++) {
     acc += pow((double)(theta1(i, 0) - theta2(i, 0)), 2);
   }

   return sqrt(acc);
 }

// Generates the symmetrix matrix (with 0 diagonal) of pairwise distances between the columns of theta.
MatrixXd getDistanceMatrix(const MatrixXd& theta) {
  int K = theta.cols();  // Number of mixture components. 
  MatrixXd distances(K, K);

  for (int k = 0; k < K; k++) {
    for (int j = 0; j < K; j++) {
      distances(k, j) = thetaDist(theta.col(j), theta.col(k));
    }
  }

  return distances;
}

// Linear search function.
bool find(std::vector<int> sigma, int j) {
  for (unsigned int i = 0; i < sigma.size(); i++) {
     if (sigma[i] == j) {
       return true;
     }
   }

   return false;
 }

// Generates the permutation alpha.
void alpha(const MatrixXd& theta, std::vector<int>& perm) {
  int K = theta.cols();  // Number of mixture components. 
  MatrixXd distances = getDistanceMatrix(theta);

   std::vector<int> sigma, tau;
   double maxEntry, tMinEntry, sMinEntry, tSum, sSum;
   int i, j, k, sResult, tResult;

   int argmaxInd[2];

   // Find the thetas which are most distant.
   maxEntry = -1;
   for (i = 0; i < distances.rows(); i++) {
     for (j = 0; j < distances.cols(); j++) {
       if (maxEntry < distances(i, j)) {
         maxEntry = distances(i, j);
         argmaxInd[0] = i;
         argmaxInd[1] = j;
       }
     }
  }

   sigma.push_back(argmaxInd[0]);
   tau.push_back(argmaxInd[1]);

   // Inductively move towards the nearest neighbor.
   for (k = 1; k < K; k++) {
     tMinEntry = sMinEntry = INT_MAX;
     sResult = -1;
     tResult = -1;

     for (j = 0; j < K; j++) {
       if (!find(sigma, j) && distances(j, sigma[k - 1]) <= sMinEntry) {
         sMinEntry = distances(j, sigma[k - 1]);
         sResult = j;
       }

       if (!find(tau, j) && distances(j, tau[k - 1]) <= tMinEntry) {
         tMinEntry = distances(j, tau[k - 1]);
         tResult = j;
       }
     }

     sigma.push_back(sResult);
     tau.push_back(tResult);
   }

   // Determine whether tau or sigma defines the shortest path, and choose it
   // as the permutation alpha.
   tSum = sSum = 0;
   for (k = 0; k < K; k++) {
     tSum += distances(k, tau[k]);
     sSum += distances(k, sigma[k]);
   }

   if (tSum < sSum) {
     for (k = 0; k < K; k++) {
       perm[k] = tau[k];
     }

   }
   else {
     for (k = 0; k < K; k++) {
       perm[k] = sigma[k];
     }
  }
 }

 // Reorders the columns of theta with respect to the permutation alpha.
 MatrixXd reorderTheta(const MatrixXd& theta) {
   int K = theta.cols();  // Number of mixture components. 
   MatrixXd result(D, K);
   std::vector<int> perm(K);

   alpha(theta, perm);

   for (int k = 0; k < K; k++) {
     result.col(k) = theta.col(perm[k]);
   }

   return result;
 }

Psi reorderResult(const Psi& psi) {
  int K = psi.theta.cols();  // Number of mixture components. 
  MatrixXd thetaResult(D, K);
  VectorXd piiResult(K);
  std::vector<int> perm(K);
  Psi newPsi;

  alpha(psi.theta, perm);

  for(int k = 0; k < K; k++){
    thetaResult.col(k) = psi.theta.col(perm[k]);
    piiResult(k)       = psi.pii(perm[k]);
  }

  newPsi.theta = thetaResult;
  newPsi.pii   = piiResult;
  newPsi.sigma = psi.sigma;

  return newPsi;
}

// Generates the graphs
//gsf
MatrixXd graphgsf(const MatrixXd& theta) {
  int K = theta.cols();  // Number of mixture components. 
  MatrixXd distances = getDistanceMatrix(theta);
  MatrixXd graph = MatrixXd::Zero(K,K);
  
  std::vector<int> sigma, tau;
  double maxEntry, tMinEntry, sMinEntry, tSum, sSum;
  int i, j, k, sResult, tResult;
  
  int argmaxInd[2];
  
  // Find the thetas which are most distant.
  maxEntry = -1;
  for (i = 0; i < distances.rows(); i++) {
    for (j = 0; j < distances.cols(); j++) {
      if (maxEntry < distances(i, j)) {
        maxEntry = distances(i, j);
        argmaxInd[0] = i;
        argmaxInd[1] = j;
      }
    }
  }
  
  sigma.push_back(argmaxInd[0]);
  tau.push_back(argmaxInd[1]);
  
  // Inductively move towards the nearest neighbor.
  for (k = 1; k < K; k++) {
    tMinEntry = sMinEntry = INT_MAX;
    sResult = -1;
    tResult = -1;
    
    for (j = 0; j < K; j++) {
      if (!find(sigma, j) && distances(j, sigma[k - 1]) <= sMinEntry) {
        sMinEntry = distances(j, sigma[k - 1]);
        sResult = j;
      }
      
      if (!find(tau, j) && distances(j, tau[k - 1]) <= tMinEntry) {
        tMinEntry = distances(j, tau[k - 1]);
        tResult = j;
      }
    }
    
    sigma.push_back(sResult);
    tau.push_back(tResult);
  }
  
  // Determine whether tau or sigma defines the shortest path, and choose it
  // as the permutation alpha.
  tSum = sSum = 0;
  /*   for (k = 0; k < K; k++) {
   tSum += distances(k, tau[k]);
   sSum += distances(k, sigma[k]);
  }
   */   
  for (k = 1; k < K; k++) {
    tSum += distances(tau[k-1], tau[k]);
    sSum += distances(sigma[k-1], sigma[k]);
  }
  
  if (tSum < sSum) {
    for (k = 1; k < K; k++) {
      graph(sigma[k-1],sigma[k]) = 1;
      graph(sigma[k],sigma[k-1]) = 1;
    }
    
  }
  else {
    for (k = 1; k < K; k++) {
      graph(tau[k-1],tau[k]) = 1;
      graph(tau[k],tau[k-1]) = 1;
    }
  }
  return graph;
}


/*mnn*/
/*MatrixXd graphmnn(const MatrixXd& theta, int m) {
  int K = theta.cols();
  MatrixXd distances = getDistanceMatrix(theta);
  MatrixXd graph = MatrixXd::Zero(K,K);
  
  //int n = distMatrix.rows();
  //Eigen::MatrixXd knnGraph(n, n);
  
  for (int i = 0; i < K; ++i) {
    // Get distances and indices for the current point
    Eigen::VectorXd distancesvec = distances.row(i);
    Eigen::VectorXi indices(K);
    for (int j = 0; j < K; ++j) {
      indices(j) = j;
    }
    
    // Exclude self (point i) and sort by distances
    std::vector<std::pair<double, int>> distIndexPairs;
    for (int j = 0; j < K; ++j) {
      if (j != i) {
        distIndexPairs.push_back(std::make_pair(distancesvec(j), indices(j)));
      }
    }
    std::sort(distIndexPairs.begin(), distIndexPairs.end());
    
    // Take the m nearest neighbors
    for (int j = 0; j < m; ++j) {
      int neighborIndex = distIndexPairs[j].second;
      graph(i, neighborIndex) = 1;
      graph(neighborIndex, i) = 1;
    }
  }
  
  return graph;
}
 */

MatrixXd graphmnn(const MatrixXd& theta, int m) {
  int K = theta.cols();
  MatrixXd graph = MatrixXd::Zero(K, K);
  
  // If m >= K-1, return a complete graph
  if (m >= K - 1) {
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < K; ++j) {
        if (i != j) {
          graph(i, j) = 1;
        }
      }
    }
    return graph;
  } else {
  
  MatrixXd distances = getDistanceMatrix(theta);
  
  for (int i = 0; i < K; ++i) {
    Eigen::VectorXd distancesvec = distances.row(i);
    Eigen::VectorXi indices(K);
    for (int j = 0; j < K; ++j) {
      indices(j) = j;
    }
    
    std::vector<std::pair<double, int>> distIndexPairs;
    for (int j = 0; j < K; ++j) {
      if (j != i) {
        distIndexPairs.push_back(std::make_pair(distancesvec(j), indices(j)));
      }
    }
    std::sort(distIndexPairs.begin(), distIndexPairs.end());
    
    for (int j = 0; j < m; ++j) {
      int neighborIndex = distIndexPairs[j].second;
      graph(i, neighborIndex) = 1;
      graph(neighborIndex, i) = 1;
    }
  }
  
  return graph;
  }}



// Helper function to find the nearest unvisited neighbor
int findNearestNeighbor(const VectorXi& visited, const MatrixXd& distances, int current) {
  int numPoints = distances.rows();
  double minDistance = std::numeric_limits<double>::max();
  int nearestNeighbor = -1;
  
  for (int i = 0; i < numPoints; i++) {
    if (i != current && visited[i] == 0 && distances(current, i) < minDistance) {
      minDistance = distances(current, i);
      nearestNeighbor = i;
    }
  }
  
  return nearestNeighbor;
}

// Helper function to find the longest edge in the graph
std::pair<int, int> findLongestEdge(const MatrixXd& distances, const MatrixXd& graph) {
  int numPoints = distances.rows();
  double maxDistance = -1.0;
  int point1 = -1;
  int point2 = -1;
  
  for (int i = 0; i < numPoints; i++) {
    for (int j = i + 1; j < numPoints; j++) {
      if (graph(i, j) == 1 && distances(i, j) > maxDistance) {
        maxDistance = distances(i, j);
        point1 = i;
        point2 = j;
      }
    }
  }
  
  return std::make_pair(point1, point2);
}

// Main function to solve the TSP and return the graph
MatrixXd graphmGSF(const Eigen::MatrixXd& theta,int m) {
  int numPoints = theta.cols();
  MatrixXd distances = getDistanceMatrix(theta);
  MatrixXd graph = MatrixXd::Zero(numPoints, numPoints);
  VectorXi visited = VectorXi::Zero(numPoints);
  
  // Find the two points with the longest distance to start the TSP
  std::pair<int, int> startPoints = findMaxDistanceIndices(distances);
  int currentPoint = startPoints.first;
  int lastPoint = startPoints.first;
  visited[currentPoint] = 1;
  //visited[nextPoint] = 1;
  
  // Iterate through the points to build the TSP path
  for (int i = 0; i < numPoints - 1; i++) {
    int nearestNeighbor = findNearestNeighbor(visited, distances, currentPoint);
    
    if (nearestNeighbor != -1) {
      graph(currentPoint, nearestNeighbor) = 1;
      graph(nearestNeighbor, currentPoint) = 1;
      currentPoint = nearestNeighbor;
      visited[currentPoint] = 1;
    }
  }
  
  // Connect the last point to the other starting point
  graph(currentPoint, lastPoint) = 1;
  graph(lastPoint, currentPoint) = 1;
  
  // Remove longest edges to achieve "m" segments
  for (int i = 0; i < m; i++) {
    std::pair<int, int> longestEdge = findLongestEdge(distances, graph);
    int point1 = longestEdge.first;
    int point2 = longestEdge.second;
    graph(point1, point2) = 0;
    graph(point2, point1) = 0;
  }
  
  return graph;
}

// Helper function to find the indices of the points with the maximum distance
std::pair<int, int> findMaxDistanceIndices(const MatrixXd& distances) {
  int numPoints = distances.rows();
  double maxDistance = -1.0;
  int point1 = -1;
  int point2 = -1;
  
  for (int i = 0; i < numPoints; i++) {
    for (int j = i + 1; j < numPoints; j++) {
      if (distances(i, j) > maxDistance) {
        maxDistance = distances(i, j);
        point1 = i;
        point2 = j;
      }
    }
  }
  
  return std::make_pair(point1, point2);
}

// MatrixXd findMinimumSpanningTree(const Eigen::MatrixXd& distances) {
//   int numPoints = distances.rows();
//   Eigen::MatrixXd mst = Eigen::MatrixXd::Zero(numPoints, numPoints);
//   std::vector<std::pair<double, std::pair<int, int>>> edges;
//   
//   // Create a list of edges sorted by distance
//   for (int i = 0; i < numPoints; i++) {
//     for (int j = i + 1; j < numPoints; j++) {
//       edges.push_back(std::make_pair(distances(i, j), std::make_pair(i, j)));
//     }
//   }
//   
//   std::sort(edges.begin(), edges.end());
//   
//   // Initialize disjoint set data structure
//   std::vector<int> parent(numPoints);
//   for (int i = 0; i < numPoints; i++) {
//     parent[i] = i;
//   }
//   
//   // Kruskal's algorithm to build the MST
//   for (auto edge : edges) {
//     int u = edge.second.first;
//     int v = edge.second.second;
//     
//     // Check if adding the edge creates a cycle
//     int parentU = u;
//     int parentV = v;
//     while (parentU != parent[parentU]) {
//       parentU = parent[parentU];
//     }
//     while (parentV != parent[parentV]) {
//       parentV = parent[parentV];
//     }
//     
//     if (parentU != parentV) {
//       // Add the edge to the MST
//       mst(u, v) = 1;
//       mst(v, u) = 1;
//       
//       // Merge the two disjoint sets
//       parent[parentU] = parentV;
//     }
//   }
//   
//   return mst;
// }

MatrixXd findMinimumSpanningTree(const Eigen::MatrixXd& distances) {
  int numPoints = distances.rows();
  MatrixXd mst = MatrixXd::Zero(numPoints, numPoints);
  
  std::set<std::pair<double, int>> minEdges;
  std::vector<bool> inMST(numPoints, false);
  
  // Start with the first vertex
  minEdges.insert({0, 0});
  std::vector<double> minWeight(numPoints, std::numeric_limits<double>::max());
  minWeight[0] = 0;
  std::vector<int> parent(numPoints, -1);
  
  while (!minEdges.empty()) {
    int u = minEdges.begin()->second;
    minEdges.erase(minEdges.begin());
    
    if (!inMST[u]) {
      inMST[u] = true;
      
      for (int v = 0; v < numPoints; v++) {
        if (!inMST[v] && distances(u, v) > 0 && distances(u, v) < minWeight[v]) {
          minWeight[v] = distances(u, v);
          minEdges.insert({distances(u, v), v});
          parent[v] = u;
        }
      }
    }
  }
  
  for (int i = 1; i < numPoints; i++) {
    if (parent[i] >= 0) {
      mst(i, parent[i]) = 1;
      mst(parent[i], i) = 1;
    }
  }
  
  return mst;
}
// Main function to generate "m" minimum spanning trees as graphs
MatrixXd graphmMST(const Eigen::MatrixXd& theta, int m) {
  MatrixXd distances = getDistanceMatrix(theta);
  MatrixXd graph = findMinimumSpanningTree(distances);
  
  // Remove longest edges to achieve "m" segments
  for (int i = 0; i < m-1; i++) {
    std::pair<int, int> longestEdge = findLongestEdge(distances, graph);
    int point1 = longestEdge.first;
    int point2 = longestEdge.second;
    graph(point1, point2) = 0;
    graph(point2, point1) = 0;
  }
  
  return graph;
}

// Helper function to perform DFS and find all points in a connected component
void dfs(int node, const MatrixXd& finalgraph, std::vector<bool>& visited, std::vector<int>& component) {
  visited[node] = true;
  component.push_back(node);
  
  for (int i = 0; i < finalgraph.cols(); ++i) {
    if (finalgraph(node, i) == 1 && !visited[i]) {
      dfs(i, finalgraph, visited, component);
    }
  }
}

Psi mergeComponents(const Psi& psi, const MatrixXd& finalgraph) {
  int K = psi.theta.cols();
  std::vector<bool> visited(K, false);
  std::vector<VectorXd> newTheta;
  std::vector<double> newPii;
  
  for (int i = 0; i < K; ++i) {
    if (!visited[i]) {
      std::vector<int> component;
      dfs(i, finalgraph, visited, component); // Find all points in the component
      
      VectorXd componentTheta = psi.theta.col(component[0]); // Use the first point's theta
      double componentPii = 0.0;
      for (int idx : component) {
        componentPii += psi.pii[idx];
      }
      
      newTheta.push_back(componentTheta);
      newPii.push_back(componentPii);
    }
  }
  
  // Convert std::vector to Eigen types
  int newK = newTheta.size();
  MatrixXd mergedTheta(psi.theta.rows(), newK);
  VectorXd mergedPii(newK);
  
  for (int i = 0; i < newK; ++i) {
    mergedTheta.col(i) = newTheta[i];
    mergedPii(i) = newPii[i];
  }
  
  // Create a new Psi object with merged values
  Psi newPsi;
  newPsi.theta = mergedTheta;
  newPsi.pii = mergedPii;
  newPsi.sigma = psi.sigma; // Assuming sigma is to be kept unchanged
  
  return newPsi;
}

void dfsEqualize(int node, const MatrixXd& finalgraph, std::vector<bool>& visited, std::vector<int>& component) {
  visited[node] = true;
  component.push_back(node);
  
  for (int i = 0; i < finalgraph.cols(); ++i) {
    if (finalgraph(node, i) == 1 && !visited[i]) {
      dfsEqualize(i, finalgraph, visited, component);
    }
  }
}

Psi equalizeThetaInClusters(const Psi& psi, const MatrixXd& finalgraph) {
  int K = psi.theta.cols();
  std::vector<bool> visited(K, false);
  MatrixXd newTheta = psi.theta;
  
  for (int i = 0; i < K; ++i) {
    if (!visited[i]) {
      std::vector<int> component;
      dfsEqualize(i, finalgraph, visited, component);
      
      // Debug: Print the identified component
      //Rcpp::Rcout << "Component starting at " << i+1 << ": ";
      //for (int idx : component) {
        //Rcpp::Rcout << idx << " ";
      //}
      //Rcpp::Rcout << "\n";
      
      // Choose the theta of the first node as the representative
      VectorXd representativeTheta = newTheta.col(component[0]);
      
      // Equalize theta for all nodes in the component
      for (int idx : component) {
        newTheta.col(idx) = representativeTheta;
      }
    }
  }
  
  // Debug: Print the new theta matrix
  //Rcpp::Rcout << "New Theta: \n" << newTheta << "\n";
  
  // Create a new Psi object with the updated theta values
  Psi newPsi;
  newPsi.theta = newTheta;
  newPsi.pii = psi.pii; // Keeping pii values unchanged
  newPsi.sigma = psi.sigma; // Assuming sigma is to be kept unchanged
  
  return newPsi;
}



int countClusters(MatrixXd& graph) {
  int K = graph.rows();
  int Order = 0; // Number of clusters
  
  // Vector to keep track of visited nodes
  std::vector<bool> visited(K, false);
  
  // Depth-First Search (DFS) function to explore a cluster
  std::function<void(int)> dfs = [&](int node) {
    visited[node] = true;
    
    for (int i = 0; i < K; ++i) {
      if (graph(node, i) == 1 && !visited[i]) {
        dfs(i);
      }
    }
  };
  
  for (int i = 0; i < K; ++i) {
    if (!visited[i]) {
      // Start a new cluster if the current node is unvisited
      dfs(i);
      Order++;
    }
  }
  
  return Order;
}

