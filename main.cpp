#include <iostream>
#include <vector>

#include "Timer.hpp"
#include "cuDBSCAN.hpp"
#include "DBSCANCPU.hpp"

#include <random>
#include <cassert>

using namespace std;

int main(int argc, char * argv[])
{
    int N = 20000;
    double eps = 0.1;
    int minClusterSize = 5;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    std::vector<Point2> vec;

    for(int i = 0; i < N; i++) {
        float r1 = static_cast<float>(dis(gen));
        float r2 = static_cast<float>(dis(gen));

        vec.push_back(Point2{r1, r2});
    }

    Timer t;
    cuDBSCAN scanCUDA(vec, eps, minClusterSize);
    int nClustersCUDA = scanCUDA.run();
    double timeCUDA = t.elapsed();
    std::cout << "[CUDA] Number of clusters: " << nClustersCUDA << "\t Time: " << timeCUDA << std::endl;

    t.reset();
    DBSCANCPU scanCPU(vec, eps, minClusterSize);
    int nCluestersCPU = scanCPU.run();
    double timeCPU = t.elapsed();
    std::cout << "[CPU] Number of clusters: " << nCluestersCPU << "\t Time: " << timeCPU << std::endl;
    std::cout << "Speedup was: " << timeCPU / timeCUDA <<std::endl;

    // Verify the results
    assert(nClustersCUDA = nCluestersCPU);
    for (int i = 0; i < nClustersCUDA; i++) {
        assert(scanCUDA.clusters[i].size() == scanCPU.clusters[i].size());

        for(int j = 0; j < scanCUDA.clusters[i].size(); j++) {
            assert(scanCUDA.clusters[i][j] == scanCPU.clusters[i][j]);
        }
    }
}
