#include "cuDBSCAN.hpp"

__global__ void assignInitialClusters(KernelObject *cudbscan);

__global__ void mergeInto(KernelObject *cudbscan, int clusterIndex);

cuDBSCAN::cuDBSCAN(std::vector<Point2> &points, float eps, int mnpts)
{
    ko.pointCount = points.size();
    ko.eps = eps;
    ko.mnpts = mnpts;

    ko.allocate();

    for (int i = 0; i < points.size(); i++)
        labels[i] = -99;

    setData(points);
}

void cuDBSCAN::setData(std::vector<Point2> points) {
    cuda_call(cudaMemcpy(ko.data, &points[0], points.size() * sizeof(Point2), cudaMemcpyHostToDevice));
    cuda_call(cudaMemcpy(ko.devPtr, &ko, sizeof(KernelObject), cudaMemcpyHostToDevice));
}

int cuDBSCAN::run() {
    int blockSize = 32;
    int nBlocks = (int) std::ceil(ko.pointCount / (float) blockSize);

    assignInitialClusters <<< nBlocks, blockSize >>> (ko.devPtr);
    cuda_call(cudaPeekAtLastError());
    cuda_call(cudaDeviceSynchronize());

    // Now merge the clusters
    for (int i = 0; i < ko.pointCount; i++)
        while (merge(i)); // Keep merging all clusters into this one

    // Copy back the results
    int *clusterMatrix = new int[ko.pointCount * ko.pointCount];
    int *valid = new int[ko.pointCount];

    cuda_call(cudaMemcpy(clusterMatrix, ko.clusterMatrix, ko.pointCount * ko.pointCount * sizeof(int), cudaMemcpyDeviceToHost));
    cuda_call(cudaMemcpy(valid, ko.valid, ko.pointCount * sizeof(int), cudaMemcpyDeviceToHost));


    int runningClusterIndex = 0;

    for (int col = 0; col < ko.pointCount; col++) {
        if (!valid[col])
            continue;

        std::vector<int> assignedIndices;

        for (int row = 0; row < ko.pointCount; row++)
            if (clusterMatrix[row * ko.pointCount + col])
                assignedIndices.push_back(row);

        if (assignedIndices.size() > ko.mnpts)
            clusters[runningClusterIndex++] = assignedIndices;
    }

    for (auto &c : clusters)
        for (auto &index : c.second)
            labels[index] = c.first;

    delete[] clusterMatrix;
    delete[] valid;

    return clusters.size();
}

bool cuDBSCAN::merge(int clusterIndex) {
    int blockSize = 32;
    int nBlocks = (int) std::ceil(ko.pointCount / (float) blockSize);

    // Set cluster unmerged!
    cuda_call(cudaMemset(ko.mergeHappened, 0, sizeof(int)));
    mergeInto <<< nBlocks, blockSize >>> (ko.devPtr, clusterIndex);
    cuda_call(cudaPeekAtLastError());
    cuda_call(cudaDeviceSynchronize());

    // Did a merge happen?
    int merged = -1;
    cuda_call(cudaMemcpy(&merged, ko.mergeHappened, sizeof(int), cudaMemcpyDeviceToHost));

    assert(merged != -1);
    bool ret = merged > 0;

    return ret;
}

__global__ void assignInitialClusters(KernelObject *cudbscan) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= cudbscan->pointCount)
        return;

    for (int i = 0; i < cudbscan->pointCount; i++) {
        // Same elem is always 1
        if (i == tid) {
            cudbscan->set(i, tid);
            continue;
        }

        // Compute the distance
        if (cudbscan->computeDistance(i, tid) <= cudbscan->eps)
            cudbscan->set(i, tid);
    }

    cudbscan->valid[tid] = 1;
    cudbscan->reroute[tid] = -1;
}

__global__ void mergeInto(KernelObject *cudbscan, int clusterIndex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= cudbscan->pointCount ||
            !cudbscan->isValid(tid) || // Am I valid?
             tid == clusterIndex) // Am I trying to merge myself into myself?
        return;


    int targetCluster = cudbscan->getReroutedIndex(clusterIndex);

    // Don't merge me into my old self!
    if (tid == targetCluster)
        return;

    bool doMerge = false;

    for (int i = 0; i < cudbscan->pointCount; i++) {
        if (cudbscan->get(i, tid) && cudbscan->get(i, targetCluster)) {
            doMerge = true;
            break;
        }
    }

    if (doMerge) {
        // Invalidate and set reroute
        cudbscan->valid[tid] = 0;
        cudbscan->reroute[tid] = targetCluster;
        *cudbscan->mergeHappened = 1;

        // Do the merge
        for (int i = 0; i < cudbscan->pointCount; i++) {
            if (cudbscan->get(i, tid))
                cudbscan->set(i, targetCluster);
        }
    }

}

cuDBSCAN::~cuDBSCAN() {
    ko.free();
}
