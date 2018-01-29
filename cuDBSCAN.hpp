#pragma once

#include <map>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>

#include "Point2.hpp"

#define cuda_call(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct KernelObject
{
    float eps;
    int mnpts;
    int pointCount;

    Point2 *data = nullptr;
    int *clusterMatrix = nullptr;
    int *valid = nullptr;
    int *reroute = nullptr;
    int *mergeHappened = nullptr;

    KernelObject* devPtr;

    __device__ inline void set(int i, int tid)
    {
        clusterMatrix[i * pointCount + tid] = 1;
        valid[tid] = 1;
    }

    __device__ inline int get(int i, int tid)
    {
        return clusterMatrix[i * pointCount + tid];
    }

    __device__ inline bool isValid(int tid)
    {
        return valid[tid] == 1;
    }

    __device__ inline int getReroutedIndex(int index)
    {
        do {
            if (valid[index])
                return index;

            index = reroute[index];
        } while (index != -1);

        // This should never happen!
        return -1;
    }

    __device__ float computeDistance(int ai, int bi)
    {
        Point2 a = data[ai];
        Point2 b = data[bi];
        float deltaX = a.x - b.x;
        float deltaY = a.y - b.y;
        return sqrtf(deltaX * deltaX + deltaY * deltaY);
    }

    __host__ void allocate()
    {
        cuda_call(cudaMalloc(&data, pointCount * sizeof(Point2)));
        cuda_call(cudaMalloc(&clusterMatrix, pointCount * pointCount * sizeof(int)));
        cuda_call(cudaMalloc(&valid, pointCount * sizeof(int)));
        cuda_call(cudaMalloc(&reroute, pointCount * sizeof(int)));
        cuda_call(cudaMalloc(&mergeHappened, sizeof(int)));
        cuda_call(cudaMalloc(&devPtr, sizeof(KernelObject)));

        // Initialize pointers and values
        cuda_call(cudaMemset(clusterMatrix, 0, pointCount * pointCount * sizeof(int)));
        cuda_call(cudaMemset(valid, 0, pointCount * sizeof(int)));
        cuda_call(cudaMemset(reroute, 0, pointCount * sizeof(int)));
        cuda_call(cudaMemset(mergeHappened, 0, sizeof(int)));
    }

    __host__ void free()
    {
        if (data != nullptr)
            cuda_call(cudaFree(data));

        if (clusterMatrix != nullptr)
            cuda_call(cudaFree(clusterMatrix));

        if (valid != nullptr)
            cuda_call(cudaFree(valid));

        if (reroute != nullptr)
            cuda_call(cudaFree(reroute));

        if (mergeHappened != nullptr)
            cuda_call(cudaFree(mergeHappened));

        data = nullptr;
        clusterMatrix = nullptr;
        valid = nullptr;
        reroute = nullptr;
        mergeHappened = nullptr;
    }
};

class cuDBSCAN
{
public:
    std::map<int, std::vector<int>> clusters;
    std::map<int, int> labels;

    KernelObject ko;
    cuDBSCAN(std::vector<Point2>& points, float eps, int mnpts);
    void setData(std::vector<Point2> points);
    int run();

    bool merge(int clusterIndex);

    ~cuDBSCAN();
};

