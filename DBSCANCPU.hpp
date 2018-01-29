#pragma once

#include <map>
#include <sstream>
#include <vector>
#include <cmath>

class DBSCANCPU
{
public:
    std::vector<Point2>& data;
    double eps;
    int mnpts;
    int pointCount;

    std::vector<std::vector<int>> clusterMatrix;
    std::vector<int> valid;
    std::vector<int> reroute;
    std::map<int, int> labels;
    std::map<int, std::vector<int>> clusters;

    DBSCANCPU(std::vector<Point2>& _data,double _eps,int _mnpts):data(_data), eps(_eps), mnpts(_mnpts)
    {
        pointCount = _data.size();
        clusterMatrix.resize(pointCount);
        valid.resize(pointCount);
        reroute.resize(pointCount);

        for (int i = 0; i < pointCount; i++) {
            clusterMatrix[i].resize(pointCount);
            valid[i] = 0;
            reroute[i] = -1;
            labels[i]=-99;
        }
    }

    int run()
    {
        for(int tid = 0 ; tid < pointCount; tid++) {
            for (int i = 0; i < pointCount; i++) {
                // Same elem is always 1
                if (i == tid) {
                    set(i, tid);
                    continue;
                }

                // Compute the distance
                double dist = computeDistance(i, tid);
                if (dist <= eps)
                    set(i, tid);
            }
        }

        // Now merge the clusters
        for(int i = 0; i < pointCount; i++)
            while (merge(i)); // Means merge everybody into this one

        // Now set the labels!

        int runningClusterIndex = 0;

        for(int col = 0 ; col < pointCount ; col++) {
            if (!valid[col])
                continue;

            std::vector<int> assignedIndices;

            for (int row = 0 ; row < pointCount ; row++)
                if (clusterMatrix[row][col])
                    assignedIndices.push_back(row);

            if (assignedIndices.size() > mnpts) {
                clusters[runningClusterIndex++] = assignedIndices;
            }
        }

        for(auto &c : clusters) {
            for(auto &index : c.second) {
                labels[index] = c.first;
            }
        }

        return clusters.size();
    }

    bool merge(int clusterIndex)
    {
        int correctedIndex = clusterIndex;

        if (!valid[clusterIndex] && reroute[clusterIndex] >= 0)
            correctedIndex = reroute[clusterIndex];
        else if (!valid[clusterIndex])
            throw std::runtime_error("Invalid index!");


        return mergeInto(correctedIndex);
    }

    bool mergeInto(int clusterIndex)
    {
        bool thisClusterWasMerged = false;

        for(int tid = 0 ; tid < pointCount ; tid++) {

            if (tid == clusterIndex)
                continue;

            if (valid[tid])
                for (int i = 0 ; i < pointCount ; i++) {
                    // Are both one?
                    if (clusterMatrix[i][tid] && clusterMatrix[i][clusterIndex]) {
                        copyClusterAndInvalidate(tid, clusterIndex);
                        thisClusterWasMerged = true;
                        break; // no need to go any further!
                    }
                }
        }

        return thisClusterWasMerged;
    }

    void copyClusterAndInvalidate(int from, int to) {
        for(int i = 0 ; i < pointCount ; i++) {
            clusterMatrix[i][to] |= clusterMatrix[i][from];
            clusterMatrix[i][from] = 0; // so that we know this was de-assigned
        }

        valid[from] = false;
        reroute[from] = to;
    }

    void set(int i, int tid)
    {
        clusterMatrix[i][tid] = 1;
        valid[tid] = 1;
    }

    double computeDistance(int ai,int bi)
    {
        Point2 a = data[ai];
        Point2 b = data[bi];
        float deltaX = a.x - b.x;
        float deltaY = a.y - b.y;
        return std::sqrt(deltaX * deltaX + deltaY * deltaY);
    }
};
