// no-networkit-format
/*
 * ONLP.cpp
 *
 *  Created on: 11.21.2021
 *      Author: Md Maruf Hossain
 */

#include <omp.h>

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Parallelism.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/community/ONLP.hpp>
#include <iostream>
#include <vector>

namespace NetworKit {

ONLP::ONLP(const Graph& G, count theta, count maxIterations) : CommunityDetectionAlgorithm(G), updateThreshold(theta), maxIterations(maxIterations) {}

ONLP::ONLP(const Graph &G, const Partition &baseClustering, count theta)
    : CommunityDetectionAlgorithm(G, baseClustering), updateThreshold(theta) {}

void ONLP::run() {
    if (hasRun) {
        throw std::runtime_error("The algorithm has already run on the graph.");
    }

    // set unique label for each node if no baseClustering was given
    index z = G->upperNodeIdBound();
    if (result.numberOfElements() != z) {
        result = Partition(z);
        result.allToSingletons();
    }

    using label = index; // a label is the same as a cluster id

    count n = G->numberOfNodes();
    // update threshold heuristic
    if (updateThreshold == none) {
        updateThreshold = (count) (n / 1e5);
    }

    count nUpdated; // number of nodes which have been updated in last iteration
    nUpdated = n; // all nodes have new labels -> first loop iteration runs

    nIterations = 0; // number of iterations

    /**
     * == Dealing with isolated nodes ==
     *
     * The pseudocode published does not deal with isolated nodes (and therefore does not terminate if they are present).
     * Isolated nodes stay singletons. They can be ignored in the while loop, but the loop condition must
     * compare to the number of non-isolated nodes instead of n.
     *
     * == Termination criterion ==
     *
     * The published termination criterion is: All nodes have got the label of the majority of their neighbors.
     * In general this does not work. It was changed to: No label was changed in last iteration.
     */

    std::vector<bool> activeNodes(z); // record if node must be processed
    activeNodes.assign(z, true);

    Aux::Timer runtime;

    index omega = result.upperBound();
    std::vector<index> data(z);
    std::vector<count> outDegree;
    const std::vector<f_weight> *outEdgeWeights;
    const std::vector<node> *outEdges;
    bool isGraphWeighted = G->isWeighted();
    bool hasEdgeId = G->hasEdgeIds();
    outEdgeWeights = G->getOutEdgeWeights();
    outEdges = G->getOutEdges();
    index max_sub = 0, min_sub = z;
    for (index i=0; i<z; ++i) {
        outDegree.push_back(outEdges[i].size());
        data[i] = i;
        if(data[i] > max_sub)
            max_sub = data[i];
        if(data[i] < min_sub)
            min_sub = data[i];
    }

    std::vector<std::vector<f_weight> >labelWeights(omp_get_max_threads(), std::vector<f_weight>(omega));
    std::vector<std::vector<f_weight> >uniqueLabels(omp_get_max_threads(), std::vector<f_weight>(z));

    std::cout<< "maxIterations: " << maxIterations << " max threads: " << omp_get_max_threads()
              << " Threshold: " << this->updateThreshold << " max sub: " << max_sub << " min sub: "
              << min_sub << " weighted: " << isGraphWeighted << " hasIndex: " << hasEdgeId
              << std::endl;
    // propagate labels
    while ((nUpdated > this->updateThreshold)  && (nIterations < maxIterations)) { // as long as a label has changed... or maximum iterations reached
        runtime.start();
        nIterations += 1;
        DEBUG("[BEGIN] LabelPropagation: iteration #" , nIterations);
//        std::cout<< "[BEGIN] LabelPropagation: iteration #" << nIterations << std::endl;
        // reset updated
        nUpdated = 0;
/*#pragma omp parallel for schedule(guided)
        for (omp_index v = 0; v < static_cast<omp_index>(z); ++v){
            if (G->hasNode(v) && (activeNodes[v]) && (G->degree(v) > 0)) {
                index tid = omp_get_thread_num();
                for (int i = 0; i < outEdges[v].size(); ++i) {
                    node w = outEdges[v][i];
                    label lw = data[w];
                    labelWeights[tid][lw] = -1;
                }
                index _cnt = 0;
                for (int i = 0; i < outEdges[v].size(); ++i) {
                    node w = outEdges[v][i];
                    label lw = data[w];
                    if (labelWeights[tid][lw] == -1) {
                        labelWeights[tid][lw] = 0;
                        uniqueLabels[tid][_cnt++] = lw;
                    }
                    labelWeights[tid][lw] += isGraphWeighted ? outEdgeWeights[v][i] : fdefaultEdgeWeight;
                }

                // get heaviest label
                label heaviest = -1;
                f_weight _heavyWeight = -1;
                label lv = data[v];
                for (int i = 0; i < _cnt; ++i) {
                    label lw = uniqueLabels[tid][i];
                    if (labelWeights[tid][lw] > _heavyWeight) {
                        heaviest = lw;
                        _heavyWeight = labelWeights[tid][lw];
                    }
                }
                if (heaviest != -1 && lv != heaviest) { // UPDATE
                    data[v] = heaviest; //result[v] = heaviest;
                    nUpdated += 1; // TODO: atomic update?
                    for (int i = 0; i < outEdges[v].size(); ++i) {
                        node u = outEdges[v][i];
                        activeNodes[u] = true;
                    }
                } else {
                    activeNodes[v] = false;
                }

            } else {
                // node is isolated
            }
        }*/

        G->balancedParallelForNodes([&](node v){
            if ((activeNodes[v]) && (G->degree(v) > 0)) {
                index tid = omp_get_thread_num();
                std::map<label, double> labelWeights2;
                for (int i = 0; i < outEdges[v].size(); ++i) {
                    node w = outEdges[v][i];
                    label lw = data[w];
                    labelWeights[tid][lw] = -1;
                }
                index _cnt = 0;
                for (int i = 0; i < outEdges[v].size(); ++i) {
                    node w = outEdges[v][i];
                    label lw = data[w];
                    if (labelWeights[tid][lw] == -1) {
                        labelWeights[tid][lw] = 0;
                        uniqueLabels[tid][_cnt++] = lw;
                    }
                    labelWeights[tid][lw] += isGraphWeighted ? outEdgeWeights[v][i] : fdefaultEdgeWeight;
                }

                // weigh the labels in the neighborhood of v
                for (int i = 0; i < outEdges[v].size(); ++i) {
                    node w = outEdges[v][i];
                    label lw = data[w]; //result.subsetOf(w);
                    labelWeights2[lw] += isGraphWeighted ? outEdgeWeights[v][i] : fdefaultEdgeWeight; // add weight of edge {v, w}
                }
                assert(_cnt == labelWeights2.size());
                for (auto m : labelWeights2) {
                    assert(m.second == labelWeights[tid][m.first]);
                }

                // get heaviest label
                label heaviest = -1;
                label heaviest2 = -1;
                f_weight _heavyWeight = -1;
                f_weight _heavyWeight2 = -1;
                label lv = data[v];
                for (auto m : labelWeights2) {
                    label lw = m.first;
                    if (m.second > _heavyWeight) {
                        heaviest = lw;
                        _heavyWeight = m.second;
                    }
                }
                for (int i = 0; i < _cnt; ++i) {
                    label lw = uniqueLabels[tid][i];
                    if (labelWeights[tid][lw] > _heavyWeight2) {
                        heaviest2 = lw;
                        _heavyWeight2 = labelWeights[tid][lw];
                    }
                }
                assert(heaviest2 == heaviest);
                if (heaviest2 != -1 && lv != heaviest2) { // UPDATE
                    data[v] = heaviest2; //result[v] = heaviest;
                    nUpdated += 1; // TODO: atomic update?
                    for (int i = 0; i < outEdges[v].size(); ++i) {
                        node u = outEdges[v][i];
                        activeNodes[u] = true;
                    }
                } else {
                    activeNodes[v] = false;
                }

            } else {
                // node is isolated
            }
        });

        // for each while loop iteration...

        runtime.stop();
        this->timing.push_back(runtime.elapsedMilliseconds());
        DEBUG("[DONE] LabelPropagation: iteration #" , nIterations , " - updated " , nUpdated , " labels, time spent: " , runtime.elapsedTag());
//        std::cout<< "[DONE] LabelPropagation: iteration #" << nIterations  << " - updated "<< nUpdated << " labels, time spent: " << runtime.elapsedTag() << std::endl;

    } // end while
    for (index i=0; i<z; ++i) {
        result.moveToSubset(data[i], i);
    }
    hasRun = true;
}

std::string ONLP::toString() const {
    std::stringstream strm;
    strm << "ONLP";
    return strm.str();
}

void ONLP::setUpdateThreshold(count th) {
    this->updateThreshold = th;
}

count ONLP::numberOfIterations() {
    return this->nIterations;
}

std::vector<count> ONLP::getTiming() {
    return this->timing;
}

} /* namespace NetworKit */
