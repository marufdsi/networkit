// no-networkit-format
/*
 * ONLP.cpp
 *
 *  Created on: 11.21.2021
 *      Author: Md Maruf Hossain
 */

#include <omp.h>

#include <iostream>
#include <vector>
#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Parallelism.hpp>
#include <networkit/auxiliary/Random.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/community/ONLP.hpp>

#include <emmintrin.h>
#include <immintrin.h>
#include <limits>
#include <mmintrin.h>
#include <xmmintrin.h>

namespace NetworKit {

ONLP::ONLP(const Graph &G, count theta, count maxIterations)
    : CommunityDetectionAlgorithm(G), updateThreshold(theta), maxIterations(maxIterations) {}

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
        updateThreshold = (count)(n / 1e5);
    }

    count nUpdated; // number of nodes which have been updated in last iteration
    nUpdated = n;   // all nodes have new labels -> first loop iteration runs

    nIterations = 0; // number of iterations
    __m512 default_edge_weight = _mm512_set1_ps(fdefaultEdgeWeight);
    /// 512 bit double register initialize by all 0.0
    const __m512d db_set0 = _mm512_set1_pd(0.0);
    /// 512 bit floating register initialize by all 0.0
    const __m512 fl_set0 = _mm512_set1_ps(0.0);
    /// 512 bit floating register initialize by all -1.0
    const __m512 fl_set1 = _mm512_set1_ps(-1.0);
    /// 512 bit integer register initialize by all 0
    const __m512i set0 = _mm512_set1_epi32(0x00000000);
    /// 512 bit integer register initialize by all -1
    const __m512i set1 = _mm512_set1_epi32(0xFFFFFFFF);
    /// 512 bit integer register initialize by all 1
    const __m512i set_plus_1 = _mm512_set1_epi32(1);
    /// 512 bit integer register initialize by all -1
    const __m512i set_minus_1 = _mm512_set1_epi32(-1);

    std::vector<int> activeNodes(z); // record if node must be processed
    activeNodes.assign(z, 1);

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
    for (index i = 0; i < z; ++i) {
        outDegree.push_back(outEdges[i].size());
        data[i] = i;
    }

    std::vector<std::vector<f_weight>> labelWeights(omp_get_max_threads(),
                                                    std::vector<f_weight>(omega));
    std::vector<std::vector<index>> uniqueLabels(omp_get_max_threads(),
                                                    std::vector<index>(z));

    std::cout << "maxIterations: " << maxIterations << " max threads: " << omp_get_max_threads()
              << " Threshold: " << this->updateThreshold << " weighted: " << isGraphWeighted
              << " hasIndex: " << hasEdgeId << std::endl;
    // propagate labels
    while (
        (nUpdated > this->updateThreshold)
        && (nIterations
            < maxIterations)) { // as long as a label has changed... or maximum iterations reached
        runtime.start();
        nIterations += 1;
        DEBUG("[BEGIN] LabelPropagation: iteration #", nIterations);
        std::cout<<"[BEGIN] LabelPropagation: iteration #" << nIterations << std::endl;
        //        std::cout<< "[BEGIN] LabelPropagation: iteration #" << nIterations << std::endl;
        // reset updated
        nUpdated = 0;
        if(nIterations <= 1) {
#pragma omp parallel for schedule(guided)
            for (omp_index v = 0; v < static_cast<omp_index>(z); ++v) {
                if (G->hasNode(v) && (activeNodes[v] == 1) && (G->degree(v) > 0)) {
                    index tid = omp_get_thread_num();
                    f_weight *pnt_labelWeights = &labelWeights[tid][0];
                    index *pnt_uniqueLabels = &uniqueLabels[tid][0];
                    index _deg = outEdges[v].size();
                    const node *pnt_outEdges = &outEdges[v][0];
                    index e = 0;
                    //#pragma unroll
                    /*for (e = 0; (e+16) <= _deg; e += 16) {
                        __m512i w_vec = _mm512_loadu_si512((__m512i *) &pnt_outEdges[e]);
                        __m512i lw_vec = _mm512_i32gather_epi32(w_vec, &data[0], 4);
                        _mm512_i32scatter_ps(&pnt_labelWeights[0], lw_vec, fl_set1, 4);
                    }*/
#pragma omp simd
                    for (index edge = 0; edge < _deg; ++edge) {
                        pnt_labelWeights[data[pnt_outEdges[edge]]] = -1.0;
                    }
                    index _cnt = 0;
#pragma unroll
                    for (e = 0; (e + 16) <= _deg; e += 16) {
                        __m512i w_vec = _mm512_loadu_si512((__m512i *)&pnt_outEdges[e]);
//                        _mm512_prefetch_i32gather_ps(w_vec, &data[0], sizeof(node), _MM_HINT_T0);
                        __m512i lw_vec = _mm512_i32gather_epi32(w_vec, &data[0], 4);
//                        _mm512_prefetch_i32gather_ps(lw_vec, &pnt_labelWeights[0], 4, _MM_HINT_T0);
                        __m512 labelWeight_vec =
                            _mm512_i32gather_ps(lw_vec, &pnt_labelWeights[0], 4);
                        /// label weight = -1 that means labels that come first time
                        const __mmask16 new_labels_mask =
                            _mm512_cmpeq_ps_mask(fl_set1, labelWeight_vec);
                        /// Detect conflict of the labels
                        __m512i lw_conflict = _mm512_conflict_epi32(lw_vec);
                        /// Calculate mask using compare to bits with zero on lw_conflict
                        const __mmask16 mask = _mm512_cmpeq_epi32_mask(lw_conflict, set0);
                        /// Now we need to collect the distinct neighbor label and vertices that didn't process yet.
                        __m512i distinct_lw;
                        /// It will find out the distinct label.
                        distinct_lw = _mm512_mask_compress_epi32(
                            set0, _mm512_kand(mask, new_labels_mask), lw_vec);
                        /// Count the set bit from the mask for neighbor labels
                        int neigh_lw_cnt =
                            _mm_popcnt_u32((unsigned)_mm512_kand(mask, new_labels_mask));
                        /// Store distinct neighbor community
                        _mm512_storeu_si512(&pnt_uniqueLabels[_cnt], distinct_lw);
                        /// Increment neighbor labels count
                        _cnt += neigh_lw_cnt;

                        /// Assign 0.0 in the label weight that contains -1.0 right now.
                        labelWeight_vec =
                            _mm512_mask_mov_ps(labelWeight_vec, new_labels_mask, fl_set0);
                        /// Add edge weight to the label weight and if mask doesn't set load from affinity
                        labelWeight_vec = _mm512_mask_add_ps(labelWeight_vec, mask, labelWeight_vec,
                                                             default_edge_weight);
//                        _mm512_mask_prefetch_i32scatter_ps(&pnt_labelWeights[0], mask, lw_vec, 4, _MM_HINT_T0);
                        /// Scatter label weight value to the label weight pointer.
                        _mm512_mask_i32scatter_ps(&pnt_labelWeights[0], mask, lw_vec,
                                                  labelWeight_vec, 4);

                        /// Count the set bit from the mask for ignore vertices
                        __mmask16 conflict_lw_mask = _mm512_knot(mask);
                        int vertex_cnt = _mm_popcnt_u32((unsigned)conflict_lw_mask);

                        if (vertex_cnt > 0) {
                            lw_vec = _mm512_mask_compress_epi32(set0, conflict_lw_mask, lw_vec);
                            index *remaining_lw = (index *)&lw_vec;
                            for (int j = 0; j < vertex_cnt; ++j) {
                                pnt_labelWeights[remaining_lw[j]] += fdefaultEdgeWeight;
                            }
                        }
                    }
                    pnt_outEdges = &outEdges[v][0];
                    for (int i = e; i < _deg; ++i) {
                        node w = pnt_outEdges[i];
                        label lw = data[w];
                        if (pnt_labelWeights[lw] == -1) {
                            pnt_labelWeights[lw] = 0;
                            pnt_uniqueLabels[_cnt++] = lw;
                        }
                        pnt_labelWeights[lw] += fdefaultEdgeWeight;
                    }

                    // get heaviest label
                    label heaviest = -1;
                    f_weight _heavyWeight = -1, max_weight = 0;
                    label lv = data[v];
//#pragma unroll
                    /*for (e = 0; (e + 16) <= _cnt; e += 16) {
                        /// Load at most 16 neighbor label.
                        __m512i lw_vec = _mm512_loadu_si512((__m512i *)&pnt_uniqueLabels[e]);
                        /// Gather label weight of the corresponding label.
                        __m512 labelWeight_vec =
                            _mm512_i32gather_ps(lw_vec, &pnt_labelWeights[0], 4);
                        max_weight = _mm512_reduce_max_ps(labelWeight_vec);
                        if (max_weight >= _heavyWeight) {
                            __m512 max_weight_vec = _mm512_set1_ps(max_weight);
                            __mmask16 gain_mask =
                                _mm512_cmpeq_ps_mask(labelWeight_vec, max_weight_vec);
                            _heavyWeight = max_weight;
                            heaviest = _mm512_mask_reduce_max_epi32(gain_mask, lw_vec);
                        }
                    }*/
                    for (int i = 0; i < _cnt; ++i) {
                        label lw = pnt_uniqueLabels[i];
                        if ((pnt_labelWeights[lw] > _heavyWeight)
                            || ((pnt_labelWeights[lw] == _heavyWeight) && (heaviest > lw))) {
                            heaviest = lw;
                            _heavyWeight = pnt_labelWeights[lw];
                        }
                    }
                    if (heaviest != -1 && lv != heaviest) { // UPDATE
                        data[v] = heaviest;                 // result[v] = heaviest;
                        nUpdated += 1;                      // TODO: atomic update?
                                                            //#pragma unroll
                                                            /* for (e=0; (e+16) <= _deg; e+= 16) {
                                                                 __m512i u_vec = _mm512_loadu_si512((__m512i *) &pnt_outEdges[e]);
                                                                 /// Scatter label weight value to the label weight pointer.
                                                                 _mm512_i32scatter_epi32(&activeNodes[0], u_vec, set_plus_1, 4);
                                                             }*/
#pragma omp simd
                        for (int i = 0; i < _deg; ++i) {
                            node u = pnt_outEdges[i];
                            activeNodes[u] = 1;
                        }
                    } else {
                        activeNodes[v] = 0;
                    }

                } else {
                    // node is isolated
                }
            }
        } else {
#pragma omp parallel for schedule(guided)
            for (omp_index v = 0; v < static_cast<omp_index>(z); ++v){
                if (G->hasNode(v) && (activeNodes[v]) && (G->degree(v) > 0)) {
                    index tid = omp_get_thread_num();
//#pragma omp simd
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
                        if ((labelWeights[tid][lw] > _heavyWeight) || ((labelWeights[tid][lw] == _heavyWeight) && (heaviest > lw))) {
                            heaviest = lw;
                            _heavyWeight = labelWeights[tid][lw];
                        }
                    }
                    if (heaviest != -1 && lv != heaviest) { // UPDATE
                        data[v] = heaviest; //result[v] = heaviest;
                        nUpdated += 1; // TODO: atomic update?
//#pragma omp simd
                        for (int i = 0; i < outEdges[v].size(); ++i) {
                            node u = outEdges[v][i];
                            activeNodes[u] = 1;
                        }
                    } else {
                        activeNodes[v] = 0;
                    }

                } else {
                    // node is isolated
                }
            }
        }

        std::cout<< "done label propagation" << std::endl;
        // for each while loop iteration...

        runtime.stop();
        this->timing.push_back(runtime.elapsedMilliseconds());
        DEBUG("[DONE] LabelPropagation: iteration #", nIterations, " - updated ", nUpdated,
              " labels, time spent: ", runtime.elapsedTag());
        //        std::cout<< "[DONE] LabelPropagation: iteration #" << nIterations  << " - updated
        //        "<< nUpdated << " labels, time spent: " << runtime.elapsedTag() << std::endl;

    } // end while
    for (index i = 0; i < z; ++i) {
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
