/*
 * ONPL.cpp
 *
 *  Created on: 10.10.2018
 *      Author: Md Maruf Hossain
 */

#include <omp.h>
#include <networkit/community/ONPL.hpp>
#include <networkit/coarsening/ParallelPartitionCoarsening.hpp>
#include <networkit/coarsening/ClusteringProjector.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/community/Modularity.hpp>
#include <networkit/auxiliary/PowerCalculator.hpp>
#include "fvec.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <fstream>
#include <limits>
#include <mmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <cmath>
#include<time.h>
#include <sys/time.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

//#ifndef L1D_CACHE_MISS_COUNT
//#define L1D_CACHE_MISS_COUNT
//#endif

#define ONE_THREAD false
#define CONFLICT_DETECT false
#define POWER_LOG false

char event_names[MAX_PACKAGES][NUM_RAPL_DOMAINS][256];
char filenames[MAX_PACKAGES][NUM_RAPL_DOMAINS][256];
char basenames[MAX_PACKAGES][256];
char tempfile[256];
long long before[MAX_PACKAGES][NUM_RAPL_DOMAINS];
long long after[MAX_PACKAGES][NUM_RAPL_DOMAINS];
int valid[MAX_PACKAGES][NUM_RAPL_DOMAINS];

char rapl_domain_names[NUM_RAPL_DOMAINS][30];

FILE *rapl_sysf_f;

typedef int32_t sint;

int m_iter = 0;


namespace NetworKit {

ONPL::ONPL(const Graph& G, bool refine, f_weight gamma, std::string par, count maxIter, bool fullVec, bool turbo, bool recurse) : CommunityDetectionAlgorithm(G), parallelism(par), refine(refine), gamma(gamma), maxIter(maxIter), fullVec(fullVec), turbo(turbo), recurse(recurse) {

}

ONPL::ONPL(const Graph& G, const ONPL& other) : CommunityDetectionAlgorithm(G), parallelism(other.parallelism), refine(other.refine), gamma(other.gamma), maxIter(other.maxIter), turbo(other.turbo), recurse(other.recurse) {

}

void ONPL::initONPL(){
    m_iter = 0;
}
std::ofstream f_conflict_log;
void ONPL::setupCSVFile(std::string conflict_file) {
#if CONFLICT_DETECT
    std::ifstream infile(conflict_file);
    bool existing_file = infile.good();
    f_conflict_log.open(conflict_file, std::ios_base::out | std::ios_base::app | std::ios_base::ate);
    if (!existing_file) {
        f_conflict_log << "Move Phase" << "," << "Move" << "," << "Vertices" << "," << "Edges" << ","
                       << "Total Conflict" << "," << "Move Time" << std::endl;
    }
#endif
}
void ONPL::setupPowerFile(std::string graphName, count threads) {
#if POWER_LOG
    _graphName = graphName;
    _threads = threads;
#endif
}

void ONPL::run() {
    Aux::SignalHandler handler;
    Modularity modularity;
//        DEBUG("calling run method on " , G->toString());
#if POWER_LOG
    //        if(m_iter == 0)
    rapl_sysfs_init();
#endif
#if CONFLICT_DETECT
    uint64_t conflict_community = 0, reiterate_conflict = 0;
#endif
#if ONE_THREAD
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    count z = G->upperNodeIdBound();
    //        count move_count, tot_move_count = 0;
    size_t alignment = 64;
    index max_deg_of_graph = 0;
    double affinity_time = 0.0;
    double clustering_time = 0.0;

    // init communities to singletons
    Partition zeta(z);
    zeta.allToSingletons();
    std::vector<index> _dataV = zeta.getVector();
    index *_data;
    index o = zeta.upperBound();
    posix_memalign((void **) &_data, alignment, o * sizeof(index));
    for (int i = 0; i < o; ++i) {
        _data[i] = _dataV[i];
    }

    // init graph-dependent temporaries
    std::vector<f_weight> volNode(z, 0.0);
    // $\omega(E)$
    f_weight total = G->totalEdgeWeight();
    //        DEBUG("total edge weight: " , total);
    f_weight divisor = (2 * total * total); // needed in modularity calculation

    //        std::vector<std::vector<index > > community_counter = std::vector<std::vector<index > >(33,std::vector<index>(16, 0));
    //        count ittr_counter = 0;
    count *outDegree;
    posix_memalign((void **) &outDegree, alignment, z * sizeof(count));
    const std::vector<f_weight> *outEdgeWeights;
    const std::vector<node> *outEdges;
    bool isGraphWeighted = G->isWeighted();
    __m512 default_edge_weight;
    f_weight f_defaultEdgeWeight;
    if(!isGraphWeighted){
        default_edge_weight = _mm512_set1_ps(fdefaultEdgeWeight);
        f_defaultEdgeWeight = fdefaultEdgeWeight;
    }
    /// Calculate affinity. 512 register, so it can load 16, 32 bit integer or floating point.
    /// 512 bit double register initialize by all 0.0
    const   __m512d db_set0 = _mm512_set1_pd(0.0);
    /// 512 bit floating register initialize by all 0.0
    const   __m512 fl_set0 = _mm512_set1_ps(0.0);
    /// 512 bit floating register initialize by all -1.0
    const   __m512 fl_set1 = _mm512_set1_ps(-1.0);
    /// 512 bit integer register initialize by all 0
    const   __m512i set0 = _mm512_set1_epi32(0x00000000);
    /// 512 bit integer register initialize by all -1
    const   __m512i set1 = _mm512_set1_epi32(0xFFFFFFFF);
    /// 512 bit integer register initialize by all 1
    const   __m512i set_plus_1 = _mm512_set1_epi32(1);
    /// 512 bit integer register initialize by all -1
    const __m512i set_minus_1 = _mm512_set1_epi32(-1);

    const   __m512 total_vec = _mm512_set1_ps(total);

    outEdgeWeights = G->getOutEdgeWeights();
    outEdges = G->getOutEdges();
    for (int i = 0; i < z; ++i) {
        outDegree[i] = G->degree(i);
    }
    index max_tid = omp_get_max_threads();
    index max_deg_arr[max_tid];

#pragma omp parallel
    {
        index tid = omp_get_thread_num();
        max_deg_arr[tid] = max_deg_of_graph;
#pragma omp for schedule(dynamic)
        for (index u = 0; u < z; ++u) {
            volNode[u] += G->weightedDegree(u);
            volNode[u] += G->weight(u, u); // consider self-loop twice
            if (max_deg_arr[tid] < outDegree[u]) {
                max_deg_arr[tid] = outDegree[u];
            }
        }
#pragma omp master
        {
            for (index i = 0; i < max_tid; i++)
                if (max_deg_of_graph < max_deg_arr[i]) {
                    max_deg_of_graph = max_deg_arr[i];
                }
        }
    }

    //        std::cout<<"max_deg_of_graph: "<<max_deg_of_graph<<std::endl;

    // init community-dependent temporaries
    std::vector<f_weight> volCommunity(o, 0.0);
    zeta.parallelForEntries([&](node u, index C) { 	// set volume for all communities
        if (C != none)
            volCommunity[C] = volNode[u];
    });

    // first move phase
    bool moved = false; // indicates whether any node has been moved in the last pass
    bool change = false; // indicates whether the communities have changed at all

    // stores the affinity for each neighboring community (index), one vector per thread
    f_weight** turboAffinity = (f_weight **) malloc(max_tid * sizeof(f_weight *));
    // stores the list of neighboring communities, one vector per thread
    index** neigh_comm = (index **) malloc(max_tid * sizeof(index *));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < max_tid; ++i) {
        posix_memalign((void **) &turboAffinity[i], alignment, zeta.upperBound() * sizeof(f_weight));
        posix_memalign((void **) &neigh_comm[i], alignment, max_deg_of_graph * sizeof(index));
        //            turboAffinity[i] = (f_weight *) malloc(zeta.upperBound() * sizeof(f_weight));
        //            neigh_comm[i] = (index *) malloc(max_deg_of_graph * sizeof(index));
    }

    /********/
    auto moveToNewCommunity = [&](node u) {

        index tid = omp_get_thread_num();
        count neigh_counter = 0;
        index _deg = outDegree[u];
        /// Pointer for neighbor vertices. We can access using edge index.
        const node *pnt_outEdges = &outEdges[u][0];
        /// Pointer for neighbor edge weight. We can access using edge index.
        const f_weight *pnt_outEdgeWeight = &outEdgeWeights[u][0];
        index *pnt_neigh_comm = &neigh_comm[tid][0];

        /// pointer to calculate the affinity of neighbor community
        f_weight *pnt_affinity = &turboAffinity[tid][0];
        /// Initialize affinity.
        index i = 0;
        for (i = 0; (i+16) <= _deg; i += 16) {
            __m512i v_vec = _mm512_loadu_si512((__m512i * ) & pnt_outEdges[i]);
            __m512i C_vec = _mm512_i32gather_epi32(v_vec, &_data[0], 4);
            _mm512_i32scatter_ps(&pnt_affinity[0], C_vec, fl_set1, 4);
        }
        for (index edge = i; edge < _deg; ++edge) {
            pnt_affinity[_data[pnt_outEdges[edge]]] = -1.0;
        }
        pnt_affinity[_data[u]] = 0;
        /// protect u != v condition
        const   __m512i check_self_loop = _mm512_set1_epi32(u);
#pragma unroll
        for (i = 0; (i+16) <= _deg; i += 16) {
            //                __mmask16 mask_neighbor_exist = pow(2, ((neighbor_processed-i) >= 16 ? 16 : (neighbor_processed - i))) - 1;
            /// Load at most 16 neighbor vertices.
            __m512i v_vec = _mm512_loadu_si512((__m512i *) &pnt_outEdges[i]);
            //                __m512i v_vec = _mm512_mask_loadu_epi32(set0, mask_neighbor_exist, (__m512i *) &pnt_outEdges[i]);
            /// Load at most 16 neighbor vertex edge weight.
            __m512 w_vec = _mm512_loadu_ps((__m512 *) &pnt_outEdgeWeight[i]);
            /// Mask to find u != v
            __mmask16 self_loop_mask = _mm512_cmpneq_epi32_mask(check_self_loop, v_vec);
            /// Gather community of the neighbor vertices.
            __m512i C_vec = _mm512_mask_i32gather_epi32(set0, self_loop_mask, v_vec, &_data[0], 4);
            sint vertex_cnt = _mm_popcnt_u32((unsigned)self_loop_mask);
            //                #if CONFLICT_DETECT
            //                    reiterate_conflict += vertex_cnt;
            //                    conflict_community += _mm_popcnt_u32((unsigned) _mm512_knot(_mm512_kand(mask, new_comm_mask)));
            //                #endif
            /// It will calculate the ignorance vertex edge weight in the previous calculation.
            w_vec = _mm512_mask_compress_ps(fl_set0, self_loop_mask, w_vec);
            /// It will find out the community that is not processed.
            C_vec = _mm512_mask_compress_epi32(set0, self_loop_mask, C_vec);
            __mmask16 comm_mask = pow(2, vertex_cnt) - 1;
            index * comm_not_processed = (index *)&C_vec;
            __m512i first_comm = _mm512_set1_epi32(comm_not_processed[0]);
            const __mmask16 first_comm_mask = _mm512_mask_cmpeq_epi32_mask(comm_mask, first_comm, C_vec);
            if(pnt_affinity[comm_not_processed[0]] == -1){
                pnt_neigh_comm[neigh_counter++] = comm_not_processed[0];
                pnt_affinity[comm_not_processed[0]] = 0;
            }
            pnt_affinity[comm_not_processed[0]] += _mm512_mask_reduce_add_ps(first_comm_mask, w_vec);
            self_loop_mask  = _mm512_kandn(first_comm_mask, comm_mask);
            vertex_cnt = _mm_popcnt_u32((unsigned) self_loop_mask);

            if(vertex_cnt>0) {
                w_vec = _mm512_mask_compress_ps(fl_set0, self_loop_mask, w_vec);
                f_weight *weight_not_processed = (f_weight *) &w_vec;
                C_vec = _mm512_mask_compress_epi32(set0, self_loop_mask, C_vec);
                index *remaining_comm = (index *) &C_vec;
                for (int j = 0; j < vertex_cnt; ++j) {
                    index C = remaining_comm[j];
                    if (pnt_affinity[C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                        pnt_affinity[C] = 0;
                        pnt_neigh_comm[neigh_counter++] = C;
                    }
                    pnt_affinity[C] += weight_not_processed[j];
                }
            }
        }

        pnt_outEdges = &outEdges[u][0];
        pnt_outEdgeWeight = &outEdgeWeights[u][0];
        for (index j= i; j < _deg; ++j) {
            node v = pnt_outEdges[j];
            if (u != v) {
                index C = _data[v];
                if (pnt_affinity[C] == -1) {
                    /// found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                    pnt_affinity[C] = 0;
                    pnt_neigh_comm[neigh_counter++] = C;
                }
                pnt_affinity[C] += pnt_outEdgeWeight[j];
            }
        }

        /*****/
        index best = none;
        f_weight deltaBest = -1;

        index C = _data[u];
        f_weight affinityC = pnt_affinity[C];
        f_weight volN = volNode[u];
        f_weight volCommunityMinusNode_C = volCommunity[C] - volN;
        f_weight max_delta = 0;
        /// protect C != D condition
        const   __m512i reg_C = _mm512_set1_epi32(C);
        const   __m512 affinityC_vec = _mm512_set1_ps(affinityC);
        const   __m512 volCommunityC_vec = _mm512_set1_ps(volCommunityMinusNode_C);
        const   __m512 total_vec = _mm512_set1_ps(total);
        f_weight coefficient = (this->gamma * volN)/divisor;
        const   __m512 coefficient_vec = _mm512_set1_ps(coefficient);

        for (i=0; (i+16) <= neigh_counter; i += 16) {
            /// Load at most 16 neighbor community.
            __m512i D_vec = _mm512_loadu_si512((__m512i *) &pnt_neigh_comm[i]);
            /// Mask to find C != D
            const __mmask16 different_comm_mask = _mm512_cmpneq_epi32_mask(reg_C, D_vec);
            /// Gather affinity of the corresponding community.
            __m512 affinityD_vec = _mm512_i32gather_ps(D_vec, &pnt_affinity[0], 4);
            __m512 volCommunityD_vec = _mm512_i32gather_ps(D_vec, &volCommunity[0], 4);
            __m512 aff_diff = _mm512_mask_div_ps(fl_set0, different_comm_mask, _mm512_mask_sub_ps(fl_set0, different_comm_mask, affinityD_vec, affinityC_vec), total_vec);
            __m512 vol_diff = _mm512_mul_ps(_mm512_mask_sub_ps(fl_set0, different_comm_mask, volCommunityC_vec, volCommunityD_vec), coefficient_vec);
            __m512 delta_vec = _mm512_add_ps(aff_diff, vol_diff);

            max_delta = _mm512_mask_reduce_max_ps(different_comm_mask, delta_vec);
            if (max_delta > deltaBest){
                __m512 max_delta_vec = _mm512_set1_ps(max_delta);
                __mmask16 gain_mask = _mm512_mask_cmpeq_ps_mask(different_comm_mask, delta_vec, max_delta_vec);
                deltaBest = max_delta;
                best = _mm512_mask_reduce_min_epi32(gain_mask, D_vec);
            }

        }


        for (index j=i; j<neigh_counter; ++j) {
            index D = pnt_neigh_comm[j];
            if (D != C) { // consider only nodes in other clusters (and implicitly only nodes other than u)
                f_weight delta = (pnt_affinity[D] - affinityC) / total + this->gamma * ((volCommunityMinusNode_C - volCommunity[D]) * volN) / divisor;
                if (delta > deltaBest) {
                    deltaBest = delta;
                    best = D;
                }
            }
        }


        if (deltaBest > 0) { // if modularity improvement possible
            assert (best != C && best != none);// do not "move" to original cluster
                                               //#pragma omp atomic update
                                               //                move_count += 1;
            _data[u] = best; // move to best cluster
            // TRACE("node " , u , " moved");

            // mod update
            f_weight volN = 0.0;
            volN = volNode[u];
            // update the volume of the two clusters
#pragma omp atomic update
            volCommunity[C] -= volN;
#pragma omp atomic update
            volCommunity[best] += volN;

            moved = true; // change to clustering has been made

        }
        /*****/


    };

    auto tryVecMove = [&](node u) {
        index tid = omp_get_thread_num();
        count neigh_counter = 0;
        index _deg = outDegree[u];
        /// Pointer for neighbor vertices. We can access using edge index.
        const node *pnt_outEdges = &outEdges[u][0];
        /// Pointer for neighbor edge weight. We can access using edge index.
        const f_weight *pnt_outEdgeWeight = &outEdgeWeights[u][0];
        index *pnt_neigh_comm = &neigh_comm[tid][0];

        /// pointer to calculate the affinity of neighbor community
        f_weight *pnt_affinity = &turboAffinity[tid][0];
        index i = 0;
        /// Initialize affinity.
        for (i = 0; (i+16) <= _deg; i += 16) {
            __m512i v_vec = _mm512_loadu_si512((__m512i *) &pnt_outEdges[i]);
            __m512i C_vec = _mm512_i32gather_epi32(v_vec, &_data[0], 4);
            _mm512_i32scatter_ps(&pnt_affinity[0], C_vec, fl_set1, 4);
        }
        for (index edge= i; edge < _deg; ++edge) {
            pnt_affinity[_data[pnt_outEdges[edge]]] = -1.0;
        }
        pnt_affinity[_data[u]] = 0;
        /// protect u != v condition
        const   __m512i check_self_loop = _mm512_set1_epi32(u);
#pragma unroll
        for (i = 0; (i+16) <= _deg; i += 16) {
            /// Load at most 16 neighbor vertices.
            __m512i v_vec = _mm512_loadu_si512((__m512i *) &pnt_outEdges[i]);
            /// Load at most 16 neighbor vertex edge weight.
            __m512 w_vec = _mm512_loadu_ps((__m512 *) &pnt_outEdgeWeight[i]);
            /// Mask to find u != v
            const __mmask16 self_loop_mask = _mm512_cmpneq_epi32_mask(check_self_loop, v_vec);
            /// Gather community of the neighbor vertices.
            __m512i C_vec = _mm512_mask_i32gather_epi32(set0, self_loop_mask, v_vec, &_data[0], 4);
            /// Gather affinity of the corresponding community.
            __m512 affinity_vec = _mm512_mask_i32gather_ps(fl_set0, self_loop_mask, C_vec, &pnt_affinity[0], 4);

            /// Mask to find out the new community that contains -1.0 value
            const __mmask16 new_comm_mask = _mm512_mask_cmpeq_ps_mask(self_loop_mask, fl_set1, affinity_vec);
            /// Detect conflict of the community
            __m512i C_conflict = _mm512_mask_conflict_epi32(set_plus_1, self_loop_mask, C_vec);
            /// Calculate mask using NAND of C_conflict and set1
            const __mmask16 mask = _mm512_mask_cmpeq_epi32_mask(self_loop_mask, C_conflict, set0);
            /// Now we need to collect the distinct neighbor community and vertices that didn't process yet.
            __m512i distinct_comm;
            /// It will find out the distinct community.
            distinct_comm = _mm512_mask_compress_epi32(set0, _mm512_kand(mask, new_comm_mask), C_vec);
            /// Count the set bit from the mask for neighbor community
            sint neigh_cnt = _mm_popcnt_u32((unsigned) _mm512_kand(mask, new_comm_mask));
            /// Store distinct neighbor community
            _mm512_storeu_si512(&pnt_neigh_comm[neigh_counter], distinct_comm);
            /// Increment neighbor community count
            neigh_counter += neigh_cnt;

            /// Assign 0.0 in the affinity that contains -1.0 right now.
            affinity_vec = _mm512_mask_mov_ps(affinity_vec, new_comm_mask, fl_set0);
            /// Add edge weight to the affinity and if mask doesn't set load from affinity
            affinity_vec = _mm512_mask_add_ps(affinity_vec, mask, affinity_vec, w_vec);
            /// Scatter affinity value to the affinity pointer.
            _mm512_mask_i32scatter_ps(&pnt_affinity[0], mask, C_vec, affinity_vec, 4);

            /// Count the set bit from the mask for ignore vertices
            __mmask16 conflict_comm_mask = _mm512_kand(_mm512_knot(mask), self_loop_mask);
            sint vertex_cnt = _mm_popcnt_u32((unsigned)conflict_comm_mask);

            if(vertex_cnt>0) {
                w_vec = _mm512_mask_compress_ps(fl_set0, self_loop_mask, w_vec);
                f_weight *weight_not_processed = (f_weight *) &w_vec;
                C_vec = _mm512_mask_compress_epi32(set0, conflict_comm_mask, C_vec);
                index *remaining_comm = (index *) &C_vec;
                for (int j = 0; j < vertex_cnt; ++j) {
                    pnt_affinity[remaining_comm[j]] += weight_not_processed[j];
                }
            }
        }
        pnt_outEdges = &outEdges[u][0];
        pnt_outEdgeWeight = &outEdgeWeights[u][0];
        for (index j= i; j < _deg; ++j) {
            node v = pnt_outEdges[j];
            if (u != v) {
                index C = _data[v];
                if (pnt_affinity[C] == -1) {
                    /// found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                    pnt_affinity[C] = 0;
                    pnt_neigh_comm[neigh_counter++] = C;
                }
                pnt_affinity[C] += pnt_outEdgeWeight[j];
            }
        }
        /*****/
        index best = none;
        f_weight deltaBest = -1;
        index C = _data[u];
        f_weight affinityC = pnt_affinity[C];
        f_weight volN = volNode[u];
        f_weight volCommunityMinusNode_C = volCommunity[C] - volN;
        f_weight max_delta = 0;
        /// protect C != D condition
        const   __m512i reg_C = _mm512_set1_epi32(C);
        const   __m512 affinityC_vec = _mm512_set1_ps(affinityC);
        const   __m512 volCommunityC_vec = _mm512_set1_ps(volCommunityMinusNode_C);
        const   __m512 total_vec = _mm512_set1_ps(total);
        f_weight coefficient = (this->gamma * volN)/divisor;
        const   __m512 coefficient_vec = _mm512_set1_ps(coefficient);

        for (i=0; (i+16) <= neigh_counter; i += 16) {
            /// Load at most 16 neighbor community.
            __m512i D_vec = _mm512_loadu_si512((__m512i *) &pnt_neigh_comm[i]);
            /// Mask to find C != D
            const __mmask16 different_comm_mask = _mm512_cmpneq_epi32_mask(reg_C, D_vec);
            /// Gather affinity of the corresponding community.
            __m512 affinityD_vec = _mm512_i32gather_ps(D_vec, &pnt_affinity[0], 4);
            __m512 volCommunityD_vec = _mm512_i32gather_ps(D_vec, &volCommunity[0], 4);
            __m512 aff_diff = _mm512_mask_div_ps(fl_set0, different_comm_mask, _mm512_mask_sub_ps(fl_set0, different_comm_mask, affinityD_vec, affinityC_vec), total_vec);
            __m512 vol_diff = _mm512_mul_ps(_mm512_mask_sub_ps(fl_set0, different_comm_mask, volCommunityC_vec, volCommunityD_vec), coefficient_vec);
            __m512 delta_vec = _mm512_add_ps(aff_diff, vol_diff);

            max_delta = _mm512_mask_reduce_max_ps(different_comm_mask, delta_vec);
            if (max_delta > deltaBest){
                __m512 max_delta_vec = _mm512_set1_ps(max_delta);
                __mmask16 gain_mask = _mm512_mask_cmpeq_ps_mask(different_comm_mask, delta_vec, max_delta_vec);
                deltaBest = max_delta;
                best = _mm512_mask_reduce_min_epi32(gain_mask, D_vec);
            }

        }
        for (index j=i; j<neigh_counter; ++j) {
            index D = pnt_neigh_comm[j];
            if (D != C) { // consider only nodes in other clusters (and implicitly only nodes other than u)
                f_weight delta = (pnt_affinity[D] - affinityC) / total + this->gamma * ((volCommunityMinusNode_C - volCommunity[D]) * volN) / divisor;
                if (delta > deltaBest) {
                    deltaBest = delta;
                    best = D;
                }
            }
        }

        if (deltaBest > 0) { // if modularity improvement possible
            assert (best != C && best != none);// do not "move" to original cluster
            _data[u] = best; // move to best cluster
            // mod update
            f_weight volN = 0.0;
            volN = volNode[u];
            // update the volume of the two clusters
#pragma omp atomic update
            volCommunity[C] -= volN;
#pragma omp atomic update
            volCommunity[best] += volN;
            moved = true; // change to clustering has been made
        }
        /*****/
    };
    auto moveToNewCommunityWithDefaultWeight = [&](node u) {
        index tid = omp_get_thread_num();
        count neigh_counter = 0;
        index _deg = outDegree[u];
        /// Pointer for neighbor vertices. We can access using edge index.
        const node *pnt_outEdges = &outEdges[u][0];
        /// Pointer for neighbor edge weight. We can access using edge index.
        index *pnt_neigh_comm = &neigh_comm[tid][0];

        /// pointer to calculate the affinity of neighbor community
        f_weight *pnt_affinity = &turboAffinity[tid][0];
        /// Initialize affinity with zero. May be we can use intel intrinsic to do that.
        index i=0;
        for (i = 0; (i+16) <= _deg; i += 16) {
            __m512i v_vec = _mm512_loadu_si512((__m512i *) &pnt_outEdges[i]);
            __m512i C_vec = _mm512_i32gather_epi32(v_vec, &_data[0], 4);
            _mm512_i32scatter_ps(&pnt_affinity[0], C_vec, fl_set1, 4);
        }
        for (index edge= i; edge < _deg; ++edge) {
            pnt_affinity[_data[pnt_outEdges[edge]]] = -1.0;
        }
        pnt_affinity[_data[u]] = 0;
        /// protect u != v condition
        const   __m512i check_self_loop = _mm512_set1_epi32(u);
#pragma unroll
        for (i = 0; (i+16) <= _deg; i += 16) {
            //                __mmask16 mask_neighbor_exist = pow(2, ((neighbor_processed-i) >= 16 ? 16 : (neighbor_processed - i))) - 1;
            /// Load at most 16 neighbor vertices.
            __m512i v_vec = _mm512_loadu_si512((__m512i *) &pnt_outEdges[i]);
            /// Mask to find u != v
            const __mmask16 self_loop_mask = _mm512_cmpneq_epi32_mask(check_self_loop, v_vec);
            /// Gather community of the neighbor vertices.
            __m512i C_vec = _mm512_mask_i32gather_epi32(set0, self_loop_mask, v_vec, &_data[0], 4);
            /// Gather affinity of the corresponding community.
            __m512 affinity_vec = _mm512_mask_i32gather_ps(fl_set0, self_loop_mask, C_vec, &pnt_affinity[0], 4);

            /// Mask to find out the new community that contains -1.0 value
            const __mmask16 new_comm_mask = _mm512_mask_cmpeq_ps_mask(self_loop_mask, fl_set1, affinity_vec);
            //                const __mmask16 new_comm_mask = _mm512_kand(_mm512_cmpeq_ps_mask(fl_set1, affinity_vec), self_loop_mask);
            /// Detect conflict of the community
            __m512i C_conflict = _mm512_mask_conflict_epi32(set_plus_1, self_loop_mask, C_vec);
            /// Calculate mask using NAND of C_conflict and set1
            const __mmask16 mask = _mm512_mask_cmpeq_epi32_mask(self_loop_mask, C_conflict, set0);
            //                const __mmask16 mask = _mm512_kand(_mm512_testn_epi32_mask(C_conflict, set1), self_loop_mask);
            //                community_counter[ittr_counter][_mm_popcnt_u32((unsigned) mask)-1]++;
            /// Now we need to collect the distinct neighbor community and vertices that didn't process yet.
            __m512i distinct_comm;
            /// It will find out the distinct community.
            distinct_comm = _mm512_mask_compress_epi32(set0, _mm512_kand(mask, new_comm_mask), C_vec);
            /// Count the set bit from the mask for neighbor community
            sint neigh_cnt = _mm_popcnt_u32((unsigned) _mm512_kand(mask, new_comm_mask));
            /// Store distinct neighbor community
            //                _mm512_mask_storeu_epi32(&pnt_neigh_comm[neigh_counter], _mm512_kand(mask, new_comm_mask), distinct_comm);
            _mm512_storeu_si512(&pnt_neigh_comm[neigh_counter], distinct_comm);
            /// Increment neighbor community count
            neigh_counter += neigh_cnt;

            /// Assign 0.0 in the affinity that contains -1.0 right now.
            affinity_vec = _mm512_mask_mov_ps(affinity_vec, new_comm_mask, fl_set0);
            /// Add edge weight to the affinity and if mask doesn't set load from affinity
            affinity_vec = _mm512_mask_add_ps(affinity_vec, mask, affinity_vec, default_edge_weight);
            /// Scatter affinity value to the affinity pointer.
            _mm512_mask_i32scatter_ps(&pnt_affinity[0], mask, C_vec, affinity_vec, 4);


            /// Count the set bit from the mask for ignore vertices
            __mmask16 conflict_comm_mask = _mm512_kand(_mm512_knot(mask), self_loop_mask);
            sint vertex_cnt = _mm_popcnt_u32((unsigned)conflict_comm_mask);

            if(vertex_cnt>0) {
                __m512i tmp_C = C_vec;
                C_vec = _mm512_mask_compress_epi32(set0, conflict_comm_mask, C_vec);
                index *remaining_comm = (index *) &C_vec;
                for (int j = 0; j < vertex_cnt; ++j) {
                    pnt_affinity[remaining_comm[j]] += f_defaultEdgeWeight;
                }
            }
        }

        pnt_outEdges = &outEdges[u][0];
        for (index j= i; j < _deg; ++j) {
            node v = pnt_outEdges[j];
            if (u != v) {
                index C = _data[v];
                if (pnt_affinity[C] == -1) {
                    /// found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                    pnt_affinity[C] = 0;
                    pnt_neigh_comm[neigh_counter++] = C;
                }
                pnt_affinity[C] += f_defaultEdgeWeight;
            }
        }


        index best = none;
        f_weight deltaBest = -1;

        index C = _data[u];
        f_weight affinityC = pnt_affinity[C];
        f_weight volN = volNode[u];
        f_weight volCommunityMinusNode_C = volCommunity[C] - volN;
        f_weight max_delta = 0;
        /// protect C != D condition
        const   __m512i reg_C = _mm512_set1_epi32(C);
        const   __m512 affinityC_vec = _mm512_set1_ps(affinityC);
        const   __m512 volCommunityC_vec = _mm512_set1_ps(volCommunityMinusNode_C);
        const   __m512 total_vec = _mm512_set1_ps(total);
        f_weight coefficient = (this->gamma * volN)/divisor;
        const   __m512 coefficient_vec = _mm512_set1_ps(coefficient);
        for (i=0; (i+16) <= neigh_counter; i += 16) {
            /// Load at most 16 neighbor community.
            __m512i D_vec = _mm512_loadu_si512((__m512i *) &pnt_neigh_comm[i]);
            /// Mask to find C != D
            const __mmask16 different_comm_mask = _mm512_cmpneq_epi32_mask(reg_C, D_vec);
            /// Gather affinity of the corresponding community.
            __m512 affinityD_vec = _mm512_i32gather_ps(D_vec, &pnt_affinity[0], 4);
            __m512 volCommunityD_vec = _mm512_i32gather_ps(D_vec, &volCommunity[0], 4);
            __m512 aff_diff = _mm512_mask_div_ps(fl_set0, different_comm_mask, _mm512_mask_sub_ps(fl_set0, different_comm_mask, affinityD_vec, affinityC_vec), total_vec);
            __m512 vol_diff = _mm512_mul_ps(_mm512_mask_sub_ps(fl_set0, different_comm_mask, volCommunityC_vec, volCommunityD_vec), coefficient_vec);
            __m512 delta_vec = _mm512_add_ps(aff_diff, vol_diff);

            max_delta = _mm512_mask_reduce_max_ps(different_comm_mask, delta_vec);
            if (max_delta > deltaBest){
                __m512 max_delta_vec = _mm512_set1_ps(max_delta);
                __mmask16 gain_mask = _mm512_mask_cmpeq_ps_mask(different_comm_mask, delta_vec, max_delta_vec);
                deltaBest = max_delta;
                best = _mm512_mask_reduce_min_epi32(gain_mask, D_vec);
            }

        }

        for (index j=i; j<neigh_counter; ++j) {
            index D = pnt_neigh_comm[j];
            if (D != C) { // consider only nodes in other clusters (and implicitly only nodes other than u)
                f_weight delta = (pnt_affinity[D] - affinityC) / total + this->gamma * ((volCommunityMinusNode_C - volCommunity[D]) * volN) / divisor;
                if (delta > deltaBest) {
                    deltaBest = delta;
                    best = D;
                }
            }
        }


        if (deltaBest > 0) { // if modularity improvement possible
            assert (best != C && best != none);// do not "move" to original cluster
                                               //#pragma omp atomic update
                                               //                move_count += 1;
            _data[u] = best; // move to best cluster
            // TRACE("node " , u , " moved");

            // mod update
            f_weight volN = 0.0;
            volN = volNode[u];
            // update the volume of the two clusters
#pragma omp atomic update
            volCommunity[C] -= volN;
#pragma omp atomic update
            volCommunity[best] += volN;

            moved = true; // change to clustering has been made

        }
    };


    auto reducedTryMove = [&](node u) {
        index tid = omp_get_thread_num();
        count neigh_counter = 0;
        index _deg = outDegree[u];
        /// Pointer for neighbor vertices. We can access using edge index.
        const node *pnt_outEdges = &outEdges[u][0];
        /// Pointer for neighbor edge weight. We can access using edge index.
        index *pnt_neigh_comm = &neigh_comm[tid][0];

        /// pointer to calculate the affinity of neighbor community
        f_weight *pnt_affinity = &turboAffinity[tid][0];
        /// Initialize affinity with zero. May be we can use intel intrinsic to do that.
#pragma omp simd
        for(index edge=0; edge<_deg; ++edge){
            pnt_affinity[_data[pnt_outEdges[edge]]] = -1.0;
        }
        pnt_affinity[_data[u]] = 0;
        index i=0;

        /// protect u != v condition
        const   __m512i check_self_loop = _mm512_set1_epi32(u);
#pragma unroll
        for (i = 0; (i+16) <= _deg; i += 16) {
            /// Load at most 16 neighbor vertices.
            __m512i v_vec = _mm512_loadu_si512((__m512i *) &pnt_outEdges[i]);
            /// Mask to find u != v
            __mmask16 self_loop_mask = _mm512_cmpneq_epi32_mask(check_self_loop, v_vec);
            /// Gather community of the neighbor vertices.
            __m512i C_vec = _mm512_mask_i32gather_epi32(set0, self_loop_mask, v_vec, &_data[0], 4);
            sint vertex_cnt = _mm_popcnt_u32((unsigned)self_loop_mask);

            /// It will find out the community that is not processed.
            C_vec = _mm512_mask_compress_epi32(set0, self_loop_mask, C_vec);
            __mmask16 comm_mask = pow(2, vertex_cnt) - 1;
            index * comm_not_processed = (index *)&C_vec;
            __m512i first_comm = _mm512_set1_epi32(comm_not_processed[0]);
            const __mmask16 first_comm_mask = _mm512_mask_cmpeq_epi32_mask(comm_mask, first_comm, C_vec);
            if(pnt_affinity[comm_not_processed[0]] == -1){
                pnt_neigh_comm[neigh_counter++] = comm_not_processed[0];
                pnt_affinity[comm_not_processed[0]] = 0;
            }
            pnt_affinity[comm_not_processed[0]] += f_defaultEdgeWeight * _mm_popcnt_u32((unsigned)first_comm_mask);
            self_loop_mask  = _mm512_kandn(first_comm_mask, comm_mask);
            vertex_cnt = _mm_popcnt_u32((unsigned) self_loop_mask);

            if(vertex_cnt>0) {
                C_vec = _mm512_mask_compress_epi32(set0, self_loop_mask, C_vec);
                index *remaining_comm = (index *) &C_vec;
                for (int j = 0; j < vertex_cnt; ++j) {
                    index C = remaining_comm[j];
                    if (pnt_affinity[C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                        pnt_affinity[C] = 0;
                        pnt_neigh_comm[neigh_counter++] = C;
                    }
                    pnt_affinity[C] += f_defaultEdgeWeight;
                }
            }
        }

        pnt_outEdges = &outEdges[u][0];
        for (index j= i; j < _deg; ++j) {
            node v = pnt_outEdges[j];
            if (u != v) {
                index C = _data[v];
                if (pnt_affinity[C] == -1) {
                    /// found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                    pnt_affinity[C] = 0;
                    pnt_neigh_comm[neigh_counter++] = C;
                }
                pnt_affinity[C] += f_defaultEdgeWeight;
            }
        }

        index best = none;
        f_weight deltaBest = -1;

        index C = _data[u];
        f_weight affinityC = pnt_affinity[C];
        f_weight volN = volNode[u];
        f_weight volCommunityMinusNode_C = volCommunity[C] - volN;

        for (index i=0; i<neigh_counter; ++i) {
            index D = pnt_neigh_comm[i];
            if (D != C) { // consider only nodes in other clusters (and implicitly only nodes other than u)
                f_weight delta = (pnt_affinity[D] - affinityC) / total + this->gamma * ((volCommunityMinusNode_C - volCommunity[D]) * volN) / divisor;
                if (delta > deltaBest) {
                    deltaBest = delta;
                    best = D;
                }
            }
        }


        if (deltaBest > 0) { // if modularity improvement possible
            assert (best != C && best != none);// do not "move" to original cluster

            _data[u] = best; // move to best cluster
            // TRACE("node " , u , " moved");

            // mod update
            f_weight volN = 0.0;
            volN = volNode[u];
            // update the volume of the two clusters
#pragma omp atomic update
            volCommunity[C] -= volN;
#pragma omp atomic update
            volCommunity[best] += volN;

            moved = true; // change to clustering has been made

        }
    };
    /********/


    // try to improve modularity by moving a node to neighboring clusters
    auto tryMove = [&](node u) {
        // TRACE("trying to move node " , u);
        index tid = omp_get_thread_num();
        count neigh_counter = 0;
        count vertex_count = 0;
        index _deg = outDegree[u];
        /// Pointer for neighbor vertices. We can access using edge index.
        const node *pnt_outEdges = &outEdges[u][0];
        /// Pointer for neighbor edge weight. We can access using edge index.
        const f_weight *pnt_outEdgeWeight;
        if (isGraphWeighted) {
            pnt_outEdgeWeight = &outEdgeWeights[u][0];
        }
        index *pnt_neigh_comm = &neigh_comm[tid][0];

        /// pointer to calculate the affinity of neighbor community
        f_weight *pnt_affinity = &turboAffinity[tid][0];
        /// Initialize affinity with zero. May be we can use intel intrinsic to do that.
#pragma omp simd
        for(index edge=0; edge<_deg; ++edge){
            pnt_affinity[_data[pnt_outEdges[edge]]] = -1.0;
        }
        pnt_affinity[_data[u]] = 0;

        for (int i = 0; i < _deg; ++i) {
            node v = pnt_outEdges[i];
            if (u != v) {
                index C = _data[v];
                if (pnt_affinity[C] == -1) {
                    // found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                    pnt_affinity[C] = 0;
                    pnt_neigh_comm[neigh_counter++] = C;
                }
                pnt_affinity[C] += isGraphWeighted ? pnt_outEdgeWeight[i] : f_defaultEdgeWeight;
            }
        }

        index best = none;
        f_weight deltaBest = -1;

        index C = _data[u];
        f_weight affinityC = pnt_affinity[C];
        f_weight volN = volNode[u];
        f_weight volCommunityMinusNode_C = volCommunity[C] - volN;

        for (index i=0; i<neigh_counter; ++i) {
            index D = pnt_neigh_comm[i];
            if (D != C) { // consider only nodes in other clusters (and implicitly only nodes other than u)
                f_weight delta = (pnt_affinity[D] - affinityC) / total + this->gamma * ((volCommunityMinusNode_C - volCommunity[D]) * volN) / divisor;
                if (delta > deltaBest) {
                    deltaBest = delta;
                    best = D;
                }
            }
        }


        // TRACE("deltaBest=" , deltaBest);
        if (deltaBest > 0) { // if modularity improvement possible
            assert (best != C && best != none);// do not "move" to original cluster
                                               //#pragma omp atomic update
                                               //                move_count += 1;
            _data[u] = best; // move to best cluster
            // TRACE("node " , u , " moved");

            // mod update
            f_weight volN = 0.0;
            volN = volNode[u];
            // update the volume of the two clusters
#pragma omp atomic update
            volCommunity[C] -= volN;
#pragma omp atomic update
            volCommunity[best] += volN;

            moved = true; // change to clustering has been made

        } else {
            // TRACE("node " , u , " not moved");
        }
    };

    // performs node moves
    double new_mod, old_mod;
    auto movePhase = [&](){
        count iter = 0;
#if POWER_LOG
        //            if(m_iter == 0)
        rapl_sysfs_before();
#endif
        do {
//                move_count = 0;
#if CONFLICT_DETECT
            struct timespec start_move, end_move;
            clock_gettime(CLOCK_REALTIME, &start_move);
#endif
            moved = false;
            //                old_mod = modularity.getQuality(zeta, *G);
            //                count vec_count = 0;
            if(fullVec) {
                if(!isGraphWeighted && m_iter == 0) {
                    G->balancedParallelForNodes(moveToNewCommunityWithDefaultWeight);
                    //                        ++vec_count;
                } else if(isGraphWeighted && m_iter == 0){
                    G->balancedParallelForNodes(tryVecMove);
                    //                        G->balancedParallelForNodes(tryMove);
                    //                        ++vec_count;
                } else/* if(isGraphWeighted)*/{
                    //                        G->balancedParallelForNodes(reducedTryMove);
                    G->balancedParallelForNodes(tryMove);
                    //                        ++vec_count;
                }
            } else {
#pragma omp parallel for schedule(guided)
                for (node uv = 0; uv < z; ++uv) {
                    if (G->hasNode(uv)) {
                        if (outDegree[uv] >= 16) {
                            if (m_iter == 0 && !isGraphWeighted && iter == 0) {
                                moveToNewCommunityWithDefaultWeight(uv);
                                //                                    ++vec_count;
                            } else if (m_iter == 0 && !isGraphWeighted && iter > 0) {
                                //                                    reducedTryMove(uv);
                                moveToNewCommunityWithDefaultWeight(uv);
                                //                                    ++vec_count;
                            } else if (m_iter == 0 && isGraphWeighted) {
                                moveToNewCommunity(uv);
                                //                                    ++vec_count;
                            } else {
                                tryMove(uv);
                            }
                        } else {
                            tryMove(uv);
                        }
                    }
                }
            }
            //G->balancedParallelForNodes(tryMove);

            if (moved) change = true;

            /*if (iter == maxIter) {
                WARN("move phase aborted after ", maxIter, " iterations");
            }*/
            //                std::cout<<"[" << m_iter << "] " << "Move: " << iter << " vector count: " << vec_count << std::endl;
            iter += 1;
//                tot_move_count += move_count;
#if CONFLICT_DETECT
            clock_gettime(CLOCK_REALTIME, &end_move);
            double elapsed_move_time = ((end_move.tv_sec * 1000 + (end_move.tv_nsec / 1.0e6)) - (start_move.tv_sec * 1000 + (start_move.tv_nsec / 1.0e6)));
            f_conflict_log << m_iter << "," << iter << "," << z << "," << G->numberOfEdges() << "," << conflict_community << "," << elapsed_move_time << std::endl;
#endif
            //                new_mod = modularity.getQuality(zeta, *G);
        } while (moved /*&& (new_mod - old_mod)>0.000001*/ && (iter < maxIter) && handler.isRunning());
#if POWER_LOG
        // End the calculate of power
        /*if(m_iter == 0)*/ {
            rapl_sysfs_after();
            // Results
            rapl_sysfs_results("Vectorized Row PLM", _graphName, _threads, m_iter+1);
        }
#endif
        if(m_iter == 0){
            std::cout<< "total move iteration: " << iter << std::endl;
        }
        m_iter++;
        //            std::cout<< m_iter << " iterations: " << iter << " total moves: " << tot_move_count << " avg moves: "
        //            << (tot_move_count/iter) << std::endl;
    };

    handler.assureRunning();
    double old_modularity = modularity.getQuality(zeta, *G);
    // first move phase
    Aux::Timer timer;
    struct timespec c_start, c_end;
    clock_gettime(CLOCK_REALTIME, &c_start);
    timer.start();
    //
    movePhase();
    //
    timer.stop();
    clock_gettime(CLOCK_REALTIME, &c_end);
    double m_time = ((c_end.tv_sec * 1000 + (c_end.tv_nsec / 1.0e6)) - (c_start.tv_sec * 1000 + (c_start.tv_nsec / 1.0e6)));
    std::cout<< m_iter << " Wall time: " << m_time << " aux time: " << timer.elapsedMilliseconds() << std::endl;
    //        printf("[%d] move-phase time (%.4f secs)\n", m_iter, (double)(c_end - c_start) / CLOCKS_PER_SEC);
    //        timing["move"].push_back(timer.elapsedMilliseconds());
    timing["move"].push_back(m_time);
    for (int i = 0; i < o; ++i) {
        _dataV[i] = _data[i];
    }
    zeta.setVector(_dataV);
    double new_modularity = modularity.getQuality(zeta, *G);
    handler.assureRunning();
    /*if(m_iter > 1 && (new_modularity - old_modularity)<0.000001)
        change = false;*/

    if (recurse && change) {
        //            DEBUG("nodes moved, so begin coarsening and recursive call");

        clock_gettime(CLOCK_REALTIME, &c_start);
        //
        std::pair<Graph, std::vector<node>> coarsened = coarsen(*G, zeta);	// coarsen graph according to communitites
        //
        clock_gettime(CLOCK_REALTIME, &c_end);
        double coarsen_time = ((c_end.tv_sec * 1000 + (c_end.tv_nsec / 1.0e6)) - (c_start.tv_sec * 1000 + (c_start.tv_nsec / 1.0e6)));
        //            timing["coarsen"].push_back(timer.elapsedMilliseconds());
        timing["coarsen"].push_back(coarsen_time);



        ONPL onCoarsened(coarsened.first, this->refine, this->gamma, this->parallelism, this->maxIter, this->turbo);
        onCoarsened.run();
        Partition zetaCoarse = onCoarsened.getPartition();

        // get timings
        auto tim = onCoarsened.getTiming();
        for (count t : tim["move"]) {
            timing["move"].push_back(t);
        }
        for (count t : tim["coarsen"]) {
            timing["coarsen"].push_back(t);
        }
        for (count t : tim["refine"]) {
            timing["refine"].push_back(t);
        }


        //            DEBUG("coarse graph has ", coarsened.first.numberOfNodes(), " nodes and ", coarsened.first.numberOfEdges(), " edges");
        zeta = prolong(coarsened.first, zetaCoarse, *G, coarsened.second); // unpack communities in coarse graph onto fine graph
        // refinement phase
        if (refine) {
            //                DEBUG("refinement phase");
            // reinit community-dependent temporaries
            o = zeta.upperBound();
            volCommunity.clear();
            volCommunity.resize(o, 0.0);
            zeta.parallelForEntries([&](node u, index C) { 	// set volume for all communities
                if (C != none) {
                    f_weight volN = volNode[u];
#pragma omp atomic update
                    volCommunity[C] += volN;
                }
            });
            // second move phase
            //                timer.start();
            clock_gettime(CLOCK_REALTIME, &c_start);
            //
            movePhase();
            //
            //                timer.stop();
            clock_gettime(CLOCK_REALTIME, &c_end);
            double ref_time = ((c_end.tv_sec * 1000 + (c_end.tv_nsec / 1.0e6)) - (c_start.tv_sec * 1000 + (c_start.tv_nsec / 1.0e6)));
            //                std::cout << "Elapsed Time Per Move: " << timer.elapsedMilliseconds() << std::endl;
            //                timing["refine"].push_back(timer.elapsedMilliseconds());
            timing["refine"].push_back(ref_time);

        }
    }
    result = std::move(zeta);
    hasRun = true;
}

std::string NetworKit::ONPL::toString() const {
    std::stringstream stream;
    stream << "ONPL(";
    stream << parallelism;
    if (refine) {
        stream << "," << "refine";
    }
    stream << "," << "pc";
    if (turbo) {
        stream << "," << "turbo";
    }
    if (!recurse) {
        stream << "," << "non-recursive";
    }
    stream << ")";

    return stream.str();
}

std::pair<Graph, std::vector<node> > ONPL::coarsen(const Graph& G, const Partition& zeta) {
    ParallelPartitionCoarsening parCoarsening(G, zeta);
    parCoarsening.run();
    return {parCoarsening.getCoarseGraph(),parCoarsening.getFineToCoarseNodeMapping()};
}

Partition ONPL::prolong(const Graph& Gcoarse, const Partition& zetaCoarse, const Graph& Gfine, std::vector<node> nodeToMetaNode) {
    Partition zetaFine(Gfine.upperNodeIdBound());
    zetaFine.setUpperBound(zetaCoarse.upperBound());

    Gfine.forNodes([&](node v) {
        node mv = nodeToMetaNode[v];
        index cv = zetaCoarse[mv];
        zetaFine[v] = cv;
    });


    return zetaFine;
}



std::map<std::string, std::vector<double > > ONPL::getTiming() {
    return timing;
}

} /* namespace NetworKit */