// no-networkit-format
#include <limits>
#include <map>

#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/community/StablePartitionNodes.hpp>
#include<time.h>
#include <sys/time.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>

void NetworKit::StablePartitionNodes::run() {
    hasRun = false;

    Aux::SignalHandler handler;

    stableMarker.clear();
    stableMarker.resize(G->upperNodeIdBound(), true);
    values.clear();

    handler.assureRunning();

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

    std::cout<<"start edge weight calculation" << std::endl;
    if(!isVectorized()) {
        // first determine which nodes are stable
        G->balancedParallelForNodes([&](node u) {
            if (G->degree(u) > 0) { // we consider isolated nodes to be stable.
                std::map<index, count> labelWeights;
                G->forNeighborsOf(u, [&](node v, edgeweight ew) {
                    if(Com[v] > Com.upperBound()){
                        std::cout<<"[stable]problem with label of the community: "
                                  << Com[v] << " where upper bound: "
                                  << Com.upperBound() << std::endl;
                        return;
                    }
                    assert(Com[v] <= Com.upperBound());
                    labelWeights[Com[v]] += ew;
                });
                if(Com[u] > Com.upperBound()){
                    std::cout<<"[stable]problem with label of the community: "
                              << Com[u] << " where upper bound: "
                              << Com.upperBound() << std::endl;
                    return;
                }
                index ownLabel = Com[u];
                assert(Com[u] <= Com.upperBound());
                double ownWeight = labelWeights[ownLabel];

                if (ownWeight == 0) {
                    stableMarker[u] = false;
                } else {
                    for (auto lw : labelWeights) {
                        if (lw.first != ownLabel && lw.second >= ownWeight) {
                            stableMarker[u] = false;
                            break;
                        }
                    }
                }
            }
        });
    }
    else {
        const Partition C = Com;
        index max_tid = omp_get_max_threads();
//        std::vector<std::vector<f_weight>> labelWeights(max_tid, std::vector<f_weight>(Com.upperBound(), 0));
        std::vector<std::vector<f_weight>> labelWeights(max_tid, std::vector<f_weight>(G->upperNodeIdBound(), 0));
        const std::vector<f_weight> *outEdgeWeights = G->getOutEdgeWeights();
        const std::vector<node> *outEdges = G->getOutEdges();
        index** neigh_comm = (index **) malloc(max_tid * sizeof(index *));
        G->balancedParallelForNodes([&](node u) {
            count _deg = G->degree(u);
            if (_deg > 0) {
                index tid = omp_get_thread_num();
                /// Pointer for neighbor vertices. We can access using edge index.
                const node *pnt_outEdges = &outEdges[u][0];
                /// Pointer for neighbor edge weight. We can access using edge index.
                const f_weight *pnt_outEdgeWeight = &outEdgeWeights[u][0];
                index *pnt_neigh_comm = &neigh_comm[tid][0];
                count neigh_counter = 0;
                f_weight* pnt_myNeighborLabel = &labelWeights[tid][0];
                index i = 0;
                for (i = 0; (i+16) <= _deg; i += 16) {
                    __m512i v_vec = _mm512_loadu_si512((__m512i * ) & pnt_outEdges[i]);
                    __m512i C_vec = _mm512_i32gather_epi32(v_vec, &Com[0], 4);
                    _mm512_i32scatter_ps(&pnt_myNeighborLabel[0], C_vec, fl_set1, 4);
                }
                for (index edge = i; edge < _deg; ++edge) {
                    pnt_myNeighborLabel[Com[pnt_outEdges[edge]]] = -1.0;
                }
//                pnt_myNeighborLabel[Com[u]] = 0;

                const   __m512i check_self_loop = _mm512_set1_epi32(u);
#pragma unroll
                for (i = 0; (i+16) <= _deg; i += 16) {
                    __m512i v_vec = _mm512_loadu_si512((__m512i *) &pnt_outEdges[i]);
                    /// Load at most 16 neighbor vertex edge weight.
                    __m512 w_vec = _mm512_loadu_ps((__m512 *) &pnt_outEdgeWeight[i]);
                    /// Mask to find u != v
                    const __mmask16 self_loop_mask = _mm512_cmpneq_epi32_mask(check_self_loop, v_vec);
                    /// Gather community of the neighbor vertices.
                    __m512i C_vec = _mm512_mask_i32gather_epi32(set0, self_loop_mask, v_vec, &Com[0], 4);
                    /// Gather affinity of the corresponding community.
                    __m512 label_vec = _mm512_mask_i32gather_ps(fl_set0, self_loop_mask, C_vec, &pnt_myNeighborLabel[0], 4);

                    /// Mask to find out the new community that contains -1.0 value
                    const __mmask16 new_comm_mask = _mm512_mask_cmpeq_ps_mask(self_loop_mask, fl_set1, label_vec);
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
                    label_vec = _mm512_mask_mov_ps(label_vec, new_comm_mask, fl_set0);
                    /// Add edge weight to the affinity and if mask doesn't set load from affinity
                    label_vec = _mm512_mask_add_ps(label_vec, mask, label_vec, w_vec);
                    /// Scatter affinity value to the affinity pointer.
                    _mm512_mask_i32scatter_ps(&pnt_myNeighborLabel[0], mask, C_vec, label_vec, 4);

                    /// Count the set bit from the mask for ignore vertices
                    __mmask16 conflict_comm_mask = _mm512_kand(_mm512_knot(mask), self_loop_mask);
                    sint vertex_cnt = _mm_popcnt_u32((unsigned)conflict_comm_mask);

                    if(vertex_cnt>0) {
                        w_vec = _mm512_mask_compress_ps(fl_set0, self_loop_mask, w_vec);
                        f_weight *weight_not_processed = (f_weight *) &w_vec;
                        C_vec = _mm512_mask_compress_epi32(set0, conflict_comm_mask, C_vec);
                        index *remaining_comm = (index *) &C_vec;
                        for (int j = 0; j < vertex_cnt; ++j) {
                            pnt_myNeighborLabel[remaining_comm[j]] += weight_not_processed[j];
                        }
                    }
                }

                pnt_outEdges = &outEdges[u][0];
                pnt_outEdgeWeight = &outEdgeWeights[u][0];
                for (index j= i; j < _deg; ++j) {
                    node v = pnt_outEdges[j];
                    if (u != v) {
                        index c = Com[v];
                        if (pnt_myNeighborLabel[c] == -1) {
                            /// found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                            pnt_myNeighborLabel[c] = 0;
                            pnt_neigh_comm[neigh_counter++] = c;
                        }
                        pnt_myNeighborLabel[c] += pnt_outEdgeWeight[j];
                    }
                }

                index my_c = Com[u];
                f_weight my_com_weight = pnt_myNeighborLabel[my_c];
                if(my_com_weight <= 0){
                    stableMarker[u] = false;
                } else {
                    const __m512i reg_C = _mm512_set1_epi32(my_c);

                    for (i = 0; (i + 16) <= neigh_counter; i += 16) {
                        /// Load at most 16 neighbor community.
                        __m512i D_vec = _mm512_loadu_si512((__m512i *)&pnt_neigh_comm[i]);
                        /// Mask to find C != D
                        const __mmask16 different_comm_mask =
                            _mm512_cmpneq_epi32_mask(reg_C, D_vec);
                        __m512 label_D_vec = _mm512_i32gather_ps(D_vec, &pnt_myNeighborLabel[0], 4);
                        f_weight max_lavel_val = _mm512_mask_reduce_max_ps(different_comm_mask, label_D_vec);
                        if (max_lavel_val > my_com_weight) {
                            stableMarker[u] = false;
                            break;
                        }
                    }
                    for (auto j=i; j<neigh_counter; ++j) {
                        if (my_c != pnt_neigh_comm[j] && pnt_myNeighborLabel[j] >= my_com_weight) {
                            stableMarker[u] = false;
                            break;
                        }
                    }
                }
            }
        });
    }
    std::cout<<"done edge weight calculation" << std::endl;
    handler.assureRunning();

    values.resize(Com.upperBound(), 0);
//    values.resize(G->upperNodeIdBound(), 0);
    std::vector<count> partitionSizes(Com.upperBound(), 0);
//    std::vector<count> partitionSizes(G->upperNodeIdBound(), 0);
    count stableCount = 0;
    std::cout<<"collect how many nodes are stable in which partition" << std::endl;
    // collect how many nodes are stable in which partition
    G->forNodes([&](node u) {
        index label = Com[u];
        if(Com[u] > Com.upperBound()){
            std::cout<<"[stable]problem with label of the community: "
                      << Com[u] << " where upper bound: "
                      << Com.upperBound() << std::endl;
            return;
        }
        assert(label <= Com.upperBound());
        if(label >= partitionSizes.size()){
            std::cout<<"partitionSizes is not correct; label: " << label << " max size: " << partitionSizes.size() << " upper bound: " << Com.upperBound() <<std::endl;
        }
        if(label >= values.size()){
            std::cout<<"values is not correct; label: " << label << " values size: " << values.size() << " upper bound: " << Com.upperBound() <<std::endl;
        }
        if(u >= stableMarker.size()){
            std::cout<<"stableMarker is not correct; u: " << u << " out of bound: " << stableMarker.size() <<std::endl;
        }
        ++partitionSizes[label];
        values[label] += stableMarker[u];
        stableCount += stableMarker[u];
    });

    count numClusters = 0;
    unweightedAverage = 0;
    minimumValue = std::numeric_limits<double>::max();
    maximumValue = std::numeric_limits<double>::lowest();
    std::cout<<"calculate all average/max/min-values" << std::endl;
    // calculate all average/max/min-values
//    for (index i = 0; i < Com.upperBound(); ++i) {
    for (index i = 0; i < G->upperNodeIdBound(); ++i) {
        if (partitionSizes[i] > 0) {
            values[i] /= partitionSizes[i];
            unweightedAverage += values[i];
            minimumValue = std::min(minimumValue, values[i]);
            maximumValue = std::max(maximumValue, values[i]);
            ++numClusters;
        }
    }
    std::cout<<"done calculation" << std::endl;
    unweightedAverage /= numClusters;
    weightedAverage = stableCount * 1.0 / G->numberOfNodes();
    std::cout<<"return" << std::endl;
    handler.assureRunning(); // make sure we do not ignore the signal sent by the user

    hasRun = true;
}
