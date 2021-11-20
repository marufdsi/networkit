/*
 * OVPL.cpp
 *
 *  Created on: 10.10.2018
 *      Author: Md Maruf Hossain
 */

#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <inttypes.h>

#include <omp.h>
#include <networkit/community/Modularity.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/coarsening/ClusteringProjector.hpp>
#include <networkit/coarsening/ParallelPartitionCoarsening.hpp>
#include <networkit/community/OVPL.hpp>
#include <networkit/auxiliary/PowerCalculator.hpp>

#include <iostream>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <fstream>
#include <limits>
//#include "fvec.h"
#include <mmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <cmath>
#include<time.h>
#include <sys/time.h>
#include <math.h>
#include <atomic>

#define MOVE_PHASE_LOG false
#define FIRST_MOVE_PHASE_LOG false
#define INNER_MOVE_LOG false
#define FIRST_MOVE_PHASE_DETAILS_LOG false
#define DETAILS_LOG false
#define ONE_THREAD false

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

//#ifndef L1D_CACHE_MISS_COUNT
//#define L1D_CACHE_MISS_COUNT
//#endif


typedef int32_t sint;

namespace NetworKit {

    OVPL::OVPL(const Graph& G, bool refine, f_weight gamma, std::string par, count maxIter, bool turbo, bool recurse) : CommunityDetectionAlgorithm(G), parallelism(par), refine(refine), gamma(gamma), maxIter(maxIter), turbo(turbo), recurse(recurse) {

    }

    OVPL::OVPL(const Graph& G, const OVPL& other) : CommunityDetectionAlgorithm(G), parallelism(other.parallelism), refine(other.refine), gamma(other.gamma), maxIter(other.maxIter), turbo(other.turbo), recurse(other.recurse) {

    }

    count block_size = 16;
    count frame_size = 0;
    std::ofstream f_move_log;
    std::ofstream f_first_move_log;
    std::ofstream f_first_move_phase_details_log;
    std::ofstream f_ovpl_details_log;
    count _count_mod = 0;

    int _itter = 0;

    void OVPL::setupCSVFile(std::string move_log_file, std::string first_move_log_file, std::string first_move_phase_details_log, std::string ovpl_details_log) {
#if MOVE_PHASE_LOG
        f_move_log.open(move_log_file);
            f_move_log << "Move Phase"<< "," << "Iteration"<< ","<< "Average Move Time"<< "," << "Time"<< "," << "Modularity"<< "," << "Clusters"<< std::endl;
#endif
#if FIRST_MOVE_PHASE_LOG
        f_first_move_log.open(first_move_log_file);
            f_first_move_log << "Move"<< "," << "Time"<< ","<< "Clusters"<<","<<"Modularity"<<","<<"Time for Affinity"<<","<<"Time for Clustering"<< std::endl;
#endif
#if FIRST_MOVE_PHASE_DETAILS_LOG
        std::ifstream infile(first_move_phase_details_log);
        bool existing_file = infile.good();
        f_first_move_phase_details_log.open(first_move_phase_details_log, std::ios_base::out | std::ios_base::app | std::ios_base::ate);
        if (!existing_file) {
            f_first_move_phase_details_log << "Coloring Time"<< "," << "Sorting Time" << "," << "Reformatting Time"<< "," << "Initialization Time" << "," << "Move Time" <<"," << "Coarsening Time"<< std::endl;
        }
//        f_first_move_phase_details_log.open(first_move_phase_details_log);
//        f_first_move_phase_details_log << "Coloring Time"<< "," << "Sorting Time" << "," << "Reformatting Time"<< "," << "Initialization Time" << "," << "Move Time" <<"," << "Coarsening Time"<< std::endl;
#endif
#if DETAILS_LOG
        f_ovpl_details_log.open(ovpl_details_log);
        f_ovpl_details_log << "Move Phase"<< "," << "Move Time" <<"," << "Coarsening Time"<< "," << "Initial Modularity" << "," << "New Modularity" << "," << "Initial Community" << "," << "New Community" << std::endl;
#endif
    }

    void parallel_coloring(sint maxNode, node nodes[], const count *outDegree, const std::vector<node> *outEdges, sint *track_groups, std::vector<sint> *markBase, sint *maxc, sint *nGroup) {

#pragma omp parallel shared(outDegree, outEdges, track_groups, nGroup, maxNode)
        {
            sint max_thread = omp_get_max_threads();
            sint tid = omp_get_thread_num();
            maxc[tid] = *nGroup;
            sint *mark = &markBase[tid][0];
#pragma omp for schedule(guided)
            for (index i = 0; i < maxNode; ++i) {
                node u = nodes[i];
                for (index edge = 0; edge < outDegree[u]; ++edge) {
                    node v = outEdges[u][edge];
                    if (u == v) continue;
                    sint groupID = track_groups[v];
                    if (groupID >= 0) {
                        mark[groupID] = u;
                    }
                }
                sint groupID;
                for (groupID = 0; mark[groupID] == u; ++groupID);
                if (groupID > maxc[tid]) {
                    maxc[tid] = groupID;
                }
                track_groups[u] = groupID;
            }
#pragma omp master
            {
                for (index i=0; i< max_thread; i++)
                    if (*nGroup < maxc[i])
                        *nGroup = maxc[i];
            }
        }
    }

    sint detectConflict(sint maxNode, const count *outDegree, const std::vector<node> *outEdges, sint *track_groups, node conflicts[], sint counts[], sint group_counts[]) {

        std::atomic<int> conflict(omp_get_max_threads());
        conflict = 0;
        index i;
#pragma omp parallel shared(outDegree, outEdges, track_groups, conflict)
        {
#pragma omp  for schedule(guided)
            for (i = 0; i < maxNode; ++i) { //for all vertices
                node u = conflicts[i];
                for (index edge = 0; edge < outDegree[u]; ++edge) { //build the mark array
                    node v = outEdges[u][edge];
                    if (u != v && track_groups[u] == track_groups[v]) { //not-self loop and conflict
                        if (u < v) {
                            conflicts[atomic_fetch_add(&conflict, 1)] = u;
                            track_groups[u] = -1;
                            break;
                        }
                    }
                }
                if(track_groups[u] >= 0) {
#pragma omp atomic update
                    counts[track_groups[u]]++;
#pragma omp atomic update
                    group_counts[track_groups[u]]++;
                }
            }
        }
        return conflict;
    }

    void seq_prefix_sum(sint size, sint sum, sint start, sint *arr, std::atomic<sint> *prefix){
        sint next = arr[start-1];
#pragma novector
        for(sint i=start; i<size; ++i){
            sum += next;
            next = arr[i];
            arr[i] = sum;
            prefix[i] = sum;
        }
    }

    void parallel_prefix_sum(sint size, sint *arr, std::atomic<sint> *prefix){
        for(sint d=0; d<=(log2(size)-1); ++d){
#pragma omp parallel for schedule(guided)
#pragma novector
            for(sint i=0; i<=(size-1); i+=(sint)pow(2,(d+1))){
                arr[i+(sint)pow(2,(d+1))-1] = arr[i + (sint)pow(2,d) -1] + arr[i+(sint)pow(2,(d+1))-1];
                prefix[i+(sint)pow(2,(d+1))-1] = arr[i+(sint)pow(2,(d+1))-1];
            }
        }
        arr[size-1] = 0;
        for(sint d=(sint)log2(size)-1; d>=0; --d){
#pragma omp parallel for schedule(guided)
#pragma novector
            for(sint i=0; i<=size-1; i+=(sint)pow(2, (d+1))){
                sint t = arr[i+(sint)pow(2,d)-1];
                arr[i+(sint)pow(2,d)-1] = arr[i+(sint)pow(2,(d+1))-1];
                prefix[i+(sint)pow(2,d)-1] = arr[i+(sint)pow(2,d)-1];
                arr[i+(sint)pow(2,(d+1))-1] = t + arr[i+(sint)pow(2,(d+1))-1];
                prefix[i+(sint)pow(2,(d+1))-1] = arr[i+(sint)pow(2,(d+1))-1];
            }
        }
    }

    void assign_group(count max_node, count nGroup, sint *track_groups, sint counts[], sint groups[]) {
        sint size = pow(2, floor(log2(nGroup)));
        sint sum = counts[size-1];
        std::atomic<sint> prefix[nGroup];
        parallel_prefix_sum(size, counts, &prefix[0]);
        seq_prefix_sum(nGroup, sum, size, counts, &prefix[0]);
#pragma omp parallel for schedule(guided)
        for (index u = 0; u < max_node; ++u) {
            sint groupID = track_groups[u];
            groups[atomic_fetch_add(&prefix[groupID], 1)] = u;
        }
    }


    long OVPL::perf_event_open2(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags){
        int ret;
        ret = syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
        return ret;
    }
    void OVPL::run() {
//        std::cout<<"Run method Start"<<std::endl;
//        maxIter = 32;
        size_t alignment = 64;
        double elapsed_coarsen_time = 0.0, elapsed_move_phase_time = 0.0, elapsed_coloring_time = 0.0, elapsed_sorting_time = 0.0, elapsed_reformatting_time = 0.0, elapsed_initialization_time = 0.0;
        Aux::SignalHandler handler;
        Modularity modularity;
        /// perf event attributes
        struct perf_event_attr pe;
        long long cache_miss_count;
        int fd;
        ///
        DEBUG("calling run method on " , G.toString());

        #if ONE_THREAD
            omp_set_dynamic(0);
            omp_set_num_threads(1);
        #endif
        count z = G.upperNodeIdBound();
//        count move_count, tot_move_count = 0;
        double affinity_time = 0.0;
        double clustering_time = 0.0;

        // init communities to singletons
        Partition zeta(z);
        zeta.allToSingletons();
        index o = zeta.upperBound();

        // init graph-dependent temporaries
        std::vector<f_weight > volNode(z, 0.0);
        // $\omega(E)$
        f_weight total = G.totalEdgeWeight();
        DEBUG("total edge weight: " , total);
        f_weight divisor = (2 * total * total); // needed in modularity calculation

        /// Declare variable to support vectorization
        const count *outDegree;
        const std::vector<f_weight> *outEdgeWeights;
        const std::vector<node> *outEdges;
        bool isGraphWeighted = G.isWeighted();
        // need to replace by _group
//        std::vector<std::vector<node> > node_sort_by_deg;
        node** node_sort_by_deg;
        std::vector<sint> track_groups;
        sint nGroup = 0;
        node *all_nodes;
        all_nodes = (node *)malloc(sizeof(node)*z);
        sint *_group;
        _group = (sint *)malloc(sizeof(sint)*z);
        sint *color_counts;
        color_counts = (sint *)malloc(sizeof(sint)*z);
        sint *group_counts;
        group_counts = (sint *)malloc(sizeof(sint)*z);
        index max_node = ((sint)(z/block_size)) * block_size;
        frame_size = z / block_size;
        std::vector<count> _max_deg;
        std::vector<count> _min_deg;
//        std::vector<std::vector<count> > deg;
        count** deg;
//        std::vector<std::vector<f_weight> > _sequential_edge_weight __attribute__((aligned(32)));
        f_weight** _sequential_edge_weight;
//        std::vector<std::vector<node> > _sequential_out_edges __attribute__((aligned(32)));
        node** _sequential_out_edges;
//        std::vector<std::vector<std::vector<f_weight> > > nodeAffinity;
        f_weight*** nodeAffinity;
//        std::vector<std::vector<std::vector<index> > > node_neigh_comm;
        index*** node_neigh_comm;

//        std::cout<<"Start Process"<<std::endl;
        sint max_deg_of_graph = 0;
        index max_tid = omp_get_max_threads();
//        std::cout<<"Max Thread: "<< max_tid <<std::endl;
        if(_itter == 0){
            outDegree = G.getOutDegree();
            index max_deg_arr[max_tid];
            #pragma omp parallel
            {
                index tid = omp_get_thread_num();
                max_deg_arr[tid] = max_deg_of_graph;
                #pragma omp for schedule(guided)
                for (index u = 0; u < z; ++u) {
                    volNode[u] += G.weightedDegree(u);
                    volNode[u] += G.weight(u, u); // consider self-loop twice
                    // TRACE("init volNode[" , u , "] to " , volNode[u]);
                    if(max_deg_arr[tid] < outDegree[u]){
                        max_deg_arr[tid] = outDegree[u];
                    }
                    all_nodes[u] = u;
//                    _group[u] = -1;
                    color_counts[u] = 0;
                    group_counts[u] = 0;
                }
                #pragma omp master
                {
                    for (index i=0; i< max_tid; i++)
                        if (max_deg_of_graph < max_deg_arr[i])
                            max_deg_of_graph = max_deg_arr[i];
                }
            }
//            std::cout<<"Max Degree_of Graph: "<<max_deg_of_graph<<std::endl;

        } else {
            G.parallelForNodes([&](node u) { // calculate and store volume of each node
                volNode[u] += G.weightedDegree(u);
                volNode[u] += G.weight(u, u); // consider self-loop twice
                // TRACE("init volNode[" , u , "] to " , volNode[u]);
            });
        }

        // init community-dependent temporaries
        std::vector<double> volCommunity(o, 0.0);
        zeta.parallelForEntries([&](node u, index C) { 	// set volume for all communities
            if (C != none)
                volCommunity[C] = volNode[u];
        });

        if(_itter == 0) {
            auto Merge = [&](sint low, sint high, sint mid) {
                // We have low to mid and mid+1 to high already sorted.
                sint i, j, k, *temp;
                temp = (sint *)malloc(sizeof(sint)*(high-low+1));
                i = low;
                k = 0;
                j = mid + 1;
                // Merge the two parts into temp[].
                while (i <= mid && j <= high) {
                    if (outDegree[_group[i]] < outDegree[_group[j]]) {
                        temp[k] = _group[i];
                        k++;
                        i++;
                    } else {
                        temp[k] = _group[j];
                        k++;
                        j++;
                    }
                }
                // Insert all the remaining values from i to mid into temp[].
                while (i <= mid) {
                    temp[k] = _group[i];
                    k++;
                    i++;
                }
                // Insert all the remaining values from j to high into temp[].
                while (j <= high) {
                    temp[k] = _group[j];
                    k++;
                    j++;
                }

                // Assign sorted data stored in temp[] to a[].
                for (i = low; i <= high; i++) {
                    _group[i] = temp[i-low];
                }
            };

            std::function<void (index, index)> sortGraph = [&](index low, index high){
                sint mid;
                if (low < high)
                {
                    mid=(low+high)/2;
                    // Split the data into two half.
                    sortGraph(low, mid);
                    sortGraph(mid+1, high);

                    // Merge them to get sorted output.
                    Merge(low, high, mid);
                }
            };
#if FIRST_MOVE_PHASE_DETAILS_LOG
            struct timespec start_initialization, end_initialization;
            clock_gettime(CLOCK_REALTIME, &start_initialization);
#endif
            outEdgeWeights = G.getOutEdgeWeights();
            outEdges = G.getOutEdges();
            std::vector<std::vector<sint> > mark;
            mark.resize(max_tid);
//            nodeAffinity.resize(max_tid);
//            node_neigh_comm.resize(max_tid);
            nodeAffinity = (f_weight ***) malloc(max_tid * sizeof(f_weight **));
            node_neigh_comm = (index ***) malloc(max_tid * sizeof(index **));
#pragma omp parallel for schedule(guided)
            for (index tid=0; tid<max_tid; ++tid) {
//                nodeAffinity[tid].resize(block_size);
//                node_neigh_comm[tid].resize(block_size);
                nodeAffinity[tid] = (f_weight **) malloc(block_size * sizeof(f_weight *));
                node_neigh_comm[tid] = (index **) malloc(block_size * sizeof(index *));
                for(index block=0; block<block_size; ++block){
//                    nodeAffinity[tid][block].resize(zeta.upperBound());
//                    node_neigh_comm[tid][block].resize(max_deg_of_graph);
                    posix_memalign((void **) &nodeAffinity[tid][block], alignment, zeta.upperBound() * sizeof(f_weight));
                    posix_memalign((void **) &node_neigh_comm[tid][block], alignment, max_deg_of_graph * sizeof(index));
                }
                mark[tid].resize(z, -1);
            }
//            std::cout<<"Initialization Done"<<std::endl;
            std::vector<sint> maxc;
            maxc.resize(max_tid);
            track_groups = std::vector<sint>(z, -1);
            count n_conflict = max_node;

            _max_deg = std::vector<count>(frame_size);
            _min_deg = std::vector<count>(frame_size);
//            deg.resize(frame_size);
//            node_sort_by_deg.resize(frame_size);
            deg = (count **) malloc(frame_size * sizeof(count *));
            node_sort_by_deg = (node **) malloc(frame_size * sizeof(node *));
            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < frame_size; ++i) {
//                deg[i].resize(block_size);
//                node_sort_by_deg[i].resize(block_size);
                posix_memalign((void **) &node_sort_by_deg[i], alignment, block_size * sizeof(node));
                posix_memalign((void **) &deg[i], alignment, block_size * sizeof(count));
            }
//            deg = std::vector<std::vector<count> >(frame_size, std::vector<count>(block_size));
//            node_sort_by_deg = std::vector<std::vector<node> >(frame_size, std::vector<node>(block_size));
//            _sequential_edge_weight.resize(frame_size);
//            _sequential_out_edges.resize(frame_size);
            _sequential_edge_weight = (f_weight **) malloc(frame_size * sizeof(f_weight *));
            _sequential_out_edges = (node **) malloc(frame_size * sizeof(node *));

#if FIRST_MOVE_PHASE_DETAILS_LOG
            clock_gettime(CLOCK_REALTIME, &end_initialization);
            elapsed_initialization_time = ((end_initialization.tv_sec*1000 + (end_initialization.tv_nsec/1.0e6)) - (start_initialization.tv_sec*1000 + (start_initialization.tv_nsec/1.0e6)));

            struct timespec start_coloring;
            clock_gettime(CLOCK_REALTIME, &start_coloring);
#endif
            do {
                parallel_coloring(n_conflict, all_nodes, &outDegree[0], &outEdges[0], &track_groups[0], &mark[0], &maxc[0], &nGroup);
                sint conflicts = detectConflict(n_conflict, &outDegree[0], &outEdges[0], &track_groups[0], all_nodes, color_counts, group_counts);
                n_conflict = conflicts;
            } while (n_conflict>0);
            nGroup++;
            assign_group(max_node, nGroup, &track_groups[0], color_counts, _group);
//            std::cout<<"Coloring Done"<<std::endl;
#if FIRST_MOVE_PHASE_DETAILS_LOG
            struct timespec end_coloring, start_sorting, end_sorting, start_reformatting, end_reformatting;
            clock_gettime(CLOCK_REALTIME, &end_coloring);
            elapsed_coloring_time = ((end_coloring.tv_sec*1000 + (end_coloring.tv_nsec/1.0e6)) - (start_coloring.tv_sec*1000 + (start_coloring.tv_nsec/1.0e6)));

            clock_gettime(CLOCK_REALTIME, &start_sorting);
#endif
//            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < nGroup; ++i) {
                if(group_counts[i] > block_size) {
                    sortGraph(color_counts[i], color_counts[i]+group_counts[i]-1);
                }
            }
//            std::cout<<"Sorting Done"<<std::endl;
#if FIRST_MOVE_PHASE_DETAILS_LOG
            clock_gettime(CLOCK_REALTIME, &end_sorting);
            elapsed_sorting_time = ((end_sorting.tv_sec*1000 + (end_sorting.tv_nsec/1.0e6)) - (start_sorting.tv_sec*1000 + (start_sorting.tv_nsec/1.0e6)));

            clock_gettime(CLOCK_REALTIME, &start_reformatting);
#endif

            if (z >= block_size) {
#pragma omp parallel for schedule(guided)
                for (node u = 0; u < frame_size; u++) {
                    count max_deg = 0;
                    count min_deg = std::numeric_limits<int>::max();
                    for (index i = 0; i < block_size; ++i) {
                        node node_u = _group[(u*block_size)+i];
                        deg[u][i] = outDegree[node_u];
                        max_deg = deg[u][i] > max_deg ? deg[u][i] : max_deg;
                        min_deg = deg[u][i] < min_deg ? deg[u][i] : min_deg;
                    }
//                    _sequential_edge_weight[u].resize(block_size * max_deg);
//                    _sequential_out_edges[u].resize(block_size * max_deg);
                    posix_memalign((void **) &_sequential_edge_weight[u], alignment, (block_size * max_deg) * sizeof(f_weight));
                    posix_memalign((void **) &_sequential_out_edges[u], alignment, (block_size * max_deg) * sizeof(node));
                    for (index i = 0; i < block_size; ++i) {
                        node node_u = _group[(u*block_size)+i];
#pragma omp simd
                        for (index ithEdge = 0; ithEdge < outDegree[node_u]; ++ithEdge) {
                            node v = outEdges[node_u][ithEdge];
                            index _position = i + ithEdge * block_size;
                            if (isGraphWeighted) {
                                _sequential_edge_weight[u][_position] = (node_u == v) ? 0 : outEdgeWeights[node_u][ithEdge];
                            } else {
                                _sequential_edge_weight[u][_position] = (node_u == v) ? 0 : defaultEdgeWeight;
                            }
                            _sequential_out_edges[u][_position] = v;
                        }
                        node_sort_by_deg[u][i] = node_u;
                    }
                    _max_deg[u] = max_deg;
                    _min_deg[u] = min_deg;
                }
            }
//            std::cout<<"Reformatting Done"<<std::endl;
#if FIRST_MOVE_PHASE_DETAILS_LOG
            clock_gettime(CLOCK_REALTIME, &end_reformatting);
            elapsed_reformatting_time = ((end_reformatting.tv_sec*1000 + (end_reformatting.tv_nsec/1.0e6)) - (start_reformatting.tv_sec*1000 + (start_reformatting.tv_nsec/1.0e6)));
#endif
        }

        // first move phase
        bool moved = false; // indicates whether any node has been moved in the last pass
        bool change = false; // indicates whether the communities have changed at all

        // stores the affinity for each neighboring community (index), one vector per thread
        std::vector<std::vector<f_weight > > turboAffinity;
        // stores the list of neighboring communities, one vector per thread
        std::vector<std::vector<index> > neigh_comm;

        if(_itter != 0) {
            if (this->parallelism != "none" &&
                this->parallelism != "none randomized") { // initialize arrays for all threads only when actually needed
                turboAffinity.resize(omp_get_max_threads());
                neigh_comm.resize(omp_get_max_threads());
                for (auto &it : turboAffinity) {
                    // resize to maximum community id
                    it.resize(zeta.upperBound());
                }
            } else { // initialize array only for first thread
                turboAffinity.emplace_back(zeta.upperBound());
                neigh_comm.emplace_back();
            }
        }

        /// Initialize perf events
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_HW_CACHE;
        pe.size = sizeof(struct perf_event_attr);
#ifdef L1D_CACHE_MISS_COUNT
        pe.config = PERF_COUNT_HW_CACHE_L1D |
                    PERF_COUNT_HW_CACHE_OP_READ << 8 |
                    PERF_COUNT_HW_CACHE_RESULT_MISS << 16;
#else
        pe.config = PERF_COUNT_HW_CACHE_LL |
                    PERF_COUNT_HW_CACHE_OP_READ << 8 |
                    PERF_COUNT_HW_CACHE_RESULT_MISS << 16;
#endif
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        // Don't count hypervisor events.
        pe.exclude_hv = 1;

        fd = perf_event_open2(&pe, 0, -1, -1, 0);
        if (fd == -1) {
            fprintf(stderr, "Error opening leader %llx\n", pe.config);
            exit(EXIT_FAILURE);
        }
        ///

        // try to improve modularity by moving a node to neighboring clusters

        auto vectorMove = [&](index _frame) {
            // TRACE("trying to move node " , u);
            index tid = omp_get_thread_num();
            // collect edge weight to neighbor clusters
            count max_deg = _max_deg[_frame];
            count min_deg = _min_deg[_frame];
            count* _deg __attribute__((aligned(32)));
            _deg = &deg[_frame][0];
            index neighbor_count[block_size];
            f_weight** affinity;
            index** neighbors;

            affinity = &nodeAffinity[tid][0];
            neighbors = &node_neigh_comm[tid][0];
            node nodes[block_size];

            const f_weight* sequential_edge_weight __attribute__((aligned(32)));
            const node* sequential_out_edges __attribute__((aligned(32)));
            sequential_edge_weight = &_sequential_edge_weight[_frame][0];
            sequential_out_edges = &_sequential_out_edges[_frame][0];

            for (index i = 0; i < block_size; i++) {
                index _node = node_sort_by_deg[_frame][i];
#pragma omp simd
                for (index edge = 0; edge < _deg[i]; ++edge) {
                    node v = sequential_out_edges[i + edge*block_size];
                    affinity[i][zeta[v]] = -1;
                }
                affinity[i][zeta[_node]] = 0;
                neighbor_count[i] = 0;
                nodes[i] = _node;
            }

#if INNER_MOVE_LOG
            //            auto start_affinity = std::chrono::high_resolution_clock::now();
            struct timespec start_affinity, end_affinity, start_clustering, end_clustering;
            clock_gettime(CLOCK_REALTIME, &start_affinity);
#endif
            for (index edge = 0; edge < min_deg; ++edge) {
#pragma omp simd
                for (node _block = 0; _block < block_size; ++_block) {
                    node v = sequential_out_edges[_block];
                    index C = zeta[v];
                    f_weight* affinity_u = &affinity[_block][C];
                    f_weight weight = *affinity_u;
                    count new_community = (weight == -1);
                    neighbors[_block][neighbor_count[_block]] = C;
                    neighbor_count[_block] += new_community;
                    weight = (1-new_community)*(weight);
                    *affinity_u = weight + sequential_edge_weight[_block];
                }
                sequential_out_edges += block_size;
                sequential_edge_weight += block_size;
            }
            f_weight _none = 0.0;
            for (index edge = min_deg; edge < max_deg; ++edge) {
#pragma omp simd
                for (node _block = 1; _block < block_size; ++_block) {

                    node v = (_deg[_block] > edge) ? sequential_out_edges[_block] : 0;
                    index C = (_deg[_block] > edge) ? zeta[v] : 0;
                    f_weight *affinity_u = (_deg[_block] > edge) ? &affinity[_block][C] : &_none;
                    f_weight weight = *affinity_u;
                    count new_community = (weight == -1);
                    if(_deg[_block] > edge) {
                        neighbors[_block][neighbor_count[_block]] = new_community * C;
                    } else{
                        neighbors[_block][0] = neighbors[_block][0];
                    }
                    neighbor_count[_block] += new_community;
                    weight = (1-new_community)*(weight);
                    *affinity_u = weight +  ((_deg[_block] > edge) ? sequential_edge_weight[_block] : _none);
                }
                sequential_out_edges += block_size;
                sequential_edge_weight += block_size;
            }
#if INNER_MOVE_LOG
            /*auto end_affinity = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> elapsed_affinity_time = end_affinity - start_affinity;
            affinity_time += elapsed_affinity_time.count();*/
            clock_gettime(CLOCK_REALTIME, &end_affinity);
            affinity_time += ((end_affinity.tv_sec*1000 + (end_affinity.tv_nsec/1.0e6)) - (start_affinity.tv_sec*1000 + (start_affinity.tv_nsec/1.0e6)));

//            auto start_clustering = std::chrono::high_resolution_clock::now();
            clock_gettime(CLOCK_REALTIME, &start_clustering);
#endif
            // sub-functions

            auto volCommunityMinusNode = [&](index C, node x) {
                f_weight volC = 0.0;
                f_weight volN = 0.0;
                volC = volCommunity[C];
                if (zeta[x] == C) {
                    volN = volNode[x];
                    return volC - volN;
                } else {
                    return volC;
                }
            };


            auto modGain = [&](node u, index C, index D, f_weight affinityC, f_weight affinityD) {
                f_weight volN = 0.0;
                volN = volNode[u];
                f_weight delta = (affinityD - affinityC) / total + this->gamma * ((volCommunityMinusNode(C, u) - volCommunityMinusNode(D, u)) * volN) / divisor;
                //TRACE("(" , affinity[D] , " - " , affinity[C] , ") / " , total , " + " , this->gamma , " * ((" , volCommunityMinusNode(C, u) , " - " , volCommunityMinusNode(D, u) , ") *" , volN , ") / 2 * " , (total * total));
                return delta;
            };

            // $\vol(C \ {x})$ - volume of cluster C excluding node x
            for (index _block = 0;  _block < block_size; ++_block) {

                index best = none;
                index C = none;
                f_weight deltaBest = -1;
                node u = nodes[_block];
                C = zeta[u];
                f_weight affinityC = affinity[_block][C];

//                float *deltas;
//                deltas = (float *)malloc(sizeof(float) * neighbor_count[_block]);
                for (index j=0; j<neighbor_count[_block]; ++j) {
                    index D = neighbors[_block][j];
                    if (D != C) { // consider only nodes in other clusters (and implicitly only nodes other than u)
                        f_weight delta = modGain(u, C, D, affinityC, affinity[_block][D]);
                        if (delta > deltaBest) {
                            deltaBest = delta;
                            best = D;
                        }
                    }
                }

                // TRACE("deltaBest=" , deltaBest);
                if (deltaBest > 0) { // if modularity improvement possible
                    assert(best != C && best != none);// do not "move" to original cluster
//#pragma omp atomic update
//                    move_count += 1;
                    zeta[u] = best; // move to best cluster
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
            }
#if INNER_MOVE_LOG
            /*auto end_clustering = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> elapsed_clustering_time = end_clustering - start_clustering;
            clustering_time += elapsed_clustering_time.count();*/
            clock_gettime(CLOCK_REALTIME, &end_clustering);
            clustering_time += ((end_clustering.tv_sec*1000 + (end_clustering.tv_nsec/1.0e6)) - (start_clustering.tv_sec*1000 + (start_clustering.tv_nsec/1.0e6)));
#endif
        };


        auto tryMove = [&](node u) {
            // TRACE("trying to move node " , u);
            index tid = omp_get_thread_num();
            // collect edge weight to neighbor clusters

            neigh_comm[tid].clear();
            G.forNeighborsOf(u, [&](node v) {
                turboAffinity[tid][zeta[v]] = -1; // set all to -1 so we can see when we get to it the first time
            });
            turboAffinity[tid][zeta[u]] = 0;
            G.forNeighborsOf(u, [&](node v, f_weight weight) {
                if (u != v) {
                    index C = zeta[v];
                    if (turboAffinity[tid][C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                        turboAffinity[tid][C] = 0;
                        neigh_comm[tid].push_back(C);
                    }
                    turboAffinity[tid][C] += weight;
                }
            });

            // sub-functions
            auto volCommunityMinusNode = [&](index C, node x) {
                f_weight volC = 0.0;
                f_weight volN = 0.0;
                volC = volCommunity[C];
                if (zeta[x] == C) {
                    volN = volNode[x];
                    return volC - volN;
                } else {
                    return volC;
                }
            };

            // // $\omega(u | C \ u)$
            // auto omegaCut = [&](node u, index C) {
            // 	return affinity[C];
            // };

            auto modGain = [&](node u, index C, index D, f_weight affinityC, f_weight affinityD) {
                f_weight volN = 0.0;
                volN = volNode[u];
                f_weight delta = (affinityD - affinityC) / total + this->gamma * ((volCommunityMinusNode(C, u) - volCommunityMinusNode(D, u)) * volN) / divisor;
                //TRACE("(" , affinity[D] , " - " , affinity[C] , ") / " , total , " + " , this->gamma , " * ((" , volCommunityMinusNode(C, u) , " - " , volCommunityMinusNode(D, u) , ") *" , volN , ") / 2 * " , (total * total));
                return delta;
            };

            // $\vol(C \ {x})$ - volume of cluster C excluding node x

            index best = none;
            index C = none;
            f_weight deltaBest = -1;

            C = zeta[u];
            f_weight affinityC = turboAffinity[tid][C];
            for (index D : neigh_comm[tid]) {
                if (D != C) { // consider only nodes in other clusters (and implicitly only nodes other than u)
                    f_weight delta = modGain(u, C, D, affinityC, turboAffinity[tid][D]);

                    // TRACE("mod gain: " , delta);
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
                zeta[u] = best; // move to best cluster
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

        auto remainingVertexMove = [&](node u) {
            // TRACE("trying to move node " , u);
            index tid = omp_get_thread_num();
            // collect edge weight to neighbor clusters
            count uniqueComm = 0;
            G.forNeighborsOf(u, [&](node v) {
                nodeAffinity[tid][0][zeta[v]] = -1; // set all to -1 so we can see when we get to it the first time
                node_neigh_comm[tid][0][uniqueComm++] = 0;
            });
            nodeAffinity[tid][0][zeta[u]] = 0;
            uniqueComm = 0;
            G.forNeighborsOf(u, [&](node v, f_weight weight) {
                if (u != v) {
                    index C = zeta[v];
                    if (nodeAffinity[tid][0][C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                        nodeAffinity[tid][0][C] = 0;
                        node_neigh_comm[tid][0][uniqueComm++] = C;
                    }
                    nodeAffinity[tid][0][C] += weight;
                }
            });

            // sub-functions
            auto volCommunityMinusNode = [&](index C, node x) {
                f_weight volC = 0.0;
                f_weight volN = 0.0;
                volC = volCommunity[C];
                if (zeta[x] == C) {
                    volN = volNode[x];
                    return volC - volN;
                } else {
                    return volC;
                }
            };

            // // $\omega(u | C \ u)$
            // auto omegaCut = [&](node u, index C) {
            // 	return affinity[C];
            // };

            auto modGain = [&](node u, index C, index D, f_weight affinityC, f_weight affinityD) {
                f_weight volN = 0.0;
                volN = volNode[u];
                f_weight delta = (affinityD - affinityC) / total + this->gamma * ((volCommunityMinusNode(C, u) - volCommunityMinusNode(D, u)) * volN) / divisor;
                //TRACE("(" , affinity[D] , " - " , affinity[C] , ") / " , total , " + " , this->gamma , " * ((" , volCommunityMinusNode(C, u) , " - " , volCommunityMinusNode(D, u) , ") *" , volN , ") / 2 * " , (total * total));
                return delta;
            };

            // $\vol(C \ {x})$ - volume of cluster C excluding node x

            index best = none;
            index C = none;
            f_weight deltaBest = -1;

            C = zeta[u];
            f_weight affinityC = nodeAffinity[tid][0][C];
            for (index i=0; i<uniqueComm; ++i) {
                index D = node_neigh_comm[tid][0][i];
                if (D != C) { // consider only nodes in other clusters (and implicitly only nodes other than u)
                    f_weight delta = modGain(u, C, D, affinityC, nodeAffinity[tid][0][D]);

                    // TRACE("mod gain: " , delta);
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
                zeta[u] = best; // move to best cluster
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
//        double new_mod, old_mod;
        auto movePhase = [&](){
            count iter = 0;
#if MOVE_PHASE_LOG
            auto start = std::chrono::high_resolution_clock::now();
#endif
//            std::cout<<"Move Phase Start with Max iteration: " << maxIter <<std::endl;
            do {
//                move_count = 0;
#if FIRST_MOVE_PHASE_LOG
                auto start_move = std::chrono::high_resolution_clock::now();
#endif
#if INNER_MOVE_LOG
                clustering_time = 0.0;
                    affinity_time = 0.0;
#endif
//                old_mod = modularity.getQuality(zeta, G);
                moved = false;
                // apply node movement according to parallelization strategy
                if (this->parallelism == "none") {
                    G.forNodes(tryMove);
                } else if (this->parallelism == "simple") {
                    G.parallelForNodes(tryMove);
                } else if (this->parallelism == "balanced") {
                    if(_itter == 0) {
                        frame_size = z / block_size;
//#pragma omp parallel for schedule(guided)
#pragma omp parallel for schedule(static)
                        for (node u = 0; u < frame_size; u++) {
                            vectorMove(u);
                        }
                        /*if (iter == 0) {
                            std::cout << "Nodes Remaining: " << (z - max_node) << std::endl;
                        }*/
#pragma omp parallel for schedule(guided)
                        for (int i = max_node; i < z; ++i) {
                            remainingVertexMove(i);
                        }
                    } else{
#pragma omp parallel for schedule(guided)
                        for (int u=0; u < z; ++u) {
                            tryMove(u);
                        }
                    }
                } else if (this->parallelism == "none randomized") {
                    G.forNodesInRandomOrder(tryMove);
                } else {
                    ERROR("unknown parallelization strategy: " , this->parallelism);
                    throw std::runtime_error("unknown parallelization strategy");
                }
                if (moved) change = true;

                if (iter == maxIter) {
                    WARN("move phase aborted after ", maxIter, " iterations");
                }
                iter += 1;
//                tot_move_count += move_count;
#if FIRST_MOVE_PHASE_LOG
                if(_itter == 0){
                        auto end_move = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> elapsed_move_time = end_move - start_move;
                        f_first_move_log << iter << "," << elapsed_move_time.count() << "," << zeta.numberOfSubsets() << "," << modularity.getQuality(zeta, G) << ","<< (affinity_time/1000) << "," << (clustering_time/1000) << std::endl;
                    }
#endif
//                new_mod = modularity.getQuality(zeta, G);
            } while (moved /*&& (new_mod - old_mod)>0.000001*/ && (iter < maxIter) && handler.isRunning());
            _itter++;
//            std::cout<< _itter << " iterations: " << iter << " total moves: " << tot_move_count << " avg moves: " << (tot_move_count/iter) << std::endl;
//            std::cout<<"Move Phase End with Max iteration: " << maxIter <<std::endl;
#if MOVE_PHASE_LOG
            auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed_time = end - start;
//                std::cout << "Elapsed Time for Move#"<< _itter <<": " << elapsed_time.count() << std::endl;
                f_move_log << _itter << "," << iter << "," << (elapsed_time.count()/iter) << "," << elapsed_time.count() << "," << modularity.getQuality(zeta, G) << "," << zeta.numberOfSubsets() << std::endl;
#endif

        };
        handler.assureRunning();
        // first move phase
        double old_modularity = modularity.getQuality(zeta, G);
        count old_community = zeta.numberOfSubsets();
//        struct timespec start_move_phase, end_move_phase;
        Aux::Timer timer;
//        clock_gettime(CLOCK_REALTIME, &start_move_phase);
#if FIRST_MOVE_PHASE_DETAILS_LOG
        /* struct timespec start_move_phase;
            if(_itter == 0) {
                clock_gettime(CLOCK_REALTIME, &start_move_phase);
            }*/

#endif
#if DETAILS_LOG
        //            struct timespec start_move_phase;
//            clock_gettime(CLOCK_REALTIME, &start_move_phase);
#endif
        timer.start();
        struct timespec c_start, c_end;
        clock_gettime(CLOCK_REALTIME, &c_start);
        //
        /// Reset Perf
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        ///
        movePhase();
        /// Read cache miss count
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        read(fd, &cache_miss_count, sizeof(long long));
        cache_info["move"].push_back(cache_miss_count);
        close(fd);
        ///
        //
        clock_gettime(CLOCK_REALTIME, &c_end);
        double m_time = ((c_end.tv_sec * 1000 + (c_end.tv_nsec / 1.0e6)) - (c_start.tv_sec * 1000 + (c_start.tv_nsec / 1.0e6)));
        timer.stop();
//        clock_gettime(CLOCK_REALTIME, &end_move_phase);
        elapsed_move_phase_time = m_time; //((end_move_phase.tv_sec*1000 + (end_move_phase.tv_nsec/1.0e6)) - (start_move_phase.tv_sec*1000 + (start_move_phase.tv_nsec/1.0e6)));
        std::cout<< _itter << " Wall time: " << m_time << " aux time: " << timer.elapsedMilliseconds() << std::endl;
#if FIRST_MOVE_PHASE_DETAILS_LOG
        if(_itter == 1) {
//                struct timespec end_move_phase;
//                clock_gettime(CLOCK_REALTIME, &end_move_phase);
//                elapsed_move_phase_time = ((end_move_phase.tv_sec*1000 + (end_move_phase.tv_nsec/1.0e6)) - (start_move_phase.tv_sec*1000 + (start_move_phase.tv_nsec/1.0e6)));
        }
#endif
#if DETAILS_LOG
        //            struct timespec end_move_phase;
//            clock_gettime(CLOCK_REALTIME, &end_move_phase);
//            elapsed_move_phase_time = ((end_move_phase.tv_sec*1000 + (end_move_phase.tv_nsec/1.0e6)) - (start_move_phase.tv_sec*1000 + (start_move_phase.tv_nsec/1.0e6)));
#endif
        count new_community = zeta.numberOfSubsets();
        double new_modularity = modularity.getQuality(zeta, G);
        /// Compare the modularity before and after move phase
        /*if(_itter > 1 && (new_modularity - old_modularity) < 0.000001 *//*old_modularity == new_modularity*//*){
            change = false;
        }*/
//        std::cout << "Modularity of the cluster: " << new_modularity << std::endl;

//        timing["move"].push_back(timer.elapsedMilliseconds());
        timing["move"].push_back(m_time);
        handler.assureRunning();
        if (recurse && change) {
            DEBUG("nodes moved, so begin coarsening and recursive call");

//            timer.start();
            //
//            struct timespec start_coarsen, end_coarsen;
//            clock_gettime(CLOCK_REALTIME, &start_coarsen);
#if FIRST_MOVE_PHASE_DETAILS_LOG
            /* struct timespec start_coarsen, end_coarsen;
                if(_itter == 1) {
                    clock_gettime(CLOCK_REALTIME, &start_coarsen);
                }*/
#endif
            clock_gettime(CLOCK_REALTIME, &c_start);
            std::pair<Graph, std::vector<node>> coarsened = coarsen(G, zeta);	// coarsen graph according to communitites
            clock_gettime(CLOCK_REALTIME, &c_end);
            double coarsen_time = ((c_end.tv_sec * 1000 + (c_end.tv_nsec / 1.0e6)) - (c_start.tv_sec * 1000 + (c_start.tv_nsec / 1.0e6)));
//            clock_gettime(CLOCK_REALTIME, &end_coarsen);
            elapsed_coarsen_time = coarsen_time; //((end_coarsen.tv_sec*1000 + (end_coarsen.tv_nsec/1.0e6)) - (start_coarsen.tv_sec*1000 + (start_coarsen.tv_nsec/1.0e6)));
#if FIRST_MOVE_PHASE_DETAILS_LOG
            if(_itter == 1) {
//                    clock_gettime(CLOCK_REALTIME, &end_coarsen);
//                    elapsed_coarsen_time = ((end_coarsen.tv_sec*1000 + (end_coarsen.tv_nsec/1.0e6)) - (start_coarsen.tv_sec*1000 + (start_coarsen.tv_nsec/1.0e6)));
                f_first_move_phase_details_log << elapsed_coloring_time << "," << elapsed_sorting_time << "," << elapsed_reformatting_time << "," << elapsed_initialization_time << "," << elapsed_move_phase_time << "," << elapsed_coarsen_time << std::endl;

            }
#endif
#if DETAILS_LOG
            f_ovpl_details_log << _itter << "," << elapsed_move_phase_time << "," << elapsed_coarsen_time << "," << old_modularity << "," << new_modularity << "," << old_community << "," << new_community << std::endl;
#endif
            //
//            timer.stop();
//            timing["coarsen"].push_back(timer.elapsedMilliseconds());
            timing["coarsen"].push_back(coarsen_time);

            OVPL onCoarsened(coarsened.first, this->refine, this->gamma, this->parallelism, this->maxIter, this->turbo);
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


            DEBUG("coarse graph has ", coarsened.first.numberOfNodes(), " nodes and ", coarsened.first.numberOfEdges(), " edges");
            struct timespec start_prolong, end_prolong;
            clock_gettime(CLOCK_REALTIME, &start_prolong);
            zeta = prolong(coarsened.first, zetaCoarse, G, coarsened.second); // unpack communities in coarse graph onto fine graph
            clock_gettime(CLOCK_REALTIME, &end_prolong);
            double elapsed_prolong_time = ((end_prolong.tv_sec*1000 + (end_prolong.tv_nsec/1.0e6)) - (start_prolong.tv_sec*1000 + (start_prolong.tv_nsec/1.0e6)));
//            std::cout<<"Prolong Time: "<<elapsed_prolong_time<<std::endl;
            // refinement phase
            if (refine) {
                DEBUG("refinement phase");
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
                clock_gettime(CLOCK_REALTIME, &c_end);
                double ref_time = ((c_end.tv_sec * 1000 + (c_end.tv_nsec / 1.0e6)) - (c_start.tv_sec * 1000 + (c_start.tv_nsec / 1.0e6)));
//                timer.stop();
//                std::cout << "Elapsed Time Per Move: " << timer.elapsedMilliseconds() << std::endl;
//                timing["refine"].push_back(timer.elapsedMilliseconds());
                timing["refine"].push_back(ref_time);

            }
        } else {
#if DETAILS_LOG
            f_ovpl_details_log << _itter << "," << elapsed_move_phase_time << "," << elapsed_coarsen_time << "," << old_modularity << "," << new_modularity << "," << old_community << "," << new_community << std::endl;
#endif
        }
        result = std::move(zeta);
#if MOVE_PHASE_LOG
        f_move_log.close();
#endif
#if FIRST_MOVE_PHASE_LOG
        f_first_move_log.close();
#endif
#if FIRST_MOVE_PHASE_DETAILS_LOG
        f_first_move_phase_details_log.close();
#endif
#if DETAILS_LOG
        f_ovpl_details_log.close();
#endif
        hasRun = true;
    }

    std::string NetworKit::OVPL::toString() const {
        std::stringstream stream;
        stream << "OVPL(";
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

    std::pair<Graph, std::vector<node> > OVPL::coarsen(const Graph& G, const Partition& zeta) {
        ParallelPartitionCoarsening parCoarsening(G, zeta);
        parCoarsening.run();
        return {parCoarsening.getCoarseGraph(),parCoarsening.getFineToCoarseNodeMapping()};
    }

    Partition OVPL::prolong(const Graph& Gcoarse, const Partition& zetaCoarse, const Graph& Gfine, std::vector<node> nodeToMetaNode) {
        Partition zetaFine(Gfine.upperNodeIdBound());
        zetaFine.setUpperBound(zetaCoarse.upperBound());

        Gfine.forNodes([&](node v) {
            node mv = nodeToMetaNode[v];
            index cv = zetaCoarse[mv];
            zetaFine[v] = cv;
        });


        return zetaFine;
    }

    std::map<std::string, std::vector<double > > OVPL::getTiming() {
        return timing;
    }

    std::map<std::string, std::vector<long long > > OVPL::getCacheCount() {
        return cache_info;
    }

} /* namespace NetworKit */