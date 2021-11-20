/*
 * MPLM.cpp
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

#include <networkit/MPLM.hpp>
#include <omp.h>
#include <networkit/community/Modularity.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/coarsening/ClusteringProjector.hpp>
#include <networkit/coarsening/ParallelPartitionCoarsening.hpp>
#include <networkit/community/PLM.hpp>
#include <networkit/auxiliary/PowerCalculator.hpp>

#include<time.h>
#include <sys/time.h>

#include <sstream>

#define ONE_THREAD false
#define MOVE_DETAILS false
#define POWER_LOG false

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

//#ifndef L1D_CACHE_MISS_COUNT
//#define L1D_CACHE_MISS_COUNT
//#endif

int move_iter = 0;

namespace NetworKit {

    MPLM::MPLM(const Graph &G, bool refine, f_weight gamma, std::string par, count maxIter, bool turbo,
                             bool recurse) : CommunityDetectionAlgorithm(G), parallelism(par), refine(refine),
                                             gamma(gamma), maxIter(maxIter), turbo(turbo), recurse(recurse) {

    }

    MPLM::MPLM(const Graph &G, const MPLM &other) : CommunityDetectionAlgorithm(G),
                                                                         parallelism(other.parallelism),
                                                                         refine(other.refine), gamma(other.gamma),
                                                                         maxIter(other.maxIter), turbo(other.turbo),
                                                                         recurse(other.recurse) {

    }

    void MPLM::initMPLM() {
        move_iter = 0;
    }

    /*long MPLM::perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags){
        int ret;
        ret = syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
        return ret;
    }*/

    std::ofstream f_move_plm_details;

    void MPLM::setupMoveDetailsCSVFile(std::string move_details) {
#if MOVE_DETAILS
        std::ifstream infile(move_details);
        bool existing_file = infile.good();
        f_move_plm_details.open(move_details, std::ios_base::out | std::ios_base::app | std::ios_base::ate);
        if (!existing_file) {
            f_move_plm_details << "Move Phase" << "," << "Move" << "," << "Vertices" << "," << "Edges" << "," << "Move Time" << std::endl;
        }
#endif
    }

    void MPLM::setupMPLMPowerFile(std::string _graphName, count threads) {
#if POWER_LOG
        graphName = _graphName;
        _thread = threads;
#endif
    }

    void MPLM::run() {
        Aux::SignalHandler handler;
        Modularity modularity;
        /// perf event attributes
//        struct perf_event_attr pe;
//        long long cache_miss_count;
//        int fd;
        ///
//        DEBUG("calling run method on ", G->toString());
#if POWER_LOG
        //        if(move_iter == 0)
                    rapl_sysfs_init();
#endif

        uint64_t conflict_community = 0, reiterate_conflict = 0;
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
//        zeta.allToSingletons();
        zeta.allToSeqSingletons();
        index o = zeta.upperBound();

        // init graph-dependent temporaries
        std::vector<f_weight> volNode(z, 0.0);
        // $\omega(E)$
        f_weight total = G->totalEdgeWeight();
//        DEBUG("total edge weight: ", total);
        f_weight divisor = (2 * total * total); // needed in modularity calculation

        std::vector<count> outDegree;
        const std::vector<f_weight> *outEdgeWeights;
        const std::vector<node> *outEdges;
        bool isGraphWeighted = G->isWeighted();

//        outDegree = G->getOutDegree();
        outEdgeWeights = G->getOutEdgeWeights();
        outEdges = G->getOutEdges();
        for (int i=0; i<z; ++i) {
            outDegree.push_back(outEdges[i].size());
        }

        index max_tid = omp_get_max_threads();
        index max_deg_arr[max_tid];

#pragma omp parallel
        {
            index tid = omp_get_thread_num();
            max_deg_arr[tid] = max_deg_of_graph;
#pragma omp for schedule(static)
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
        zeta.parallelForEntries([&](node u, index C) {    // set volume for all communities
            if (C != none)
                volCommunity[C] = volNode[u];
        });

        // first move phase
        bool moved = false; // indicates whether any node has been moved in the last pass
        bool change = false; // indicates whether the communities have changed at all

        // stores the affinity for each neighboring community (index), one vector per thread
//        std::vector<std::vector<f_weight> > turboAffinity;
        f_weight **turboAffinity = (f_weight **) malloc(max_tid * sizeof(f_weight *));
        // stores the list of neighboring communities, one vector per thread
        index **neigh_comm = (index **) malloc(max_tid * sizeof(index *));
//                turboAffinity.resize(max_tid);
//                neigh_comm.resize(max_tid);
#pragma omp for schedule(static)
        for (int i = 0; i < max_tid; ++i) {
//                    turboAffinity[i].resize(zeta.upperBound());
//                    neigh_comm[i].resize(max_deg_of_graph);
            posix_memalign((void **) &turboAffinity[i], alignment, zeta.upperBound() * sizeof(f_weight));
            posix_memalign((void **) &neigh_comm[i], alignment, max_deg_of_graph * sizeof(index));
        }

        /// Initialize perf events
        /*memset(&pe, 0, sizeof(struct perf_event_attr));
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

        fd = perf_event_open(&pe, 0, -1, -1, 0);
        if (fd == -1) {
            fprintf(stderr, "Error opening leader %llx\n", pe.config);
            exit(EXIT_FAILURE);
        }*/
        ///

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

            for (index edge = 0; edge < _deg; ++edge) {
                pnt_affinity[zeta[pnt_outEdges[edge]]] = -1.0;
            }
            pnt_affinity[zeta[u]] = 0;

            for (int i = 0; i < _deg; ++i) {
                node v = pnt_outEdges[i];
                if (u != v) {
                    index C = zeta[v];
                    if (pnt_affinity[C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                        pnt_affinity[C] = 0;
                        pnt_neigh_comm[neigh_counter++] = C;
                    }
                    pnt_affinity[C] += isGraphWeighted ? pnt_outEdgeWeight[i] : fdefaultEdgeWeight;
                }
            }

            index best = none;
            f_weight deltaBest = -1;

            index C = zeta[u];
            f_weight affinityC = pnt_affinity[C];
            f_weight volN = volNode[u];
            f_weight volCommunityMinusNode_C = volCommunity[C] - volN;

            for (index i = 0; i < neigh_counter; ++i) {
                index D = pnt_neigh_comm[i];
                if (D != C) { // consider only nodes in other clusters (and implicitly only nodes other than u)
                    f_weight delta = (pnt_affinity[D] - affinityC) / total +
                                     this->gamma * ((volCommunityMinusNode_C - volCommunity[D]) * volN) / divisor;
                    if (delta > deltaBest) {
                        deltaBest = delta;
                        best = D;
                    }
                }
            }

            // TRACE("deltaBest=" , deltaBest);
            if (deltaBest > 0) { // if modularity improvement possible
//#pragma omp atomic update
//                move_count += 1;
                assert(best != C && best != none);// do not "move" to original cluster
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
        struct timespec start_move, end_move;
        auto movePhase = [&]() {
            /*if(move_iter == 0){
                count edge_count = 0;
                for (int e = 0; e < z; ++e) {
                    edge_count += outDegree[e];
                }
                std::cout<< "edge count : "  << edge_count << std::endl;
            }*/
            count iter = 0;
#if POWER_LOG
            //            if(move_iter == 0)
                            rapl_sysfs_before();
#endif
            do {
//                move_count = 0;
#if MOVE_DETAILS
                clock_gettime(CLOCK_REALTIME, &start_move);
#endif
                clock_gettime(CLOCK_REALTIME, &start_move);
                moved = false;
//                old_mod = modularity.getQuality(zeta, G);
                // apply node movement according to parallelization strategy
                if (this->parallelism == "none") {
                    G->forNodes(tryMove);
                } else if (this->parallelism == "simple") {
                    G->parallelForNodes(tryMove);
                } else if (this->parallelism == "balanced") {
//                    G->balancedParallelForNodes(tryMove);
                    #pragma omp parallel for schedule(guided)
                    for (index u = 0; u < z; ++u) {
//                        if(G->hasNode(u))
                            tryMove(u);
                    }
                } else if (this->parallelism == "none randomized") {
                    G->forNodesInRandomOrder(tryMove);
                } else {
                    ERROR("unknown parallelization strategy: ", this->parallelism);
                    throw std::runtime_error("unknown parallelization strategy");
                }
                if (moved) change = true;

                /*if (iter == maxIter) {
                    WARN("move phase aborted after ", maxIter, " iterations");
                }*/
//                std::cout<<"[" << move_iter << "] " << "Move: " << iter << std::endl;
                if (move_iter == 0) {
                    clock_gettime(CLOCK_REALTIME, &end_move);
                    double elapsed_move_time = ((end_move.tv_sec * 1000 + (end_move.tv_nsec / 1.0e6)) -
                                                (start_move.tv_sec * 1000 + (start_move.tv_nsec / 1.0e6)));
//                    std::cout << "[" << iter << "] time: " << elapsed_move_time << " for z= " << z << std::endl;
                }
                iter += 1;
//                tot_move_count += move_count;

#if MOVE_DETAILS
                clock_gettime(CLOCK_REALTIME, &end_move);
                double elapsed_move_time = ((end_move.tv_sec*1000 + (end_move.tv_nsec/1.0e6)) - (start_move.tv_sec*1000 + (start_move.tv_nsec/1.0e6)));
                f_move_plm_details<< move_iter << "," << iter << "," << z << "," << G->numberOfEdges() << "," << elapsed_move_time <<std::endl;
#endif
//                new_mod = modularity.getQuality(zeta, G);
            } while (moved /*&& (new_mod - old_mod)>0.000001*/  && (iter < maxIter) && handler.isRunning());
//            DEBUG("iterations in move phase: ", iter);
#if POWER_LOG
            /*if(move_iter == 0)*/ {
                // End the calculate of power
                rapl_sysfs_after();
                // Results
                rapl_sysfs_results(Modified Legacy", graphName, _thread, move_iter+1);
            }
#endif
            if (move_iter == 0) {
                std::cout << "total move iteration: " << iter << std::endl;
            }
            move_iter++;
//            std::cout<< move_iter << " iterations: " << iter << " total moves: " << tot_move_count << " avg moves: "
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
        /// Reset Perf
//        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
//        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        ///
        movePhase();
        /// Read cache miss count
//        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
//        read(fd, &cache_miss_count, sizeof(long long));
//        cache_info["move"].push_back(cache_miss_count);
//        close(fd);
        ///
        //
        timer.stop();
        clock_gettime(CLOCK_REALTIME, &c_end);
        double m_time = ((c_end.tv_sec * 1000 + (c_end.tv_nsec / 1.0e6)) -
                         (c_start.tv_sec * 1000 + (c_start.tv_nsec / 1.0e6)));
        std::cout<< move_iter << " Wall time: " << m_time << " aux time: " << timer.elapsedMilliseconds() << std::endl;
//        timing["move"].push_back(timer.elapsedMilliseconds());
        timing["move"].push_back(m_time);
//        std::cout<< "Phase: " << move_iter << " Time: " << (double)(c_end - c_start) / CLOCKS_PER_SEC * 1000 << std::endl;
        double new_modularity = modularity.getQuality(zeta, *G);
        handler.assureRunning();
        /*if(move_iter > 1 && (new_modularity - old_modularity)<0.000001)
            change = false;*/
        if (recurse && change) {
//            DEBUG("nodes moved, so begin coarsening and recursive call");
            clock_gettime(CLOCK_REALTIME, &c_start);
//            timer.start();
            //
            std::pair<Graph, std::vector<node>> coarsened = coarsen(G,
                                                                    zeta);    // coarsen graph according to communitites
            //
            clock_gettime(CLOCK_REALTIME, &c_end);
            double coarsen_time = ((c_end.tv_sec * 1000 + (c_end.tv_nsec / 1.0e6)) -
                                   (c_start.tv_sec * 1000 + (c_start.tv_nsec / 1.0e6)));
//            timer.stop();
//            timing["coarsen"].push_back(timer.elapsedMilliseconds());
            timing["coarsen"].push_back(coarsen_time);

            MPLM onCoarsened(coarsened.first, this->refine, this->gamma, this->parallelism, this->maxIter,
                                    this->turbo);
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
            zeta = prolong(coarsened.first, zetaCoarse, *G,
                           coarsened.second); // unpack communities in coarse graph onto fine graph
            // refinement phase
            if (refine) {
//                DEBUG("refinement phase");
                // reinit community-dependent temporaries
                o = zeta.upperBound();
                volCommunity.clear();
                volCommunity.resize(o, 0.0);
                zeta.parallelForEntries([&](node u, index C) {    // set volume for all communities
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
                double ref_time = ((c_end.tv_sec * 1000 + (c_end.tv_nsec / 1.0e6)) -
                                   (c_start.tv_sec * 1000 + (c_start.tv_nsec / 1.0e6)));
//                timer.stop();
//                timing["refine"].push_back(timer.elapsedMilliseconds());
                timing["refine"].push_back(ref_time);

            }
        }
        result = std::move(zeta);
        hasRun = true;
    }

    std::string NetworKit::MPLM::toString() const {
        std::stringstream stream;
        stream << "MPLM(";
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

    std::pair<Graph, std::vector<node> > MPLM::coarsen(const Graph &G, const Partition &zeta) {
        ParallelPartitionCoarsening parCoarsening(*G, zeta);
        parCoarsening.run();
        return {parCoarsening.getCoarseGraph(), parCoarsening.getFineToCoarseNodeMapping()};
    }

    Partition MPLM::prolong(const Graph &Gcoarse, const Partition &zetaCoarse, const Graph &Gfine,
                                   std::vector<node> nodeToMetaNode) {
        Partition zetaFine(Gfine.upperNodeIdBound());
        zetaFine.setUpperBound(zetaCoarse.upperBound());

        Gfine.forNodes([&](node v) {
            node mv = nodeToMetaNode[v];
            index cv = zetaCoarse[mv];
            zetaFine[v] = cv;
        });


        return zetaFine;
    }


    std::map<std::string, std::vector<double> > MPLM::getTiming() {
        return timing;
    }

    std::map<std::string, std::vector<long long > > MPLM::getCacheCount() {
        return cache_info;
    }

} /* namespace NetworKit */