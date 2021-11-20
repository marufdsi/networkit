/*
 * MLPLM.cpp
 *
 *  Created on: 20.11.2013
 *      Author: cls
 */

#include <networkit/community/OPLM.hpp>
#include <iostream>
#include <omp.h>

#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/auxiliary/Timer.hpp>
#include <networkit/coarsening/ClusteringProjector.hpp>
#include <networkit/coarsening/ParallelPartitionCoarsening.hpp>
#include <networkit/community/OPLM.hpp>
#include <networkit/community/Modularity.hpp>
#include<time.h>
#include <sys/time.h>


#include <sstream>

#define ONE_THREAD false
int ori_move_iter = 0;
namespace NetworKit {

    OriginalPLM::OriginalPLM(const Graph& G, bool refine, f_weight gamma, std::string par, count maxIter, bool turbo, bool recurse) : CommunityDetectionAlgorithm(G), parallelism(par), refine(refine), gamma(gamma), maxIter(maxIter), turbo(turbo), recurse(recurse) {

    }

    OriginalPLM::OriginalPLM(const Graph& G, const OriginalPLM& other) : CommunityDetectionAlgorithm(G), parallelism(other.parallelism), refine(other.refine), gamma(other.gamma), maxIter(other.maxIter), turbo(other.turbo), recurse(other.recurse) {

    }


    void OriginalPLM::run() {
        Aux::SignalHandler handler;
        Modularity modularity;
        DEBUG("calling run method on " , G->toString());
        #if ONE_THREAD
            omp_set_dynamic(0);
            omp_set_num_threads(1);
        #endif
        count z = G->upperNodeIdBound();


        // init communities to singletons
        Partition zeta(z);
        zeta.allToSingletons();
        index o = zeta.upperBound();

        // init graph-dependent temporaries
        std::vector<f_weight > volNode(z, 0.0);
        // $\omega(E)$
        f_weight total = G->totalEdgeWeight();
        DEBUG("total edge weight: " , total);
        f_weight divisor = (2 * total * total); // needed in modularity calculation

        G->parallelForNodes([&](node u) { // calculate and store volume of each node
            volNode[u] += G->weightedDegree(u);
            volNode[u] += G->weight(u, u); // consider self-loop twice
            // TRACE("init volNode[" , u , "] to " , volNode[u]);
        });

        // init community-dependent temporaries
        std::vector<f_weight > volCommunity(o, 0.0);
        zeta.parallelForEntries([&](node u, index C) { 	// set volume for all communities
            if (C != none)
                volCommunity[C] = volNode[u];
        });

        // first move phase
        bool moved = false; // indicates whether any node has been moved in the last pass
        bool change = false; // indicates whether the communities have changed at all

        // stores the affinity for each neighboring community (index), one vector per thread
        std::vector<std::vector<f_weight> > turboAffinity;
        // stores the list of neighboring communities, one vector per thread
        std::vector<std::vector<index> > neigh_comm;


        turboAffinity.resize(omp_get_max_threads());
        neigh_comm.resize(omp_get_max_threads());
#pragma omp for schedule(static)
        for (int i=0; i<omp_get_max_threads(); ++i) {
            // resize to maximum community id
            turboAffinity[i].resize(zeta.upperBound());
        }
        /*for (auto &it : turboAffinity) {
            // resize to maximum community id
            it.resize(zeta.upperBound());
        }*/

        // try to improve modularity by moving a node to neighboring clusters
        auto tryMove = [&](node u) {
            // TRACE("trying to move node " , u);
            index tid = omp_get_thread_num();
            neigh_comm[tid].clear();
            const node* outEdges = G->getOutEdge(u);
            for (int i = 0; i < G->degree(u); ++i) {
                turboAffinity[tid][zeta[outEdges[i]]] = -1;
            }
            /*G->forNeighborsOf(u, [&](node v) {
                turboAffinity[tid][zeta[v]] = -1; // set all to -1 so we can see when we get to it the first time
            });*/
            turboAffinity[tid][zeta[u]] = 0;
            for (int i = 0; i < G->degree(u); ++i) {
                node v = outEdges[i];
                if (u != v) {
                    index C = zeta[v];
                    if (turboAffinity[tid][C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                        turboAffinity[tid][C] = 0;
                        neigh_comm[tid].push_back(C);
                    }
                    turboAffinity[tid][C] += G->isWeighted() ? G->getOutEdgeWeight<true>(u, i) : fdefaultEdgeWeight;
                }
            }
            /*G->forNeighborsOf(u, [&](node v, f_weight weight) {
                if (u != v) {
                    index C = zeta[v];
                    if (turboAffinity[tid][C] == -1) {
                        // found the neighbor for the first time, initialize to 0 and add to list of neighboring communities
                        turboAffinity[tid][C] = 0;
                        neigh_comm[tid].push_back(C);
                    }
                    turboAffinity[tid][C] += weight;
                }
            });*/
            // sub-functions
            // $\vol(C \ {x})$ - volume of cluster C excluding node x
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

            index best = none;
            index C = none;
            f_weight deltaBest = -1;
            C = zeta[u];
            f_weight affinityC = turboAffinity[tid][C];
            f_weight volN = volNode[u];
            f_weight volCommunityMinusNode_C = volCommunity[C] - volN;
            for (index D : neigh_comm[tid]) {
                if (D != C) { // consider only nodes in other clusters (and implicitly only nodes other than u)
//                    f_weight delta = modGain(u, C, D, affinityC, turboAffinity[tid][D]);
                    f_weight delta = (turboAffinity[tid][D] - affinityC) / total + this->gamma * ((volCommunityMinusNode_C - volCommunity[D]) * volN) / divisor;
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
        double new_mod, old_mod;
        auto movePhase = [&](){
            count iter = 0;
            do {
                moved = false;
                old_mod = modularity.getQuality(zeta, *G);
                // apply node movement according to parallelization strategy
                if (this->parallelism == "none") {
                    G->forNodes(tryMove);
                } else if (this->parallelism == "simple") {
                    G->parallelForNodes(tryMove);
                } else if (this->parallelism == "balanced") {
//                    G->balancedParallelForNodes(tryMove);
#pragma omp parallel for schedule(static)
                    for (index u = 0; u < z; ++u) {
                        if(G->hasNode(u))
                            tryMove(u);
                    }
                } else if (this->parallelism == "none randomized") {
                    G->forNodesInRandomOrder(tryMove);
                } else {
                    ERROR("unknown parallelization strategy: " , this->parallelism);
                    throw std::runtime_error("unknown parallelization strategy");
                }
                if (moved) change = true;

                if (iter == maxIter) {
                    WARN("move phase aborted after ", maxIter, " iterations");
                }
                iter += 1;
                new_mod = modularity.getQuality(zeta, *G);
            } while (moved  /*&& (new_mod - old_mod)>0.000001 */&& (iter < maxIter) && handler.isRunning());
            DEBUG("iterations in move phase: ", iter);
            ori_move_iter++;
        };
        handler.assureRunning();
        double old_modularity = modularity.getQuality(zeta, *G);
        // first move phase
//        Aux::Timer timer;
        struct timespec c_start, c_end;
        clock_gettime(CLOCK_REALTIME, &c_start);
//        timer.start();
        //
        movePhase();
        //
//        timer.stop();
        clock_gettime(CLOCK_REALTIME, &c_end);
        double m_time = ((c_end.tv_sec * 1000 + (c_end.tv_nsec / 1.0e6)) - (c_start.tv_sec * 1000 + (c_start.tv_nsec / 1.0e6)));
//        timing["move"].push_back(timer.elapsedMilliseconds());
        timing["move"].push_back(m_time);
//        std::cout<< "Phase: " << ori_move_iter << " Time: " << (double)(c_end - c_start) / CLOCKS_PER_SEC * 1000 << std::endl;
        double new_modularity = modularity.getQuality(zeta, *G);
        handler.assureRunning();
        /*if(ori_move_iter > 1 && (new_modularity - old_modularity)<0.000001)
            change = false;*/
        if (recurse && change) {
            DEBUG("nodes moved, so begin coarsening and recursive call");
            clock_gettime(CLOCK_REALTIME, &c_start);
//            timer.start();
            //
            std::pair<Graph, std::vector<node>> coarsened = coarsen(*G, zeta);	// coarsen graph according to communitites
            //
//            timer.stop();
            clock_gettime(CLOCK_REALTIME, &c_end);
            double coarsen_time = ((c_end.tv_sec * 1000 + (c_end.tv_nsec / 1.0e6)) - (c_start.tv_sec * 1000 + (c_start.tv_nsec / 1.0e6)));
//            timing["coarsen"].push_back(timer.elapsedMilliseconds());
            timing["coarsen"].push_back(coarsen_time);

            OriginalPLM onCoarsened(coarsened.first, this->refine, this->gamma, this->parallelism, this->maxIter, this->turbo);
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
            zeta = prolong(coarsened.first, zetaCoarse, *G, coarsened.second); // unpack communities in coarse graph onto fine graph
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
//                timing["refine"].push_back(timer.elapsedMilliseconds());
                timing["refine"].push_back(ref_time);

            }
        }
        result = std::move(zeta);
        hasRun = true;
    }

    std::string NetworKit::OriginalPLM::toString() const {
        std::stringstream stream;
        stream << "OriginalPLM(";
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

    std::pair<Graph, std::vector<node> > OriginalPLM::coarsen(const Graph& G, const Partition& zeta) {
        ParallelPartitionCoarsening parCoarsening(G, zeta);
        parCoarsening.run();
        return {parCoarsening.getCoarseGraph(),parCoarsening.getFineToCoarseNodeMapping()};
    }

    Partition OriginalPLM::prolong(const Graph& Gcoarse, const Partition& zetaCoarse, const Graph& Gfine, std::vector<node> nodeToMetaNode) {
        Partition zetaFine(Gfine.upperNodeIdBound());
        zetaFine.setUpperBound(zetaCoarse.upperBound());

        Gfine.forNodes([&](node v) {
            node mv = nodeToMetaNode[v];
            index cv = zetaCoarse[mv];
            zetaFine[v] = cv;
        });


        return zetaFine;
    }



    std::map<std::string, std::vector<double > > OriginalPLM::getTiming() {
        return timing;
    }

} /* namespace NetworKit */