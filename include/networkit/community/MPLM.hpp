/*
 * ModifiedPLM.h
 *
 *  Created on: 20.11.2013
 *      Author: cls
 */

#ifndef ModifiedPLM_H_
#define ModifiedPLM_H_

#include "CommunityDetectionAlgorithm.h"

#include <fstream>

namespace NetworKit {

/**
 * @ingroup community
 * Parallel Louvain Method - a multi-level modularity maximizer.
 */
class ModifiedPLM: public NetworKit::CommunityDetectionAlgorithm {

/*private:
    int move_iter = 0;
    std::string graphName = "";
    int _thread = 0;*/
public:
	/**
	 * @param[in]	G	input graph
	 * @param[in]	refine	add a second move phase to refine the communities
	 * @param[in]	par		parallelization strategy
	 * @param[in]	gamma	multi-resolution modularity parameter:
	 * 							1.0 -> standard modularity
	 * 							0.0 -> one community
	 * 							2m 	-> singleton communities
	 * @param[in]	maxIter		maximum number of iterations for move phase
	 * @param[in]	parallelCoarsening	use parallel graph coarsening
	 * @param[in]	turbo	faster but uses O(n) additional memory per thread
	 * @param[in]	recurse	use recursive coarsening, see http://journals.aps.org/pre/abstract/10.1103/PhysRevE.89.049902 for some explanations (default: true)
	 *
	 */
	ModifiedPLM(const Graph& G, bool refine=false, f_weight gamma = 1.0, std::string par="balanced", count maxIter=32, bool turbo = true, bool recurse = true);

	ModifiedPLM(const Graph& G, const ModifiedPLM& other);

	void initMPLM();

    long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags);
	/**
	 * Get string representation.
	 *
	 * @return String representation of this algorithm.
	 */
	std::string toString() const override;

	/**
	 * Detect communities.
	 */
	void run() override;

	void setupMoveDetailsCSVFile(std::string move_details);

	void setupMPLMPowerFile(std::string graphName, count threads);

	static std::pair<Graph, std::vector<node>> coarsen(const Graph& G, const Partition& zeta);

	static Partition prolong(const Graph& Gcoarse, const Partition& zetaCoarse, const Graph& Gfine, std::vector<node> nodeToMetaNode);

	/**
	 * Returns fine-grained running time measurements for algorithm engineering purposes.
	 */
	std::map<std::string, std::vector<double > > getTiming();
	std::map<std::string, std::vector<long long > > getCacheCount();

private:

	std::string parallelism;
	bool refine;
	f_weight gamma = 1.0;
	count maxIter;
	bool turbo;
	bool recurse;
    std::string graphName = "";
    int _thread = 0;
	std::map<std::string, std::vector<double > > timing;	 // fine-grained running time measurement
	std::map<std::string, std::vector<long long > > cache_info;	 // fine-grained running time measurement
};

} /* namespace NetworKit */

#endif /* PLM_H_ */
