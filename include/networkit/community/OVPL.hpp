/*
 * OVPL.hpp
 *
 *  Created on: 10.10.2018
 *      Author: Md Maruf Hossain
 */

#ifndef OVPL_H_
#define OVPL_H_

#include <networkit/community/CommunityDetectionAlgorithm.hpp>

#include <fstream>

namespace NetworKit {

/**
 * @ingroup community
 * Parallel Louvain Method - a multi-level modularity maximizer.
 */
class OVPL final: public CommunityDetectionAlgorithm {

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
	OVPL(const Graph& G, bool refine=false, f_weight gamma = 1.0, std::string par="balanced", count maxIter=32, bool turbo = true, bool recurse = true);

	OVPL(const Graph& G, const OVPL& other);


//    long perf_event_open2(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags);
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

	void setupCSVFile(std::string move_log_file, std::string first_move_log_file, std::string first_move_phase_details_log, std::string ovpl_details_log);

	static std::pair<Graph, std::vector<node>> coarsen(const Graph& G, const Partition& zeta);

	static Partition prolong(const Graph& Gcoarse, const Partition& zetaCoarse, const Graph& Gfine, std::vector<node> nodeToMetaNode);

	/**
	 * Returns fine-grained running time measurements for algorithm engineering purposes.
	 */
	std::map<std::string, std::vector<double > > getTiming();
    std::map<std::string, std::vector<long long > > getCacheCount();
    std::map<std::string, std::vector<long > > getGraphBlockInfo();

private:

	std::string parallelism;
	bool refine;
	f_weight gamma = 1.0;
	count maxIter;
	bool turbo;
	bool recurse;
	std::map<std::string, std::vector<double > > timing;	 // fine-grained running time measurement
    std::map<std::string, std::vector<long long > > cache_info;	 // fine-grained running time measurement

    std::map<std::string, std::vector<long > > graph_block_info;	 // fine-grained running time measurement
};

} /* namespace NetworKit */

#endif /* OVPL_H_ */
