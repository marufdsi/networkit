/*
 * OPLM.hpp
 *
 *  Created on: 20.11.2013
 *      Author: cls
 */

#ifndef OPLM_H_
#define OPLM_H_

#include <networkit/community/CommunityDetectionAlgorithm.hpp>

#include <fstream>

namespace NetworKit {

/**
 * @ingroup community
 * Parallel Louvain Method - a multi-level modularity maximizer.
 */
class OPLM final: public CommunityDetectionAlgorithm {

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
	OPLM(const Graph& G, bool refine=false, f_weight gamma = 1.0, std::string par="balanced", count maxIter=32, bool turbo = true, bool recurse = true);

	OPLM(const Graph& G, const OPLM& other);

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

	static std::pair<Graph, std::vector<node>> coarsen(const Graph& G, const Partition& zeta);

	static Partition prolong(const Graph& Gcoarse, const Partition& zetaCoarse, const Graph& Gfine, std::vector<node> nodeToMetaNode);

	/**
	 * Returns fine-grained running time measurements for algorithm engineering purposes.
	 */
	std::map<std::string, std::vector<double > > getTiming();

private:

	std::string parallelism;
	bool refine;
	f_weight gamma = 1.0;
	count maxIter;
	bool turbo;
	bool recurse;
	std::map<std::string, std::vector<double > > timing;	 // fine-grained running time measurement
};

} /* namespace NetworKit */

#endif /* PLM_H_ */
