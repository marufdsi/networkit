// no-networkit-format
/*
 * Unittests-X.cpp
 *
 *  Updated: 2021
 *      Editor: Md Maruf Hossain
 *
 */

#include <algorithm>
#include <iostream>
#include <omp.h>
#include <utility>

#include <gtest/gtest.h>

#include <tlx/cmdline_parser.hpp>

#include <networkit/GlobalState.hpp>
#include <networkit/auxiliary/Log.hpp>
#include <networkit/auxiliary/Parallelism.hpp>
#include <networkit/auxiliary/PowerCalculator.hpp>
#include <networkit/generators/RmatGenerator.hpp>
#include <networkit/community/PLP.hpp>
#include <networkit/community/MPLP.hpp>
#include <networkit/community/ONLP.hpp>
#include <networkit/community/PLM.hpp>
#include <networkit/community/ONPL.hpp>
#include <networkit/community/OVPL.hpp>
#include <networkit/community/MPLM.hpp>
#include <networkit/community/OPLM.hpp>
#include <networkit/community/ParallelAgglomerativeClusterer.hpp>
#include <networkit/community/Modularity.hpp>
#include <networkit/community/EdgeCut.hpp>
#include <networkit/community/ClusteringGenerator.hpp>
#include <networkit/io/METISGraphReader.hpp>
#include <networkit/io/SNAPGraphReader.hpp>
#include <networkit/overlap/HashingOverlapper.hpp>
#include <networkit/community/GraphClusteringTools.hpp>
#include <networkit/structures/Partition.hpp>
#include <networkit/community/Modularity.hpp>
#include <networkit/community/Coverage.hpp>
#include <networkit/community/ClusteringGenerator.hpp>
#include <networkit/community/JaccardMeasure.hpp>
#include <networkit/community/NodeStructuralRandMeasure.hpp>
#include <networkit/community/GraphStructuralRandMeasure.hpp>
#include <networkit/community/NMIDistance.hpp>
#include <networkit/community/DynamicNMIDistance.hpp>
#include <networkit/auxiliary/NumericTools.hpp>
#include <networkit/generators/DynamicBarabasiAlbertGenerator.hpp>
#include <networkit/community/SampledGraphStructuralRandMeasure.hpp>
#include <networkit/community/SampledNodeStructuralRandMeasure.hpp>
#include <networkit/community/GraphClusteringTools.hpp>
#include <networkit/community/PartitionIntersection.hpp>
#include <networkit/community/HubDominance.hpp>
#include <networkit/community/IntrapartitionDensity.hpp>
#include <networkit/community/PartitionFragmentation.hpp>
#include <networkit/generators/ClusteredRandomGraphGenerator.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/generators/LFRGenerator.hpp>
#include <networkit/community/CoverF1Similarity.hpp>
#include <networkit/community/LFM.hpp>
#include <networkit/community/OverlappingNMIDistance.hpp>
#include <networkit/community/StablePartitionNodes.hpp>
#include <networkit/scd/LocalTightnessExpansion.hpp>

#include <tlx/unused.hpp>

#include <sys/stat.h>
#include<time.h>
#include <sys/time.h>

#ifndef OVERALL_LOG
#define OVERALL_LOG true
#endif

#ifndef RMAT_GRAPH
#define RMAT_GRAPH true
#endif

//#ifndef L1D_CACHE_MISS_COUNT
//#define L1D_CACHE_MISS_COUNT
//#endif

#ifndef THREAD_DEFINE
#define THREAD_DEFINE true
#endif

#ifndef CONFIDENCE_INTERVAL_LOG
#define CONFIDENCE_INTERVAL_LOG false
#endif

#ifndef POWER_LOG
#define POWER_LOG false
#endif

#ifndef SKYLAKE_X_LOG
#define SKYLAKE_X_LOG false
#endif

#ifndef COPPERHEAD
#define COPPERHEAD false
#endif

#ifndef NUM_RUN
#define NUM_RUN 12
#endif

#ifndef SKIP_RUN
#define SKIP_RUN 2
#endif

using namespace NetworKit;

struct Options {
    std::string loglevel = "ERROR";
    bool sourceLocation{false};
    unsigned numThreads{0};

    bool modeTests{false};
    bool modeDebug{false};
    bool modeBenchmarks{false};
    bool modeRunnable{false};

    bool parse(int argc, char* argv[]) {
        tlx::CmdlineParser parser;

        parser.add_bool('t', "tests",      modeTests,      "Run unit tests");
        parser.add_bool('r', "run",        modeRunnable,   "Run unit tests which don't use assertions");
        parser.add_bool('d', "debug",      modeDebug,      "Run tests to debug some algorithms");
        parser.add_bool('b', "benchmarks", modeBenchmarks, "Run benchmarks");

        parser.add_unsigned("threads",     numThreads,     "set the maximum number of threads; 0 (=default) uses OMP default");
        parser.add_string("loglevel",      loglevel,       "set the log level (TRACE|DEBUG|INFO|WARN|ERROR|FATAL)");
        parser.add_bool("srcloc",          sourceLocation, "print source location of log messages");

        if (!parser.process(argc, argv, std::cerr))
            return false;

        if (modeTests + modeDebug + modeBenchmarks + modeRunnable > 1) {
            std::cerr << "Select at most one of -t, -r, -d, -b\n";
            return false;
        }

        return true;
    }
};

std::string movephase_folder = "MovePhaseLog";
std::string first_movephase_folder = "FirstMovePhaseLog";
std::string vlm_details_folder = "VLMDetailsLog";
std::string plm_details_folder = "PLMDetailsLog";

int main(int argc, char *argv[]) {
    std::cout << "*** NetworKit Unit Tests ***\n";

    /// Collect info from commandline arguments
    count argi = 1;
    std::string path = "";
//    std::cout<<"Params size:" << argc << std::endl;
    if(argc>argi) {
        path = argv[argi++];
        std::cout << "Path: " << path << std::endl;
    }
    std::string ppn;
    if (argc >argi) {
        char* th = argv[argi];
        ppn = argv[argi++];
#if THREAD_DEFINE
        omp_set_dynamic(0);
        omp_set_num_threads((int)std::strtol((th), (char**)NULL, 10));
//        Aux::setNumberOfThreads((int)std::strtol((th), (char**)NULL, 10));
#endif
    }
//    std::cout<<"Threads:" << ppn << std::endl;
    int version = 0;
    if (argc > argi) {
        version = (int)std::strtol(argv[argi++], (char**)NULL, 10);
    }
//    std::cout<<"Version:" << version << std::endl;
    count _inputMethod = 1;
    if (argc > argi) {
        _inputMethod = (int)std::strtol(argv[argi++], (char**)NULL, 10);
    }
//    std::cout<<"Input Method:" << _inputMethod << std::endl;
    count _iterations = 25;
    if (argc > argi) {
        _iterations = (uint64_t)std::strtol(argv[argi++], (char**)NULL, 10);
    }
//    std::cout<<"Iterations:" << _iterations << std::endl;
    bool fullVec = false;
    if (argc > argi) {
        fullVec = std::stoi(argv[argi++]);
    }
//    std::cout<<"FullVec:" << fullVec << std::endl;
    count architecture = 1;
    if (argc > argi) {
        architecture = (int)std::strtol(argv[argi++], (char**)NULL, 10);
    }
    count scale = 20;
    if (argc > argi) {
        scale = (int)std::strtol(argv[argi++], (char**)NULL, 10);
    }
    count edgeFactor = 16;
    if (argc > argi) {
        edgeFactor = (int)std::strtol(argv[argi++], (char**)NULL, 10);
    }
    char *d_end;
    double a = 0.57;
    if (argc > argi) {
        a = (double)std::strtod(argv[argi++], &d_end);
        a = round(a*100)/100;
    }
    double b = 0.19;
    if (argc > argi) {
        b = (double)std::strtod(argv[argi++], &d_end);
        b = round(b*100)/100;
    }
    double c = 0.19;
    if (argc > argi) {
        c = (double)std::strtod(argv[argi++], &d_end);
        c = round(c*100)/100;
    }
    double d = 0.05;
    if (argc > argi) {
        d = (double)std::strtod(argv[argi++], &d_end);
        d = round(d*100)/100;
    }
    std::cout<<"scale:" << scale << " edgeFactor: " << edgeFactor << " a: " << a << " b: " << b
              << " c: " << c << " d: " << d << std::endl;
    bool refine = false;
    if (argc > argi) {
        refine = std::stoi(argv[argi++]);
    }
//    std::cout<<"refine:" << refine << std::endl;
    long cache_size = 25*1024*1024;
    if (argc > argi) {
        cache_size = std::strtol(argv[argi++], (char**)NULL, 10);
    }
//    std::cout<<"cache_size:" << cache_size << std::endl;

    Modularity modularity;
    std::string _graphName, dirName;
#if RMAT_GRAPH
    RmatGenerator rmat(scale, edgeFactor, a, b, c, d);
    Graph G = rmat.generate();
    _graphName = "RMAT_" + std::to_string(scale) + "_" + std::to_string(edgeFactor) + "-" +
                 std::to_string(a)  + "-" + std::to_string(b)  + "-" + std::to_string(c)  + "-"
                 + std::to_string(d);
#else
    /// Initialize reader
    SNAPGraphReader snapReader;
    METISGraphReader metisReader;
    std::vector<std::string>tokens = Aux::StringTools::split(path, '/');
    _graphName = Aux::StringTools::split(tokens[tokens.size()-1], '.')[0];
    Graph G;
    if(_inputMethod == 1)
        G = snapReader.read(path);
    else
        G = metisReader.read(path);
#endif


#if OVERALL_LOG
    std::ofstream graph_log;
    std::string conference = "Journal_Results/";
    std::string folderName = conference; // + (version >=4 ? "LP/" : "LM/");
    if (mkdir(folderName.c_str(), 0777) == -1)
        std::cout<<"Directory " << folderName << " is already exist" << std::endl;
    else
        std::cout<<"Directory " << folderName << " created" << std::endl;
    std::string logFileName = folderName + "RMAT_" + (fullVec ? "Full_Vec_" : "Partial_Vec_") + (architecture == 1 ? "SkyLake" : "CascadeLake") + ".csv";

    std::ifstream infile(logFileName);
    graph_log.open(logFileName, std::ios_base::out | std::ios_base::app | std::ios_base::ate);

    bool existing_file = infile.good();
    if (!existing_file) {
        graph_log << "GraphName" << "," << "Version" << "," << "Nodes" << ","<< "Edges" << ","
                  << "Wall Time" << "," << "CPU Time" << "," << "Clusters" << "," << "Modularity"
                  << "," << "MaxIterations" << "," << "FirstMoveTime" << "," << "MoveTime" << ","
                  << "CoarsenTime" << "," << "RefineTime" << "," << "Threads" << "," << "CacheLevel"
                  << "," << "CacheMissCount" << "," << "Refine" << "," << "StablePartitioningTime"
                  << "," << "Scale" << "," << "EdgeFactor" << "," << "a" << "," << "b" << "," << "c"
                  << "," << "d" << std::endl;
    }
    infile.close();
#endif
    count z = G.upperNodeIdBound();
    count edges = G.numberOfEdges();
    struct timespec start, cpu_start, end, cpu_end, stable_partition_start;
    double et_plm=0, et_mplm=0.0, et_onpl=0, et_ovpl=0, plm_mod=0, onpl_mod=0, ovpl_mod=0, mplm_mod=0;
    count plm_subsets=0, onpl_subsets=0, ovpl_subsets=0, mplm_subsets=0;

    auto Parallel_LM = [&](){
        std::cout << "***** Legacy Version *****" << std::endl;
        for (int k = 0; k < NUM_RUN+SKIP_RUN; ++k) {
            Graph gCopy = G;
            OPLM plm(gCopy, refine, 1.0, "balanced", _iterations);
#if POWER_LOG
            //            modifiedPLM.setupMPLMPowerFile(_graphName, std::stoi(ppn));
            rapl_sysfs_init();
            rapl_sysfs_before();
#endif
            //            clock_gettime(CLOCK_MONOTONIC, &start);
            clock_gettime(CLOCK_REALTIME, &start);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
            plm.run();
            if (k>=SKIP_RUN) {
#if POWER_LOG
                rapl_sysfs_after();
                // Results
                rapl_sysfs_results("PLM", _graphName, std::stoi(ppn), 1, architecture, folderName);
#endif
                //                clock_gettime(CLOCK_MONOTONIC, &end);
                clock_gettime(CLOCK_REALTIME, &end);
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end);
                long seconds = end.tv_sec - start.tv_sec;
                long nanoseconds = end.tv_nsec - start.tv_nsec;
                double elapsed = seconds + nanoseconds*1e-9;
                seconds = cpu_end.tv_sec - cpu_start.tv_sec;
                nanoseconds = cpu_end.tv_nsec - cpu_start.tv_nsec;
                double cpu_elapsed = seconds + nanoseconds*1e-9;

                Partition s_zeta = plm.getPartition();
                count comm = s_zeta.numberOfSubsets();
                double mod = modularity.getQuality(s_zeta, G);
                auto times = plm.getTiming();
                double move_time = 0.0, coarsen_time = 0.0, refine_time = 0.0;
                for (double t : times["move"]) {
                    move_time += t;
                }
                for (double t : times["coarsen"]) {
                    coarsen_time += t;
                }
                for (double t : times["refine"]) {
                    refine_time += t;
                }
#if OVERALL_LOG
                graph_log << _graphName << "," << "PLM" << "," << z << "," << edges << ","
                          << elapsed << "," << cpu_elapsed << "," << comm << "," << mod << ","
                          << (_iterations) << "," << times["move"][0] << "," << move_time << ","
                          << coarsen_time << "," << refine_time << "," << ppn << "," << "L1D"
                          << "," << 0 << "," << refine << "," << 0 << "," << scale << ","
                          << edgeFactor << "," << a << "," << b << "," << c << "," << d
                          << std::endl;
#endif
                et_plm += cpu_elapsed;
                plm_subsets += comm;
                plm_mod += mod;
            }
            //            std::cout<< "[" << k << "] ******************* Successfully Done ************************ " << std::endl;
        }
        et_plm = et_plm/NUM_RUN;
        plm_subsets = plm_subsets/NUM_RUN;
        plm_mod = plm_mod/NUM_RUN;
        std::cout << "Total CPU time without refinement: " << et_plm << std::endl;
        std::cout << "number of clusters: " << plm_subsets << std::endl;
        std::cout << "modularity: " << plm_mod << std::endl;

    };
    auto OneNeighborPerLane_LM = [&](){
        std::cout << "***** Vectorized Row Version *****" << std::endl;
        for (int k = 0; k < NUM_RUN+SKIP_RUN; ++k) {
            Graph gCopy = G;
            ONPL onpl(gCopy, refine, 1.0, "balanced", _iterations, fullVec);
            std::string vplm_conflict_file = plm_details_folder + "/vplm_conflict_log_" + ppn + "_" + _graphName + ".csv";
            onpl.setupCSVFile(vplm_conflict_file);
            onpl.initONPL();
#if POWER_LOG
            //            onpl.setupPowerFile(_graphName, std::stoi(ppn));
            std::cout<<"Setup energy" << std::endl;
            rapl_sysfs_init();
            rapl_sysfs_before();
            std::cout<<"Setup done" << std::endl;
#endif
            //            clock_gettime(CLOCK_MONOTONIC, &start_vplm);
            clock_gettime(CLOCK_REALTIME, &start);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
            std::cout<<"Run ONPL" << std::endl;
            onpl.run();
            std::cout<<"Done ONPL" << std::endl;
            if (k>=SKIP_RUN) {
                //                clock_gettime(CLOCK_MONOTONIC, &end_vplm);
                clock_gettime(CLOCK_REALTIME, &end);
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end);
                long seconds = end.tv_sec - start.tv_sec;
                long nanoseconds = end.tv_nsec - start.tv_nsec;
                double elapsed = seconds + nanoseconds*1e-9;
                seconds = cpu_end.tv_sec - cpu_start.tv_sec;
                nanoseconds = cpu_end.tv_nsec - cpu_start.tv_nsec;
                double cpu_elapsed = seconds + nanoseconds*1e-9;
#if POWER_LOG
                std::cout<<"prepare results of energy" << std::endl;
                rapl_sysfs_after();
                // Results
                rapl_sysfs_results("ONPL", _graphName, std::stoi(ppn), 1, architecture, folderName);
                std::cout<<"Done" << std::endl;
#endif

                Partition v_zeta = onpl.getPartition();
                count comm = v_zeta.numberOfSubsets();
                double mod = modularity.getQuality(v_zeta, G);
                auto times = onpl.getTiming();
                double move_time = 0, coarsen_time = 0, refine_time = 0;
                for (double t : times["move"]) {
                    move_time += t;
                }
                for (double t : times["coarsen"]) {
                    coarsen_time += t;
                }
                for (double t : times["refine"]) {
                    refine_time += t;
                }
//                auto c_counts = onpl.getCacheCount();
                long long cache_count = 0;
                /*for(auto c : c_counts["move"]){
                    cache_count += c;
                }*/
#if OVERALL_LOG
#ifdef L1D_CACHE_MISS_COUNT
                graph_log << _graphName << "," << "ONPL" << "," << z << "," << edges << ","
                          << elapsed << "," << cpu_elapsed << "," << comm << "," << mod << ","
                          << (_iterations) << "," << times["move"][0] << "," << move_time << ","
                          << coarsen_time << "," << refine_time << "," << ppn << "," << "L1D"
                          << "," << cache_count << "," << refine << "," << 0 << "," << scale
                          << "," << edgeFactor << "," << a << "," << b << "," << c << "," << d
                          << std::endl;
#else

                graph_log << _graphName << "," << "ONPL" << "," << z << "," << edges << ","
                          << elapsed << "," << cpu_elapsed << "," << comm << "," << mod << ","
                          << (_iterations) << "," << times["move"][0] << "," << move_time << ","
                          << coarsen_time << "," << refine_time << "," << ppn << "," << "LL"
                          << "," << cache_count << "," << refine << "," << 0 << "," << scale
                          << "," << edgeFactor << "," << a << "," << b << "," << c << "," << d
                          << std::endl;
#endif
#endif
                et_onpl += cpu_elapsed;
                onpl_subsets += comm;
                onpl_mod += mod;
            }
            //            std::cout<< "[" << k << "] ******************* Successfully Done ************************ " << std::endl;
        }
        et_onpl = et_onpl/NUM_RUN;
        onpl_subsets = onpl_subsets/NUM_RUN;
        onpl_mod = onpl_mod/NUM_RUN;
        std::cout << "Total CPU time without refinement: " << et_onpl << std::endl;
        std::cout << "number of clusters: " << onpl_subsets << std::endl;
        std::cout << "modularity: " << onpl_mod << std::endl;

    };
    auto OneVertexPerLane_LM = [&](){
        std::cout << "***** Vectorized Block Version *****" << std::endl;
        for (int k = 0; k < NUM_RUN+SKIP_RUN; ++k) {
            Graph gCopy = G;
            OVPL plm(gCopy, refine, 1.0, "balanced", _iterations);
            std::string move_phase_log_file = movephase_folder + "/move_phase_log_" + ppn + "_" + _graphName + ".csv";
            std::string first_move_phase_log_file =
                first_movephase_folder + "/first_move_phase_log_" + ppn + "_" + _graphName + ".csv";
            std::string vlm_details_log_file =
                vlm_details_folder + "/vlm_details_log_" + ppn + "_" + _graphName + ".csv";
            std::string plm_details_log = plm_details_folder + "/plm_details_log_" + ppn + "_" + _graphName + ".csv";
            plm.setupCSVFile(move_phase_log_file, first_move_phase_log_file, vlm_details_log_file, plm_details_log);
#if POWER_LOG
            //            modifiedPLM.setupMPLMPowerFile(_graphName, std::stoi(ppn));
            rapl_sysfs_init();
            rapl_sysfs_before();
#endif
            //            clock_gettime(CLOCK_MONOTONIC, &start);
            clock_gettime(CLOCK_REALTIME, &start);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
            plm.run();
            if (k>=SKIP_RUN) {
#if POWER_LOG
                rapl_sysfs_after();
                // Results
                rapl_sysfs_results("OVPL", _graphName, std::stoi(ppn), 1, architecture, folderName);
#endif
                //                clock_gettime(CLOCK_MONOTONIC, &end);
                clock_gettime(CLOCK_REALTIME, &end);
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end);
                long seconds = end.tv_sec - start.tv_sec;
                long nanoseconds = end.tv_nsec - start.tv_nsec;
                double elapsed = seconds + nanoseconds*1e-9;
                seconds = cpu_end.tv_sec - cpu_start.tv_sec;
                nanoseconds = cpu_end.tv_nsec - cpu_start.tv_nsec;
                double cpu_elapsed = seconds + nanoseconds*1e-9;

                Partition zeta = plm.getPartition();
                count comm = zeta.numberOfSubsets();
                double mod = modularity.getQuality(zeta, G);
                auto times = plm.getTiming();
                double move_time = 0, coarsen_time = 0, refine_time = 0;
                for (double t : times["move"]) {
                    move_time += t;
                }
                for (double t : times["coarsen"]) {
                    coarsen_time += t;
                }
                for (double t : times["refine"]) {
                    refine_time += t;
                }

//                auto c_counts = plm.getCacheCount();
                long long cache_count = 0;
                /*for(auto c : c_counts["move"]){
                    cache_count += c;
                }*/

#if OVERALL_LOG
#ifdef L1D_CACHE_MISS_COUNT
                graph_log << _graphName << "," << "OVPL" << "," << z << "," << edges << ","
                          << elapsed << "," << cpu_elapsed << "," << comm << "," << mod << ","
                          << (_iterations) << "," << times["move"][0] << "," << move_time << ","
                          << coarsen_time << "," << refine_time << "," << ppn << "," << "L1D"
                          << "," << cache_count << "," << refine << "," << 0 << "," << scale << ","
                          << edgeFactor << "," << a << "," << b << "," << c << "," << d
                          << std::endl;
#else
                graph_log << _graphName << "," << "OVPL" << "," << z << "," << edges << ","
                          << elapsed << "," << cpu_elapsed << "," << comm << "," << mod << ","
                          << (_iterations) << "," << times["move"][0] << "," << move_time << ","
                          << coarsen_time << "," << refine_time << "," << ppn << "," << "LL"
                          << "," << cache_count << "," << refine << "," << 0 << "," << scale << ","
                          << edgeFactor << "," << a << "," << b << "," << c << "," << d
                          << std::endl;
#endif
#endif
                et_ovpl += cpu_elapsed;
                ovpl_subsets += comm;
                ovpl_mod += mod;
            }
            //            std::cout<< "[" << k << "] ******************* Successfully Done ************************ " << std::endl;
        }
        et_ovpl = et_ovpl/NUM_RUN;
        ovpl_subsets = ovpl_subsets/NUM_RUN;
        ovpl_mod = ovpl_mod/NUM_RUN;
        std::cout << "Total CPU time without refinement: " << et_ovpl << std::endl;
        std::cout << "number of clusters: " << ovpl_subsets << std::endl;
        std::cout << "modularity: " << ovpl_mod << std::endl;

    };
    auto Modified_PLM = [&](){
        std::cout << "***** Modified Legacy Version *****" << std::endl;
        for (int k = 0; k < NUM_RUN+SKIP_RUN; ++k) {
            Graph gCopy = G;
            MPLM modifiedPLM(gCopy, refine, 1.0, "balanced", _iterations);
            std::string mplm_move_details_file =
                plm_details_folder + "/mplm_move_log_" + ppn + "_" + _graphName + ".csv";
            modifiedPLM.setupMoveDetailsCSVFile(mplm_move_details_file);
            modifiedPLM.initMPLM();
#if POWER_LOG
            //            modifiedPLM.setupMPLMPowerFile(_graphName, std::stoi(ppn));
            rapl_sysfs_init();
            rapl_sysfs_before();
#endif
            //            clock_gettime(CLOCK_MONOTONIC, &start_modified);
            clock_gettime(CLOCK_REALTIME, &start);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
            modifiedPLM.run();
            if (k>=SKIP_RUN) {
                //                clock_gettime(CLOCK_MONOTONIC, &end_modified);
                clock_gettime(CLOCK_REALTIME, &end);
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end);
                long seconds = end.tv_sec - start.tv_sec;
                long nanoseconds = end.tv_nsec - start.tv_nsec;
                double elapsed = seconds + nanoseconds*1e-9;
                seconds = cpu_end.tv_sec - cpu_start.tv_sec;
                nanoseconds = cpu_end.tv_nsec - cpu_start.tv_nsec;
                double cpu_elapsed = seconds + nanoseconds*1e-9;
#if POWER_LOG
                rapl_sysfs_after();
                // Results
                rapl_sysfs_results("MPLM", _graphName, std::stoi(ppn), 1, architecture, folderName);
#endif

                Partition s_zeta = modifiedPLM.getPartition();
                count comm = s_zeta.numberOfSubsets();
                double mod = modularity.getQuality(s_zeta, G);
                auto times = modifiedPLM.getTiming();
                double move_time = 0, coarsen_time = 0, refine_time = 0;
                for (double t : times["move"]) {
                    move_time += t;
                }
                for (double t : times["coarsen"]) {
                    coarsen_time += t;
                }
                for (double t : times["refine"]) {
                    refine_time += t;
                }

//                auto c_counts = modifiedPLM.getCacheCount();
                long long cache_count = 0;
                /*for(auto c : c_counts["move"]){
                    cache_count += c;
                }*/
#if OVERALL_LOG
#ifdef L1D_CACHE_MISS_COUNT
                graph_log << _graphName << "," << "MPLM" << "," << z << "," << edges << ","
                          << elapsed << "," << cpu_elapsed << "," << comm << "," << mod << ","
                          << (_iterations) << "," << times["move"][0] << "," << move_time << ","
                          << coarsen_time << "," << refine_time << "," << ppn << "," << "L1D"
                          << "," << cache_count << "," << refine << "," << 0 << "," << scale << ","
                          << edgeFactor << "," << a << "," << b << "," << c << "," << d
                          << std::endl;
#else
                graph_log << _graphName << "," << "MPLM" << "," << z << "," << edges << ","
                          << elapsed << "," << cpu_elapsed << "," << comm << "," << mod << ","
                          << (_iterations) << "," << times["move"][0] << "," << move_time << ","
                          << coarsen_time << "," << refine_time << "," << ppn << "," << "LL"
                          << "," << cache_count << "," << refine << "," << 0 << "," << scale << ","
                          << edgeFactor << "," << a << "," << b << "," << c << "," << d
                          << std::endl;
#endif
#endif
                et_mplm += cpu_elapsed;
                mplm_subsets += comm;
                mplm_mod += mod;
            }
            //            std::cout<< "[" << k << "] ******************* Successfully Done ************************ " << std::endl;
        }
        et_mplm = et_mplm/NUM_RUN;
        mplm_subsets = mplm_subsets/NUM_RUN;
        mplm_mod = mplm_mod/NUM_RUN;
        std::cout << "Total CPU time without refinement: " << et_mplm << std::endl;
        std::cout << "number of clusters: " << mplm_subsets << std::endl;
        std::cout << "modularity: " << mplm_mod << std::endl;
    };
    auto Parallel_LP = [&](){
        std::cout << "***** Legacy PLP *****" << std::endl;
        for (int k = 0; k < NUM_RUN+SKIP_RUN; ++k) {
            Graph gCopy = G;
            PLP lp(gCopy, (count)std::numeric_limits<uint64_t>::max(), _iterations);
#if POWER_LOG
            //            modifiedPLM.setupMPLMPowerFile(_graphName, std::stoi(ppn));
            rapl_sysfs_init();
            rapl_sysfs_before();
#endif
            //            clock_gettime(CLOCK_MONOTONIC, &start_modified);
            clock_gettime(CLOCK_REALTIME, &start);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
            lp.run();
            if (k>=SKIP_RUN) {
                //                clock_gettime(CLOCK_MONOTONIC, &end_modified);
                clock_gettime(CLOCK_REALTIME, &end);
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end);
                long seconds = end.tv_sec - start.tv_sec;
                long nanoseconds = end.tv_nsec - start.tv_nsec;
                double elapsed = seconds + nanoseconds*1e-9;
                seconds = cpu_end.tv_sec - cpu_start.tv_sec;
                nanoseconds = cpu_end.tv_nsec - cpu_start.tv_nsec;
                double cpu_elapsed = seconds + nanoseconds*1e-9;
#if POWER_LOG
                rapl_sysfs_after();
                // Results
                rapl_sysfs_results("PLP", _graphName, std::stoi(ppn), 1, architecture, folderName);
#endif

                Partition s_zeta = lp.getPartition();
                count comm = s_zeta.numberOfSubsets();
                double mod = modularity.getQuality(s_zeta, G);
                auto times = lp.getTiming();
                count run_time = 0;
                for (count t : times) {
                    run_time += t;
                }

#if OVERALL_LOG
                graph_log << _graphName << "," << "PLP" << "," << z << "," << edges << ","
                          << elapsed << "," << cpu_elapsed << "," << comm << "," << mod << ","
                          << (_iterations) << "," << 0 << "," << run_time << "," << 0 << "," << 0
                          << "," << ppn << "," << "LL" << "," << 0 << "," << 0 << "," << 0 << ","
                          << scale << "," << edgeFactor << "," << a << "," << b << "," << c << ","
                          << d << std::endl;
#endif
                et_mplm += cpu_elapsed;
                mplm_subsets += comm;
                mplm_mod += mod;
            }
            //            std::cout<< "[" << k << "] ******************* Successfully Done ************************ " << std::endl;
        }
        et_mplm = et_mplm/NUM_RUN;
        mplm_subsets = mplm_subsets/NUM_RUN;
        mplm_mod = mplm_mod/NUM_RUN;
        std::cout << "Total CPU time without refinement: " << et_mplm << std::endl;
        std::cout << "number of clusters: " << mplm_subsets << std::endl;
        std::cout << "modularity: " << mplm_mod << std::endl;
    };
    auto Modified_PLP = [&](){
        std::cout << "***** Modified PLP *****" << std::endl;
        for (int k = 0; k < NUM_RUN+SKIP_RUN; ++k) {
            Graph gCopy = G;
            MPLP mplp(gCopy, (count)std::numeric_limits<uint64_t>::max(), _iterations);
#if POWER_LOG
            //            modifiedPLM.setupMPLMPowerFile(_graphName, std::stoi(ppn));
            rapl_sysfs_init();
            rapl_sysfs_before();
#endif
            //            clock_gettime(CLOCK_MONOTONIC, &start_modified);
            clock_gettime(CLOCK_REALTIME, &start);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
            mplp.run();
            clock_gettime(CLOCK_REALTIME, &stable_partition_start);
//            StablePartitionNodes stablePartitionNodes(G, mplp.getPartition(), mplp.getPartition());
//            stablePartitionNodes.run();
            if (k>=SKIP_RUN) {
                //                clock_gettime(CLOCK_MONOTONIC, &end_modified);
                clock_gettime(CLOCK_REALTIME, &end);
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end);
//                long seconds = end.tv_sec - start.tv_sec;
//                long nanoseconds = end.tv_nsec - start.tv_nsec;
                long seconds = stable_partition_start.tv_sec - start.tv_sec;
                long nanoseconds = stable_partition_start.tv_nsec - start.tv_nsec;
                double elapsed = seconds + nanoseconds*1e-9;
                double stable_partitioning_time = (end.tv_sec - stable_partition_start.tv_sec) +
                                              (end.tv_nsec - stable_partition_start.tv_nsec)*1e-9;
                seconds = cpu_end.tv_sec - cpu_start.tv_sec;
                nanoseconds = cpu_end.tv_nsec - cpu_start.tv_nsec;
                double cpu_elapsed = seconds + nanoseconds*1e-9;
#if POWER_LOG
                rapl_sysfs_after();
                // Results
                rapl_sysfs_results("MPLP", _graphName, std::stoi(ppn), 1, architecture, folderName);
#endif

                Partition s_zeta = mplp.getPartition();
                count comm = s_zeta.numberOfSubsets();
                double mod = modularity.getQuality(s_zeta, G);
                auto times = mplp.getTiming();
                count run_time = 0;
                for (count t : times) {
                    run_time += t;
                }

#if OVERALL_LOG
                graph_log << _graphName << "," << "MPLP" << "," << z << "," << edges << ","
                          << elapsed << "," << cpu_elapsed << "," << comm << "," << mod << ","
                          << (_iterations) << "," << 0 << "," << run_time << "," << 0 << "," << 0
                          << "," << ppn << "," << "LL" << "," << 0 << "," << 0 << ","
                          << stable_partitioning_time << "," << scale << "," << edgeFactor << ","
                          << a << "," << b << "," << c << "," << d << std::endl;
#endif
                et_mplm += cpu_elapsed;
                mplm_subsets += comm;
                mplm_mod += mod;
            }
            //            std::cout<< "[" << k << "] ******************* Successfully Done ************************ " << std::endl;
        }
        et_mplm = et_mplm/NUM_RUN;
        mplm_subsets = mplm_subsets/NUM_RUN;
        mplm_mod = mplm_mod/NUM_RUN;
        std::cout << "Total CPU time without refinement: " << et_mplm << std::endl;
        std::cout << "number of clusters: " << mplm_subsets << std::endl;
        std::cout << "modularity: " << mplm_mod << std::endl;
    };
    auto OneNeighborPerLane_LP = [&](){
        std::cout << "***** One Neighbor PLP *****" << std::endl;
        std::cout<<"Nodes: " << G.upperNodeIdBound() << " Edges: " << G.numberOfEdges() << std::endl;
        for (int k = 0; k < NUM_RUN+SKIP_RUN; ++k) {
            Graph gCopy = G;
            ONLP onlp(gCopy, (count)std::numeric_limits<uint64_t>::max(), _iterations);
#if POWER_LOG
            //            modifiedPLM.setupMPLMPowerFile(_graphName, std::stoi(ppn));
            rapl_sysfs_init();
            rapl_sysfs_before();
#endif
            //            clock_gettime(CLOCK_MONOTONIC, &start_modified);
            clock_gettime(CLOCK_REALTIME, &start);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
            onlp.run();
            std::cout<< "ONLP done" << std::endl;
            /// perform stable partitioning check
            clock_gettime(CLOCK_REALTIME, &stable_partition_start);
//            StablePartitionNodes stablePartitionNodes(G, onlp.getPartition(), onlp.getPartition());
//            stablePartitionNodes.setVectorized(true);
//            stablePartitionNodes.run();
            if (k>=SKIP_RUN) {
                //                clock_gettime(CLOCK_MONOTONIC, &end_modified);
                clock_gettime(CLOCK_REALTIME, &end);
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end);
//                long seconds = end.tv_sec - start.tv_sec;
//                long nanoseconds = end.tv_nsec - start.tv_nsec;
                long seconds = stable_partition_start.tv_sec - start.tv_sec;
                long nanoseconds = stable_partition_start.tv_nsec - start.tv_nsec;
                double elapsed = seconds + nanoseconds*1e-9;
                double stable_partitioning_time = (end.tv_sec - stable_partition_start.tv_sec) +
                                              (end.tv_nsec - stable_partition_start.tv_nsec)*1e-9;
                seconds = cpu_end.tv_sec - cpu_start.tv_sec;
                nanoseconds = cpu_end.tv_nsec - cpu_start.tv_nsec;
                double cpu_elapsed = seconds + nanoseconds*1e-9;
#if POWER_LOG
                rapl_sysfs_after();
                // Results
                rapl_sysfs_results("ONLP", _graphName, std::stoi(ppn), 1, architecture, folderName);
#endif

                Partition s_zeta = onlp.getPartition();
                count comm = s_zeta.numberOfSubsets();
                double mod = modularity.getQuality(s_zeta, G);
                auto times = onlp.getTiming();
                count run_time = 0;
                for (count t : times) {
                    run_time += t;
                }

#if OVERALL_LOG
                graph_log << _graphName << "," << "ONLP" << "," << z << "," << edges << ","
                          << elapsed << "," << cpu_elapsed << "," << comm << "," << mod << ","
                          << (_iterations) << "," << 0 << "," << run_time << "," << 0 << "," << 0
                          << "," << ppn << "," << "LL" << "," << 0 << "," << 0 << ","
                          << stable_partitioning_time << "," << scale << "," << edgeFactor << ","
                          << a << "," << b << "," << c << "," << d << std::endl;
#endif
                et_mplm += cpu_elapsed;
                mplm_subsets += comm;
                mplm_mod += mod;
            }
            //            std::cout<< "[" << k << "] ******************* Successfully Done ************************ " << std::endl;
        }
        et_mplm = et_mplm/NUM_RUN;
        mplm_subsets = mplm_subsets/NUM_RUN;
        mplm_mod = mplm_mod/NUM_RUN;
        std::cout << "Total CPU time without refinement: " << et_mplm << std::endl;
        std::cout << "number of clusters: " << mplm_subsets << std::endl;
        std::cout << "modularity: " << mplm_mod << std::endl;
    };
    Parallel_LM();
    OneNeighborPerLane_LM();
    OneVertexPerLane_LM();
    Modified_PLM();
    Parallel_LP();
    Modified_PLP();
    OneNeighborPerLane_LP();

#if OVERALL_LOG
    graph_log.close();
#endif

    return 0;

    /*
    ::testing::InitGoogleTest(&argc, argv);
    Options options;
    if (!options.parse(argc, argv)) {
        return -1;
    }

    // Configure logging
    Aux::Log::setLogLevel(options.loglevel);
    NetworKit::GlobalState::setPrintLocation(options.sourceLocation);
    std::cout << "Loglevel: " << Aux::Log::getLogLevel() << "\n";
#ifdef NETWORKIT_RELEASE_LOGGING
    if (options.loglevel == "TRACE" || options.loglevel == "DEBUG")
        std::cout << "WARNING: TRACE and DEBUG messages are missing"
                     " in NETWORKIT_RELEASE_LOGGING builds" << std::endl;
#endif

    // Configure parallelism
    {
        if (options.numThreads) {
            Aux::setNumberOfThreads(options.numThreads);
        }
        std::cout << "Max. number of threads: " << Aux::getMaxNumberOfThreads() << "\n";
    }

    // Configure test filter
    {
        auto setFilter = [&](const char *filter) {
            ::testing::GTEST_FLAG(filter) = filter;
        };

        if (options.modeTests) {
            setFilter("*Test.test*");
        } else if (options.modeDebug) {
            setFilter("*Test.debug*");
        } else if (options.modeBenchmarks) {
            setFilter("*Benchmark*");
        } else if (options.modeRunnable) {
            setFilter("*Test.run*");
        }
    }

    INFO("=== starting unit tests ===");

    return RUN_ALL_TESTS();*/
}
