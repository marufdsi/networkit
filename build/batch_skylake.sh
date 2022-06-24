#!/bin/bash

#path=../../Partitioning_Tools/metis-5.1.0/graphs
#for g in 333SP AS365 M6 NACA0015 NLR Oregon-2 asia.osm belgium.osm delaunay_n24 europe.osm germany.osm "in-2004" kkt_power "loc-gowalla_edges" luxembourg.osm netherlands.osm nlpkkt200 roadNet-PA uk-2002
#do
#	echo $path"/"$g".graph"
#	sbatch skylake_job.sh $path"/"$g".graph" 36 2
#	sleep 10
#done


for scale in 20 17 18 19 21 22 23 24
do
    for ef in 32 16 64 128 1 2 4 8
    do
      sbatch skylake_job.sh "RMAT" 36 2 $scale $ef 0.57 0.19 0.19 0.05
      sbatch skylake_job.sh "RMAT" 36 2 $scale $ef 0.33 0.33 0.33 0.01
      sbatch skylake_job.sh "RMAT" 36 2 $scale $ef 0.40 0.30 0.20 0.10
    done
done
