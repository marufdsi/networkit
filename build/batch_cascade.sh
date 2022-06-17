#!/bin/bash

#path=../../Partitioning_Tools/metis-5.1.0/graphs
#for g in 333SP #AS365 M6 NACA0015 NLR Oregon-2 asia.osm belgium.osm delaunay_n24 europe.osm germany.osm "in-2004" kkt_power "loc-gowalla_edges" luxembourg.osm netherlands.osm nlpkkt200 roadNet-PA uk-2002
#do
#	echo $path"/"$g".graph"
	#sbatch cascade_job.sh $path"/"$g".graph" 48 2
#	sleep 10
#done

for scale in 17 #18 19 #21 22 23 24 #20
do
    for ef in 16 #32 64 128 2 4 8
    do
      sbatch cascade_job.sh "RMAT" 48 2 $scale $ef 0.57 0.19 0.19 0.05
      sbatch cascade_job.sh "RMAT" 48 2 $scale $ef 0.33 0.33 0.33 0.01
      sbatch cascade_job.sh "RMAT" 48 2 $scale $ef 0.40 0.30 0.20 0.10
    done
done
