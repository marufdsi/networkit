#!/bin/bash

path=../../Partitioning_Tools/metis-5.1.0/graphs
for g in 333SP #AS365 M6 NACA0015 NLR Oregon-2 asia.osm belgium.osm delaunay_n24 europe.osm germany.osm "in-2004" kkt_power "loc-gowalla_edges" luxembourg.osm netherlands.osm nlpkkt200 roadNet-PA uk-2002
do
#	echo $path"/"$g".graph"
	sbatch cascade_job.sh $path"/"$g".graph" 48 2
#	sleep 10
done
