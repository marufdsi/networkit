#!/bin/bash


#SBATCH --job-name=skl # Job name
#SBATCH --partition=Orion
# Number of MPI tasks
#SBATCH --nodes=1                   # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --ntasks-per-node=1       # maximum task per node	
#SBATCH --cpus-per-task=36	# Number of CPU cores per task
#SBATCH --constraint=skylake    # for Skylake Processor
#SBATCH --mem=300gb                  # Total memory limit
#SBATCH --time=30:00:00              # Time limit hrs:min:sec
#SBATCH --output=skylake_%j.log     # Standard output and error log

# Clear the environment from any previously loaded modules
#module purge > /dev/null 2>&1
#date;hostname;pwd

#module load gcc/8.2.0

#ulimit -s 10240

vecType=0
arch=1
it=25

echo "Path: "$1" threads: "$2" io Method: "$3
for v in 6 #0 1 2 3 4 5 6
do
#	perf stat -B -e cycles:uk,cache-misses:uk,branches:uk,branch-misses:uk,cpu-clock:uk,alignment-faults:uk,cs:uk,L1-dcache-load-misses:uk,L1-dcache-store-misses:uk
	./networkit_tests $1 $2 $v $3 $it $vecType $arch $4 $5 $6 $7 $8 $9
  sleep 10
done

