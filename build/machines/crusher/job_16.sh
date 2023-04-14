#!/bin/bash
#SBATCH -A stf006
#SBATCH -J miniWeatherML
#SBATCH -o %x-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 16

NODES=16
MW_HOME=/ccs/home/imn/miniWeatherML
# JOBDIR=/lustre/orion/stf006/scratch/imn/miniWeatherML_jobs/nodes_$NODES
JOBDIR=/gpfs/alpine/stf006/scratch/imn/miniWeatherML_jobs/nodes_$NODES

TASKS=`echo "8*$NODES" | bc`
echo "*** USING $TASKS TASKS ***"

mkdir -p $JOBDIR
cd $JOBDIR
source $MW_HOME/build/machines/crusher/crusher_gpu.env
cp -f $MW_HOME/build/driver* .
cp -Lrf $MW_HOME/build/input* .

srun -N $NODES -n $TASKS -c 1 --gpus-per-node=8 --gpus-per-task=1 --gpu-bind=closest ./driver ./inputs/input_euler3d.yaml | tee output.txt

