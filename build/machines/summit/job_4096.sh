#!/bin/bash
#BSUB -P stf006
#BSUB -W 0:15
#BSUB -nnodes 4096
#BSUB -q debug
#BSUB -J miniWeatherML
#BSUB -o miniWeatherML.%J
#BSUB -e miniWeatherML.%J

nodes=4096
mydir=job_$nodes
ntasks=`echo "6*$nodes" | bc`

cd /gpfs/alpine/proj-shared/stf006/imn/miniWeatherML/build
source machines/summit/summit_gpu.env

echo "Nodes: $nodes"
echo "Tasks: $ntasks"
echo "Dir:   `pwd`/$mydir"

mkdir -p $mydir
cp ./driver $mydir
cp ./inputs/* $mydir
cd $mydir

jsrun -r 6 -n $ntasks -a 1 -c 1 -g 1 ./driver ./input_euler3d_1024x1024x100.yaml 2>&1 | tee job_output_1024x1024x100.txt
jsrun -r 6 -n $ntasks -a 1 -c 1 -g 1 ./driver ./input_euler3d_2048x2048x100.yaml 2>&1 | tee job_output_2048x2048x100.txt
jsrun -r 6 -n $ntasks -a 1 -c 1 -g 1 ./driver ./input_euler3d_4096x4096x100.yaml 2>&1 | tee job_output_4096x4096x100.txt


