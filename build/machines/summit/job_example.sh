#!/bin/bash
#BSUB -P stf006
#BSUB -W 0:10
#BSUB -nnodes 32
#BSUB -q debug
#BSUB -J miniWeatherML
#BSUB -o miniWeatherML.%J
#BSUB -e miniWeatherML.%J

cd /gpfs/alpine/proj-shared/stf006/imn/miniWeatherML/build
source machines/summit/summit_gpu.env

jsrun -r 6 -n 192 -a 1 -c 1 -g 1 ./driver ./inputs/input_euler3d.yaml

