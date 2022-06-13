#!/bin/bash
#BSUB -P stf006
#BSUB -W 6:00
#BSUB -nnodes 6
#BSUB -J miniWeatherML
#BSUB -o miniWeatherML.%J
#BSUB -e miniWeatherML.%J


cd /gpfs/alpine/stf006/proj-shared/imn/miniWeatherML_private/build
source machines/summit/summit_gpu.env
jsrun -n 36 -a 1 -c 1 -g 1 ./driver ./inputs/input_euler3d.yaml
