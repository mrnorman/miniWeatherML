#!/bin/bash
#SBATCH -J miniWeatherML
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH --partition eval-gpu
#SBATCH --exclusive

nodes=1
mydir=job_$nodes
ntasks=`echo "8*$nodes" | bc`

cd /home/$USER/miniWeatherML/build
source machines/aws/aws_a100_gpu.env

echo "Nodes: $nodes"
echo "Tasks: $ntasks"
echo "Dir:   `pwd`/$mydir"

mkdir -p $mydir
cp ./driver $mydir
cp ./inputs/* $mydir
cd $mydir

export OMP_NUM_THREADS=1
nvidia-smi
srun -N $nodes -n $ntasks --gpus-per-task=1 -c 1 /home/$USER/hello_jobstep.exe 2>&1 | tee hello_jsrun_output.txt
srun -N $nodes -n $ntasks --gpus-per-task=1 --gpu-bind=closest -c 1 ./driver ./input_euler3d_1024x1024x100.yaml 2>&1 | tee job_output_1024x1024x100.txt
# jsrun -r 6 -n $ntasks -a 1 -c 1 -g 1 ./driver ./input_euler3d_2048x2048x100.yaml 2>&1 | tee job_output_2048x2048x100.txt
# jsrun -r 6 -n $ntasks -a 1 -c 1 -g 1 ./driver ./input_euler3d_4096x4096x100.yaml 2>&1 | tee job_output_4096x4096x100.txt


