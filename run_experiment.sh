#!/bin/bash
#SBATCH -c 8                                        # Number of cores (-c)
#SBATCH --gres=gpu:1                                # GPU
#SBATCH -t 0-12:00                                  # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_test                                 # Partition to submit to
#SBATCH --mem=32000                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o results_lenet/run_lenet_3_%j.out                          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errors_lenet/run_lenet_3_%j.err

conda activate chainer_env
python adaptive_lenet_mnist.py 3