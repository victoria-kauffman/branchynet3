#!/bin/bash
#SBATCH -c 8                                        # Number of cores (-c)
#SBATCH --gres=gpu:1                                # GPU
#SBATCH -t 0-12:00                                  # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu                                 # Partition to submit to
#SBATCH --mem=32000                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o results/run_lenet_mnist_%j.out                          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errors/run_lenet_mnist_%j.err

conda activate chainer_env
python adaptive_lenet_mnist.py