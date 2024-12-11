#!/bin/bash
#SBATCH -c 8                                        # Number of cores (-c)
#SBATCH --gres=gpu:1                                # GPU
#SBATCH -t 0-12:00                                  # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu                                 # Partition to submit to
#SBATCH --mem=32000                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o results_alex_f/run_alex_2_%j.out                          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errors_alex_f/run_alex_2_%j.err

mamba activate chainer_env
python adaptive_alex.py 2