#!/bin/bash
#SBATCH -c 8                                        # Number of cores (-c)
#SBATCH --gres=gpu:1                                # GPU
#SBATCH -t 0-12:00                                  # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu                                 # Partition to submit to
#SBATCH --mem=32000                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o run_lenet_mnist_%j.out                          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e run_lenet_mnist_%j.err

module load python/2.7 cuda/12.0.1-fasrc01 cuda/9.1.85-fasrc01 cudnn/8.8.0.121_cuda12-fasrc01

mamba activate cs242_proj
# mamba install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# pip3 install numpy, ptflops, torchjpeg, matplotlib
python experiment_lenet_mnist.py