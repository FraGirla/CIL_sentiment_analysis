#!/bin/bash

#SBATCH -n 10
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=roberta_script
#SBATCH --output=roberta_script.out
#SBATCH --gpus=1
#SBATCH --gres=gpumem:29g

module load gcc/8.2.0 python_gpu/3.10.4 r/4.0.2 git-lfs/2.3.0 eth_proxy npm/6.14.9 libsndfile/1.0.23
pip install -r requirements.txt
python roBERTa_regression_script.py
