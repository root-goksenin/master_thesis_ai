#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=BM25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output_%A.out

module load 2022
module load Anaconda3/2022.05 
source activate gpl_env_1

cd $HOME/master_thesis_ai/elasticsearch-7.9.2/bin/
./elasticsearch -q &


cd $HOME/master_thesis_ai
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8


cat lotte_paths.txt | xargs -I {} python3 generate_bm25_results.py --data_name {}