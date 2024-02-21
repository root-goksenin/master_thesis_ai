#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=covid-tinybert
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=slurm_output_%A.out
#SBATCH --mem=80G

module load 2022
module load Anaconda3/2022.05 
source activate gpl_env_1

cd $HOME/master_thesis_ai
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

models=(msmarco-distilbert-dot-v5 multi-qa-distilbert-dot-v1 GPL/msmarco-distilbert-margin-mse)
datasets=(beir_data/scidocs beir_data/trec-covid beir_data/fever)

for model in "${models[@]}" 
    do
    for data in "${datasets[@]}" 
    do
        # for data_path in "$data"/*
        # do
            echo $model $data_path
            python3 zero_shot.py --model_name "$model" --data_path "$data"
        done 
    # done
done 
