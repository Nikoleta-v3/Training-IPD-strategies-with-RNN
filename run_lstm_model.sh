#!/bin/bash
#SBATCH --job-name=lstm  # Job Name
#SBATCH -o lstm_model.out           # STDOUT file from the job (%J job number)
#SBATCH -e lstm_model.err           # STDERR file from the job (%x job name)
#SBATCH --ntasks=40            # number of parallel processes (tasks)
#SBATCH --ntasks-per-node=40   # tasks to run per node
#SBATCH --time=2-00:00:00      # max wall time
#SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH --exclusive            # exclusive node access
#SBATCH --account=scw1224      # project account code

module load CUDA/9.2
module load anaconda/3

source activate keras-gpu

python train_lstm.py lstm_model 1000 data/targets.csv data/sequences.csv
