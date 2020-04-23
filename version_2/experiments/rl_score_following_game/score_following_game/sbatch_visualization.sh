#!/bin/bash
#SBATCH --partition=cpsc424_gpu

# set total number of MPI processes
#SBATCH --ntasks=1
# set number of cpus per MPI process
#SBATCH --cpus-per-task=1
# set memory per cpu
#SBATCH --mem-per-cpu=6100mb
# add gpu support
#SBATCH --gres-flags=enforce-binding
#SBATCH --gres=gpu:1
#SBATCH --job-name=Viz
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out

echo "Starting visualization"
echo ""
module load miniconda
module load FFmpeg/3.4.2-foss-2018a
module load GCC CUDA
source activate score_following
# Update the train and eval paths as necessary
python test_agent.py --params recurrent_approach/params/rnn-ScoreFollowingNetMSMDLCHSDeepDoLight-msmd_all_train-mutopia_lchs1_20200419_012146-cpsc424_alg76/best_model.pt  --data_set ./data/test_sample --piece Anonymous__lesgraces__lesgraces --game_config game_configs/mutopia_lchs1.yaml  --agent_type gru
