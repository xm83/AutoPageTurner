#!/bin/bash
#SBATCH --partition=cpsc424

# set total number of MPI processes
#SBATCH --ntasks=8
# set number of cpus per MPI process
#SBATCH --cpus-per-task=1
# set memory per cpu
#SBATCH --mem-per-cpu=6100mb
#SBATCH --job-name=Train
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

echo "Starting training"
echo ""
module load miniconda
module load FFmpeg/3.4.2-foss-2018a
source activate score_following
python -W ignore -u experiment.py --net ScoreFollowingNetMSMDLCHSDeepDoLight --train_set msmd/msmd_all/msmd_all_train --eval_set msmd/msmd_all/msmd_all_valid --game_config game_configs/mutopia_lchs1.yaml --log_root recurrent_approach/logs --param_root recurrent_approach/params --agent rnn
