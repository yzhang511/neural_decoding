#!/bin/bash
#SBATCH --account=stats             
#SBATCH --job-name="data"
#SBATCH --output="data.%j.out"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=10G
#SBATCH --time=0-1:00
#SBATCH --export=ALL

module load anaconda
. ~/.bashrc
conda activate decoding

session_id=${1}

# Step 1: Download data
cd /burg/stats/users/yz4123/neural_decoding
python src/allen_visual_behavior_neuropixels/download_data.py \
    --session_id $session_id \
    --output_dir /burg/stats/users/yz4123/allen/datasets/raw/


# Step 2: Prepare data
cd /burg/stats/users/yz4123/neural_decoding
python src/allen_visual_behavior_neuropixels/prepare_data.py \
    --session_id $session_id \
    --data_dir /burg/stats/users/yz4123/allen/datasets/ 

# Step 3: Delete raw data
rm -rf "/burg/stats/users/yz4123/allen/datasets/raw/session_${session_id}"

conda deactivate
