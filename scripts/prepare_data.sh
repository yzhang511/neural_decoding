#!/bin/bash
#SBATCH --account=stats             
#SBATCH --job-name="prepare"
#SBATCH --output="prepare.%j.out"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=10G
#SBATCH --time=0-1:00
#SBATCH --export=ALL

module load anaconda

. ~/.bashrc
echo $TMPDIR
conda activate decoding

session_id=${1}

cd ..
python src/allen_visual_behavior_neuropixels/prepare_data.py \
    --session_id $session_id \
    --data_dir /burg/stats/users/yz4123/allen/datasets/ 

conda deactivate

