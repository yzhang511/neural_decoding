#!/bin/bash
#SBATCH --account=stats             
#SBATCH --job-name="cache"
#SBATCH --output="cache.%j.out"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=20G       
#SBATCH --time=0-1:00              

module load anaconda

. ~/.bashrc
eid=${1}

echo $TMPDIR

conda activate decoding

cd /burg/stats/users/yz4123/neural_decoding

python src/1_data_caching.py \
    --base_path /burg/stats/users/yz4123/Downloads \
    --fold_idx 5 \
    --eid $eid \
    --n_workers 1

conda deactivate

cd scripts
