#!/bin/bash
#SBATCH --account=stats             
#SBATCH --job-name="data"
#SBATCH --output="data.%j.out"
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

python src/download_data.py \
       --eid $eid \
       --base_path /burg/stats/users/yz4123/Downloads

conda deactivate

cd ../scripts
