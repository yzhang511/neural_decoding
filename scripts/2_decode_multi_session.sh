#!/bin/bash

#SBATCH --job-name=decode_multi
#SBATCH --output=decode_multi-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-1:00:00 
#SBATCH --mem=16g

. ~/.bashrc
conda activate decoding

cd ..

python src/2_decode_multi_session.py --target choice --region all --base_path XXX 

cd script

conda deactivate