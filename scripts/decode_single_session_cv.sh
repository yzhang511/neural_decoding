#!/bin/bash
#SBATCH --account=stats             
#SBATCH --job-name="single_session"
#SBATCH --output="single_session.%j.out"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1        
#SBATCH --mem 100000
#SBATCH --time=0-2:00
#SBATCH --export=ALL

export TMPDIR=/local

# Load env
module load anaconda
python --version

session_id=${1}
target=${2}
method=${3}

. ~/.bashrc
echo $TMPDIR
conda activate decoding

cd ..

python src/decode_single_session_cv.py \
    --session_id $session_id \
    --target $target \
    --method $method \
    --base_path /burg/stats/users/yz4123/allen/

conda deactivate

