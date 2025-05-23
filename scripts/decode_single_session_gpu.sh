#!/bin/bash
#SBATCH --account=stats             
#SBATCH --job-name="single_session"
#SBATCH --output="single_session.%j.out"
#SBATCH --gres=gpu:1   
#SBATCH --constraint=rtx8000
#SBATCH -c 1       
#SBATCH --mem 100000
#SBATCH --time=0-2:00:00
#SBATCH --export=ALL

export TMPDIR=/local

# Load env
module load anaconda
module load cuda11.1/toolkit
python --version

eid=${1}
target=${2}
method=${3}
region=${4}
search=${5}
use_nlb=${6}
bin_size=${7}
fold_idx=${8}

echo "eid: $eid"
echo "target: $target"
echo "fold_idx: $fold_idx"

if [ "$use_nlb" = "True" ]; then
    echo "Use NLB Data"
    use_nlb="--use_nlb"
else
    use_nlb=""
fi


if [ "$search" = "True" ]; then
    echo "Doing hyperparameter search"
    search="--search"

    set -x

    # Getting the node names
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)

    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    # if we detect a space character in the head node IP, we'll
    # convert it to an ipv4 address. This step is optional.
    if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
      head_node_ip=${ADDR[1]}
    else
      head_node_ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
    fi

    # Starting the Ray head node
    port="8889"
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    echo "Starting HEAD at $head_node"
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "1" --num-gpus "1" --block &

    # Starting the Ray worker nodes
    sleep 10

    worker_num=$((SLURM_JOB_NUM_NODES - 1))  # number of nodes other than the head node

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus "1" --num-gpus "1" --block &
        sleep 5
    done
else
    echo "Not doing hyperparameter search"
    search=""
fi

. ~/.bashrc
echo $TMPDIR
conda activate decoding

cd ..

python src/decode_single_session.py \
    --eid $eid \
    --target $target \
    --method $method \
    --region $region \
    --base_path /burg/stats/users/yz4123/Downloads/ \
    --n_workers "$SLURM_CPUS_PER_TASK" \
    $search \
    $use_nlb \
    --bin_size $bin_size \
    --fold_idx $fold_idx

conda deactivate
