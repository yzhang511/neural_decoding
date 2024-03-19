#!/bin/bash

#SBATCH --account=stats             
#SBATCH --job-name=me_09b2c
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G
#SBATCH --time=5-00:00              
#SBATCH --output=me_09b2c.out

# Load env
module load anaconda
python --version

set -x

#conda create --name shared_decoding python=3.11
conda activate shared_decoding

#cd /burg/home/yz4123/decoding/shared_decoding/
#pip install -r requirements.txt


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
port=6500
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &

# Starting the Ray worker nodes
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))  # number of nodes other than the head node

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done

echo "Start job :"`date`

cd /burg/home/yz4123/decoding/shared_decoding/scripts

python -u decode_single_behave_decomp_slurm.py --n_workers "$SLURM_CPUS_PER_TASK"

echo "Stop job :"`date`

conda deactivate

