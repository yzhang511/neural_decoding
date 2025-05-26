target=${1}
method=${2}
region=${3}
search=${4}
use_nlb=${5}
bin_size=${6}
fold_idx=${7}

while IFS= read -r session_id; do
    # Skip empty lines and comments
    if [ -z "$session_id" ] || [[ "$session_id" =~ ^#.* ]]; then
        continue
    fi

    echo "Submitting job for session $session_id"
    sbatch decode_single_session_cpu.sh $session_id $target $method $region $search $use_nlb $bin_size $fold_idx

    sleep 1
done < "/burg/stats/users/yz4123/neural_decoding/data/repro_ephys_test.txt"

echo "All sessions have been submitted!"
