target=${1}
region=${2}

while IFS= read -r session_id; do
    # Skip empty lines and comments
    if [ -z "$session_id" ] || [[ "$session_id" =~ ^#.* ]]; then
        continue
    fi

    echo "Submitting job for session $session_id"
    sbatch decode_single_session_cpu.sh $session_id $target linear $region False

    sleep 1

done < "/burg/stats/users/yz4123/neural_decoding/data/region_session_ids.txt"

echo "All sessions have been submitted!"
