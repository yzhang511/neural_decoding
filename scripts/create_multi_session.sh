partition=${1}

while IFS= read -r session_id; do
    # Skip empty lines and comments
    if [ -z "$session_id" ] || [[ "$session_id" =~ ^#.* ]]; then
        continue
    fi

    echo "Submitting job for session $session_id"
    sbatch create_single_session.sh $session_id $partition

    sleep 1

done < "/burg/stats/users/yz4123/neural_decoding/data/allen_test_ids.txt"

echo "All sessions have been submitted!"
