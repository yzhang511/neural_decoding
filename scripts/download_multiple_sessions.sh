while IFS= read -r session_id; do
    # Skip empty lines and comments
    if [ -z "$session_id" ] || [[ "$session_id" =~ ^#.* ]]; then
        continue
    fi

    echo "Submitting job for session $session_id"
    sbatch download_data.sh $session_id

    sleep 1

done < "/burg/stats/users/yz4123/neural_decoding/data/repro_ephys_release.txt"

echo "All sessions have been submitted!"
