Decode the full IBL reproducible ephys (RE) datasets.


### Environment setup

Create conda environment:
```
conda env create -f env.yaml
```

Activate the environment:
```
conda activate decoding
```


### Quickstart

NOTE: Replace the `--base_path` in the following slurm scripts with your own storage path.

Run the following to preprocess IBL aligned data:

```
sbatch download_data.sh EID
```
or if you are not using hpc:
```
python src/preprocess/download_data.py --eid EID --base_path PATH
```


Run the following to decode a single session using linear model on CPU:
```
sbatch decode_single_session_cpu.sh EID BEHAVIOR MODEL
```
Example:
```
sbatch decode_single_session_cpu.sh d23a44ef-1402-4ed7-97f5-47e9a7a504d9 choice linear
```
or if you are not using hpc:
```
python src/decode_single_session.py --eid EID --target BEHAVIOR --method MODEL --base_path PATH --search
```
Here, `--search` indicates that hyperparameter tuning is enabled; remove it if you don't want to do hyperparameter tuning.


Run the following to decode a single session using RRR on multiple GPUs while doing hyperparameter tuning:
```
sbatch decode_single_session_tune.sh EID BEHAVIOR MODEL
```
Example:
```
sbatch decode_single_session_tune.sh d23a44ef-1402-4ed7-97f5-47e9a7a504d9 choice reduced_rank True
```
Here, `True` indicates that hyperparameter tuning is enabled.

Optionally, you can run the following to decode a single session using RRR on a single GPUw/o hyperparameter tuning:
```
sbatch decode_single_session_gpu.sh d23a44ef-1402-4ed7-97f5-47e9a7a504d9 choice reduced_rank
```
