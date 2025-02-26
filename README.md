Decode IBL unaligned datasets.


### Environment setup

Create conda environment:
```
conda env create -f env.yaml
```

Activate the environment:
```
conda activate ibl_repro_ephys
```

Clone the IBL reproducible ephys repo:
```
git clone https://github.com/int-brain-lab/paper-reproducible-ephys.git
```

Install requirements and repo:
```
cd paper-reproducible-ephys
pip install -e .
```


### Datasets

NOTE: Replace the `--base_path` in the following slurm scripts with your own storage path.

Run the following to preprocess IBL unaligned data:

```
cd scripts
sbatch 0_data_filtering.sh
```

```
sbatch 1_data_caching.sh EID
```

Run the following to decode a single session using linear model:
```
sbatch 2_decode_single_session_cpu.sh EID BEHAVIOR MODEL
```
Example:
```
sbatch 2_decode_single_session_cpu.sh d23a44ef-1402-4ed7-97f5-47e9a7a504d9 wheel-speed linear
```

Run the following to decode a single session using RRR while doing hyperparameter tuning:
```
sbatch 2_decode_single_session_tune.sh EID BEHAVIOR MODEL
```
Example:
```
sbatch 2_decode_single_session_tune.sh d23a44ef-1402-4ed7-97f5-47e9a7a504d9 wheel-speed reduced_rank True
```
Here, `True` indicates that hyperparameter tuning is enabled.

Optionally, you can run the following to decode a single session using RRR w/o hyperparameter tuning:
```
sbatch 2_decode_single_session_gpu.sh d23a44ef-1402-4ed7-97f5-47e9a7a504d9 wheel-speed reduced_rank
```

### Multi-Session

First, run hyperparameter tuning on a 10-session model to find the best model hyperparameters. The first step is to run `1_data_caching.sh` to download the data from the 10 sessions. Then, run the following:
```
sbatch 3_decode_multi_session_tune.sh wheel-speed True
```
From the output, find the best model hyperparameters and manually change `configs/decoder.yaml` to use the best hyperparameters.

Then, fit the model on more IBL sessions (download the data before running the following):
```
sbatch 3_decode_multi_session_gpu.sh wheel-speed
```
Remember to change the requested time or GPU numbers in the slurm script.


