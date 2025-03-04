import os
import sys
import argparse
from pathlib import Path
import numpy as np
import datasets

ap = argparse.ArgumentParser()
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
args = ap.parse_args()

SEED = 42
np.random.seed(SEED)

with open('../data/ibl_session_ids.txt', 'r') as f:
    eids = f.read().splitlines()  # removes newlines
    eids = [eid.strip() for eid in eids if eid.strip()]  # remove empty lines and whitespace

valid_eids = []
for eid_idx, eid in enumerate(eids):

    print(f"Processing {eid} ({eid_idx+1} / {len(eids)})")

    dataset = datasets.load_from_disk(Path(args.base_path)/"ibl_aligned"/eid)
    neuron_regions = np.array(dataset["train"]["cluster_regions"])[0]
    unique_regions = np.unique(neuron_regions)

    print(unique_regions)

    for re_idx, re_name in enumerate(unique_regions):
        if "PO" in re_name:
            unique_regions[re_idx] = "PO"

        if "DG" in re_name:
            unique_regions[re_idx] = "DG"

        if ("VISa" in re_name) or ("VISam" in re_name):
            unique_regions[re_idx] = "VISa"

    is_valid = []
    for region in ["PO", "LP", "DG", "CA1", "VISa"]:
        if region in unique_regions:
            is_valid.append(True)
        else:
            is_valid.append(False)

    if all(is_valid):
        print(eid)
        print(unique_regions)
        valid_eids.append(eid)


for eid in valid_eids:
    print(eid)
