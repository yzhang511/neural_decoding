"""Script adapted from this notebook: 
https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html
"""

import argparse
import requests
import os

from tqdm import tqdm
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


def delete_directory(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))  
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))  
    os.rmdir(directory)  


if __name__ == "__main__":
    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="./raw", help="Output directory"
    )
    parser.add_argument("--session_id", type=int, default=None)
    args = parser.parse_args()

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    # download data for session
    truncated_file = True
    directory = os.path.join(args.output_dir + "/session_" + str(args.session_id))

    while truncated_file:
        session = cache.get_session_data(args.session_id)
        try:
            print(session.specimen_name)
            truncated_file = False
        except OSError:
            delete_directory(directory)
            print(" Truncated spikes file, re-downloading")
