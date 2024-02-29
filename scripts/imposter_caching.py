import argparse
from pathlib import Path
from shared_decoding.utils.ibl_data_utils import save_imposter_sessions

"""
-----------
USER INPUTS
-----------
"""

ap = argparse.ArgumentParser()
ap.add_argument("--base_dir", type=str)
ap.add_argument("--n_samples", type=int, default=10)
args = ap.parse_args()

base_dir = Path(args.base_dir)
imposter_dir = base_dir/'imposter'

"""
---------------
CACHE IMPOSTERS
---------------
"""

save_imposter_sessions(data_dir, imposter_dir, n_samples=args.n_samples)

