import sys
from pathlib import Path
path_root = '../'
sys.path.append(str(path_root))

from utils.ibl_data_utils import save_imposter_sessions

"""
-----------
USER INPUTS
-----------
"""

ap = argparse.ArgumentParser()
ap.add_argument("--base_dir", type=str)
ap.add_argument("--n_samples", type=int, default=10)
args = ap.parse_args()

"""
---------------
CACHE IMPOSTERS
---------------
"""

base_dir = Path(args.base_dir)
imposter_dir = base_dir/'imposter'
save_imposter_sessions(data_dir, imposter_dir, n_samples=args.n_samples)

