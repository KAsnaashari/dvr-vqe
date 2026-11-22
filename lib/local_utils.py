import numpy as np

from .paths import get_data_dir, get_scratch_dir, get_results_dir

# Kept for scripts that import from lib.local_utils; same values as lib.utils.
base_dir = get_data_dir()
scratch_dir = get_scratch_dir()
results_dir = get_results_dir()
