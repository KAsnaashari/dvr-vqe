"""Locations of input data and run output.

All paths default to directories inside the repository and can be overridden
with environment variables (the runner in ``scripts/`` also accepts
command-line flags that set these variables):

``DVR_VQE_DATA_DIR``
    Directory holding the potential-energy files ``arhcl.txt`` and
    ``mgnh.txt``. Default: the repository root.
``DVR_VQE_SCRATCH_DIR``
    Directory for scratch output (VQE logs and ``.npz`` files).
    Default: ``<repo>/scratch``.
``DVR_VQE_RESULTS_DIR``
    Directory for results kept long term. Default: ``<repo>/results``.

Every path is returned as a string ending in a path separator, so it can be
concatenated with a relative path the way the run scripts do
(``scratch_dir + 'excited_states/'``).
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dir_from_env(var, default):
    path = os.environ.get(var) or default
    return os.path.join(os.path.abspath(path), '')


def get_data_dir():
    """Return the directory containing the potential-energy data files."""
    return _dir_from_env('DVR_VQE_DATA_DIR', REPO_ROOT)


def get_scratch_dir():
    """Return the directory for scratch output."""
    return _dir_from_env('DVR_VQE_SCRATCH_DIR', os.path.join(REPO_ROOT, 'scratch'))


def get_results_dir():
    """Return the directory for results kept long term."""
    return _dir_from_env('DVR_VQE_RESULTS_DIR', os.path.join(REPO_ROOT, 'results'))
