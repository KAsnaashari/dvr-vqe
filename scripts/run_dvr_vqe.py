"""Run a DVR-VQE calculation described by a YAML config file.

Every config in ``configs/`` reproduces one of the original run scripts
(``configs/<script name>.yaml``). The runner builds the molecule parameters
and the potential, creates a :class:`lib.vqe.DVR_VQE` object and calls
:meth:`~lib.vqe.DVR_VQE.get_dvr_vqe` once per ansatz, exactly as the original
script did.

Usage::

    python scripts/run_dvr_vqe.py configs/arhcl_vqe_excited2.yaml
    python scripts/run_dvr_vqe.py CONFIG --scratch-dir /path/to/scratch
    python scripts/run_dvr_vqe.py CONFIG --dry-run   # print the options only

Config keys
-----------
group            Label for the kind of calculation (informational only).
source_script    Name of the original script the config reproduces.
molecule         ``arhcl``, ``mgnh`` or ``cr2``; selects the parameter dict
                 from ``lib.dvr2d`` / ``lib.dvr1d``.
spin             Cr2 only: total spin S of the potential curve. ``_<spin>``
                 is appended to the molecule name, as in the original scripts.
potential_file   ArHCl / MgNH only: potential file, relative to the data
                 directory.
take_cos         ArHCl / MgNH only: passed to ``lib.pot_gen.get_pot``.
output_root      ``scratch`` or ``results``: which directory ``log_dir`` is
                 relative to.
log_dir          Directory (relative to ``output_root``) for the VQE logs.
noise_model      Optional: ``sherbrooke`` for the FakeSherbrooke noise model.
dvr_options      Passed to ``get_dvr_vqe``.
vqe_options      Passed to ``get_dvr_vqe``.
ansatz_options   Passed to ``get_dvr_vqe``. Use ``ansatz_options_list``
                 instead to run several ansatze one after another.
excited_states   Passed to ``get_dvr_vqe`` (default ``false``).
cont             Passed to ``get_dvr_vqe`` to continue a previous run
                 (default ``0``).
save_opt_params  Optional path, relative to the results directory, where the
                 optimal parameters returned by ``get_dvr_vqe`` are saved with
                 ``numpy.savez``.

Directories default to the locations in ``lib/paths.py`` and can be set with
``--data-dir`` / ``--scratch-dir`` / ``--results-dir`` or the environment
variables ``DVR_VQE_DATA_DIR`` / ``DVR_VQE_SCRATCH_DIR`` /
``DVR_VQE_RESULTS_DIR``.
"""

import argparse
import copy
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# YAML has no tuple type; these option values were tuples in the original
# scripts and are converted back so the options are identical.
_TUPLE_KEYS = {'dvr_options': ['box_lims']}


def load_config(path):
    """Read a YAML config file and return it as a dict."""
    import yaml

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _restore_tuples(name, options):
    options = copy.deepcopy(options)
    for key in _TUPLE_KEYS.get(name, []):
        if key in options and isinstance(options[key], list):
            options[key] = tuple(options[key])
    return options


def get_mol_params(config):
    """Return the molecule parameter dict selected by the config."""
    molecule = config['molecule']
    if molecule == 'arhcl':
        from lib.dvr2d import arhcl_params as params
    elif molecule == 'mgnh':
        from lib.dvr2d import mgnh_params as params
    elif molecule == 'cr2':
        from lib.dvr1d import cr2_params as params
    else:
        raise ValueError(f'Unknown molecule: {molecule}')

    mol_params = copy.deepcopy(params)
    if config.get('spin') is not None:
        mol_params['name'] += f"_{config['spin']}"
    return mol_params


def get_potential(config):
    """Return the potential-energy function selected by the config."""
    from lib.paths import get_data_dir

    if config['molecule'] == 'cr2':
        from lib.pot_gen import get_pot_cr2
        pot, _ = get_pot_cr2(config['spin'])
        return pot

    from lib.pot_gen import get_pot
    return get_pot(get_data_dir() + config['potential_file'], take_cos=config['take_cos'])


def _output_root(name):
    from lib.paths import get_scratch_dir, get_results_dir

    if name == 'scratch':
        return get_scratch_dir()
    if name == 'results':
        return get_results_dir()
    raise ValueError(f"output_root must be 'scratch' or 'results', not {name!r}")


def print_exact_energies(config, n=6):
    """Print the lowest exact eigenvalues (cm^-1) of the DVR Hamiltonian.

    Diagonalises the same matrix that ``get_dvr_vqe`` maps onto qubits, as a
    reference for the VQE energies. Does not affect the VQE run.
    """
    import numpy as np
    from lib.utils import hartree
    from lib.vqe import DVR_VQE

    dvr_options = _restore_tuples('dvr_options', config['dvr_options'])
    dvr_vqe = DVR_VQE(get_mol_params(config), get_potential(config))
    energies = np.linalg.eigvalsh(dvr_vqe.get_h_dvr(dvr_options, J=0) * hartree)
    print(f'Exact DVR eigenvalues (cm^-1), lowest {n}: '
          + ', '.join(f'{e:.6f}' for e in energies[:n]))


def run(config, dry_run=False):
    """Run every calculation described by ``config``."""
    import numpy as np
    from lib.vqe import DVR_VQE

    mol_params = get_mol_params(config)
    dvr_options = _restore_tuples('dvr_options', config['dvr_options'])
    vqe_options = _restore_tuples('vqe_options', config['vqe_options'])
    if 'ansatz_options_list' in config:
        ansatz_options_list = config['ansatz_options_list']
    else:
        ansatz_options_list = [config['ansatz_options']]
    log_dir = _output_root(config['output_root']) + config['log_dir']
    run_kwargs = {
        'excited_states': config.get('excited_states', False),
        'cont': config.get('cont', 0),
    }

    if dry_run:
        print(json.dumps({
            'mol_params': mol_params,
            'log_dir': log_dir,
            'noise_model': config.get('noise_model'),
            'dvr_options': dvr_options,
            'vqe_options': vqe_options,
            'ansatz_options_list': ansatz_options_list,
            **run_kwargs,
        }, indent=2, default=str))
        return

    pot = get_potential(config)
    for ansatz_options in ansatz_options_list:
        ansatz_options = _restore_tuples('ansatz_options', ansatz_options)
        dvr_vqe = DVR_VQE(mol_params, pot, log_dir=log_dir, noise_model=config.get('noise_model'))
        opt_params = dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, **run_kwargs)

        if config.get('save_opt_params'):
            np.savez(_output_root('results') + config['save_opt_params'], opt_params=opt_params)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('config', help='YAML config file (see configs/)')
    parser.add_argument('--data-dir', help='directory with the potential files (DVR_VQE_DATA_DIR)')
    parser.add_argument('--scratch-dir', help='directory for scratch output (DVR_VQE_SCRATCH_DIR)')
    parser.add_argument('--results-dir', help='directory for results (DVR_VQE_RESULTS_DIR)')
    parser.add_argument('--dry-run', action='store_true', help='print the resolved options and exit')
    parser.add_argument('--exact', action='store_true',
                        help='after the run, print the exact eigenvalues of the DVR Hamiltonian')
    args = parser.parse_args(argv)

    for value, var in [(args.data_dir, 'DVR_VQE_DATA_DIR'),
                       (args.scratch_dir, 'DVR_VQE_SCRATCH_DIR'),
                       (args.results_dir, 'DVR_VQE_RESULTS_DIR')]:
        if value:
            os.environ[var] = value

    config = load_config(args.config)
    if not args.dry_run:
        # DVR_VQE creates one sub-directory per run but not its parents.
        os.makedirs(_output_root(config['output_root']) + config['log_dir'], exist_ok=True)
    run(config, dry_run=args.dry_run)
    if args.exact and not args.dry_run:
        print_exact_energies(config)


if __name__ == '__main__':
    main()
