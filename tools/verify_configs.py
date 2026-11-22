"""Check that the configs in ``configs/`` reproduce the original run scripts.

Each original script is read from a git revision (``--ref``, default
``58aec81``: the last commit before the scripts were consolidated) and
executed with the expensive parts replaced by recorders:

* ``lib.vqe.DVR_VQE`` records its constructor arguments and every call to
  ``get_dvr_vqe`` (molecule parameters, DVR / ansatz / VQE options, log
  directory, noise model, ``excited_states``, ``cont``);
* ``lib.pot_gen.get_pot`` / ``get_pot_cr2`` record which potential is loaded;
* ``numpy.savez`` records which files would be written.

No Hamiltonian is built and no VQE is run. ``scripts/run_dvr_vqe.py`` is then
run on ``configs/<script>.yaml`` under the same recorders and the two
recordings are compared value by value, including Python types (so ``3`` and
``3.0``, or a tuple and a list, count as different).

Directories are compared symbolically: the data, scratch and results
directories are replaced by ``$DATA``, ``$SCRATCH`` and ``$RESULTS``. The
hardcoded cluster paths in the original scripts map to the same symbols
(``.../dvr_vqe/`` -> ``$DATA``, ``.../dvr_vqe/results/`` -> ``$RESULTS``).

Usage::

    python tools/verify_configs.py              # verify every config
    python tools/verify_configs.py --generate   # (re)write configs from the originals
"""

import argparse
import copy
import os
import re
import subprocess
import sys
import tempfile
import traceback
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, 'configs')
DEFAULT_REF = '58aec81'

# Hardcoded cluster locations in the original scripts and what they stand for.
LEGACY_PATTERNS = [
    (re.compile(r'^/cluster/home/[^/]+/projects/bin/dvr_vqe/results/'), '$RESULTS/'),
    (re.compile(r'^/cluster/home/[^/]+/projects/bin/dvr_vqe/'), '$DATA/'),
]

GROUPS = [
    (r'_vqe_block(_64)?$', 'block'),
    (r'_vqe_excited_opt\d(_64)?$', 'excited_opt'),
    (r'_vqe_excited\d*$', 'excited_twolocal'),
    (r'_vqe_greedy$', 'greedy'),
    (r'_vqe_greedyent_const_noisy$', 'greedyent_const_noisy'),
    (r'_vqe_greedyent_const\d*$', 'greedyent_const'),
    (r'_vqe_greedyent\d+$', 'greedyent'),
    (r'_vqe_noent_noisy$', 'noent_noisy'),
    (r'_vqe_noent$', 'noent'),
]


class NonStandardScript(Exception):
    """The script does more than build options and call get_dvr_vqe."""


class Recorder:
    def __init__(self, dirs):
        self.events = []
        self.prefixes = [(dirs['results'], '$RESULTS/'), (dirs['scratch'], '$SCRATCH/'),
                         (dirs['data'], '$DATA/')]

    def norm(self, path):
        if path is None:
            return None
        for prefix, symbol in self.prefixes:
            if path.startswith(prefix):
                return symbol + path[len(prefix):].replace('\\', '/')
        for pattern, symbol in LEGACY_PATTERNS:
            m = pattern.match(path)
            if m:
                return symbol + path[m.end():]
        return path

    # --- stand-ins -------------------------------------------------------
    def make_pot_gen(self):
        import importlib
        cr2_pot_points = importlib.import_module('lib.pot_gen').cr2_pot_points

        rec = self
        mod = types.ModuleType('lib.pot_gen')

        def get_pot(pot_file, take_cos):
            rec.events.append(('get_pot', rec.norm(pot_file), take_cos))
            return ('pot', 'file', rec.norm(pot_file), take_cos)

        def get_pot_cr2(s):
            rec.events.append(('get_pot_cr2', s))
            rs = cr2_pot_points[s][0]
            return ('pot', 'cr2', s), (rs[0], rs[-1])

        def unsupported(*args, **kwargs):
            raise NonStandardScript('calls another lib.pot_gen function')

        mod.get_pot = get_pot
        mod.get_pot_cr2 = get_pot_cr2
        mod.interp_pot2d = unsupported
        mod.pot_fun = unsupported
        return mod

    def make_vqe(self):
        rec = self
        mod = types.ModuleType('lib.vqe')

        class DVR_VQE:
            def __init__(self, mol_params, pot_fun, log_dir=None, noise_model=None, multiprocess=False):
                self._init = dict(mol_params=mol_params, pot=pot_fun, log_dir=rec.norm(log_dir),
                                  noise_model=noise_model, multiprocess=multiprocess)

            def get_dvr_vqe(self, dvr_options, ansatz_options, vqe_options, excited_states=False, cont=0):
                rec.events.append(('get_dvr_vqe', copy.deepcopy(dict(
                    self._init, dvr_options=dvr_options, ansatz_options=ansatz_options,
                    vqe_options=vqe_options, excited_states=excited_states, cont=cont))))
                return ('opt_params',)

            def __getattr__(self, name):
                raise NonStandardScript(f'calls DVR_VQE.{name}')

        def unsupported(*args, **kwargs):
            raise NonStandardScript('calls a lib.vqe function directly')

        mod.DVR_VQE = DVR_VQE
        mod.opt_vqe = unsupported
        return mod

    def savez(self, file, *args, **kwargs):
        self.events.append(('savez', self.norm(file), sorted(kwargs), [repr(v) for v in args],
                            {k: repr(v) for k, v in sorted(kwargs.items())}))


def _install(rec):
    """Replace lib.vqe, lib.pot_gen and numpy.savez with recorders."""
    import numpy as np

    saved = {name: sys.modules.get(name) for name in ('lib.vqe', 'lib.pot_gen')}
    sys.modules['lib.vqe'] = rec.make_vqe()
    sys.modules['lib.pot_gen'] = rec.make_pot_gen()
    import lib
    saved_attrs = {name: getattr(lib, name, None) for name in ('vqe', 'pot_gen')}
    lib.vqe = sys.modules['lib.vqe']
    lib.pot_gen = sys.modules['lib.pot_gen']
    saved_savez = np.savez
    np.savez = rec.savez

    def restore():
        np.savez = saved_savez
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        for name, mod in saved_attrs.items():
            setattr(lib, name, mod)

    return restore


def _fresh_lib():
    """Re-import lib so module-level dicts (e.g. cr2_params) start unmodified."""
    for name in list(sys.modules):
        if name == 'lib' or name.startswith('lib.'):
            del sys.modules[name]


def record_original(source, dirs):
    _fresh_lib()
    rec = Recorder(dirs)
    restore = _install(rec)
    cwd = os.getcwd()
    try:
        os.chdir(REPO_ROOT)
        exec(compile(source, '<original>', 'exec'), {'__name__': '__main__'})
    finally:
        os.chdir(cwd)
        restore()
    return rec.events


def record_runner(config_path, dirs):
    _fresh_lib()
    rec = Recorder(dirs)
    restore = _install(rec)
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))
        import run_dvr_vqe
        run_dvr_vqe.run(run_dvr_vqe.load_config(config_path))
    finally:
        sys.path.pop(0)
        sys.modules.pop('run_dvr_vqe', None)
        restore()
    return rec.events


def strict_diff(a, b, path='root'):
    """Return a list of differences between a and b, comparing types too."""
    if type(a) is not type(b):
        return [f'{path}: type {type(a).__name__} != {type(b).__name__} ({a!r} vs {b!r})']
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f'{path}[{k!r}]: only in {"runner" if k not in a else "original"}')
            else:
                out += strict_diff(a[k], b[k], f'{path}[{k!r}]')
        return out
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} != {len(b)} ({a!r} vs {b!r})']
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += strict_diff(x, y, f'{path}[{i}]')
        return out
    return [] if a == b else [f'{path}: {a!r} != {b!r}']


def group_of(script):
    stem = os.path.splitext(script)[0]
    for pattern, group in GROUPS:
        if re.search(pattern, stem):
            return group
    return None


def config_from_events(script, events):
    """Build a config dict from the recording of an original script."""
    from lib.dvr1d import cr2_params
    from lib.dvr2d import arhcl_params, mgnh_params

    calls = [e[1] for e in events if e[0] == 'get_dvr_vqe']
    pots = [e for e in events if e[0] in ('get_pot', 'get_pot_cr2')]
    saves = [e for e in events if e[0] == 'savez']
    if not calls:
        raise NonStandardScript('never calls get_dvr_vqe')
    if len(pots) != 1:
        raise NonStandardScript(f'loads {len(pots)} potentials')

    first = calls[0]
    shared = ('mol_params', 'pot', 'log_dir', 'noise_model', 'multiprocess',
              'dvr_options', 'vqe_options', 'excited_states', 'cont')
    for c in calls[1:]:
        for key in shared:
            if strict_diff(c[key], first[key]):
                raise NonStandardScript(f'{key} changes between get_dvr_vqe calls')
    if first['multiprocess']:
        raise NonStandardScript('uses multiprocess=True')

    config = {'group': group_of(script), 'source_script': script}
    mol = first['mol_params']
    for name, params in (('arhcl', arhcl_params), ('mgnh', mgnh_params), ('cr2', cr2_params)):
        if not strict_diff(dict(mol, name=params['name']), params):
            config['molecule'] = name
            break
    else:
        raise NonStandardScript('molecule parameters do not match a lib dict')

    pot = pots[0]
    if pot[0] == 'get_pot_cr2':
        config['spin'] = pot[1]
    else:
        if not pot[1].startswith('$DATA/'):
            raise NonStandardScript(f'potential file outside the data directory: {pot[1]}')
        config['potential_file'] = pot[1][len('$DATA/'):]
        config['take_cos'] = pot[2]

    for symbol, root in (('$SCRATCH/', 'scratch'), ('$RESULTS/', 'results')):
        if first['log_dir'].startswith(symbol):
            config['output_root'] = root
            config['log_dir'] = first['log_dir'][len(symbol):]
            break
    else:
        raise NonStandardScript(f"log_dir outside scratch/results: {first['log_dir']}")

    if first['noise_model'] is not None:
        config['noise_model'] = first['noise_model']
    config['excited_states'] = first['excited_states']
    if first['cont']:
        config['cont'] = first['cont']

    if saves:
        paths = {s[1] for s in saves}
        if len(saves) != len(calls) or len(paths) != 1 or not saves[0][1].startswith('$RESULTS/'):
            raise NonStandardScript('saves files in a way the runner does not support')
        config['save_opt_params'] = saves[0][1][len('$RESULTS/'):]

    def plain(x):
        if isinstance(x, tuple):
            return [plain(v) for v in x]
        if isinstance(x, list):
            return [plain(v) for v in x]
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        return x

    config['dvr_options'] = plain(first['dvr_options'])
    config['vqe_options'] = plain(first['vqe_options'])
    if len(calls) == 1:
        config['ansatz_options'] = plain(first['ansatz_options'])
    else:
        config['ansatz_options_list'] = [plain(c['ansatz_options']) for c in calls]
    return config


def write_config(path, config):
    import yaml

    header = (f"# Reproduces {config['source_script']} (group: {config['group']}).\n"
              "# Generated from the original script by tools/verify_configs.py --generate.\n"
              "# Run with: python scripts/run_dvr_vqe.py " + os.path.relpath(path, REPO_ROOT).replace('\\', '/') + "\n")
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(header)
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None, width=100)


def original_scripts(ref):
    out = subprocess.run(['git', 'ls-tree', '--name-only', ref], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    return sorted(f for f in out if f.endswith('.py'))


def read_original(ref, script):
    return subprocess.run(['git', 'show', f'{ref}:{script}'], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True, encoding='utf-8').stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--ref', default=DEFAULT_REF, help='git revision holding the original scripts')
    parser.add_argument('--generate', action='store_true', help='write configs/<script>.yaml from the originals')
    parser.add_argument('scripts', nargs='*', help='only these scripts (default: all root-level .py files)')
    args = parser.parse_args(argv)

    tmp = tempfile.mkdtemp(prefix='dvr_vqe_verify_')
    for var, sub in (('DVR_VQE_DATA_DIR', 'DATA'), ('DVR_VQE_SCRATCH_DIR', 'SCRATCH'),
                     ('DVR_VQE_RESULTS_DIR', 'RESULTS')):
        os.environ[var] = os.path.join(tmp, sub)
    sys.path.insert(0, REPO_ROOT)
    from lib.paths import get_data_dir, get_scratch_dir, get_results_dir
    dirs = {'data': get_data_dir(), 'scratch': get_scratch_dir(), 'results': get_results_dir()}

    scripts = args.scripts or original_scripts(args.ref)
    results = {'match': [], 'mismatch': [], 'not_covered': [], 'error': []}
    for script in scripts:
        name = os.path.splitext(script)[0]
        config_path = os.path.join(CONFIG_DIR, name + '.yaml')
        try:
            orig = record_original(read_original(args.ref, script), dirs)
            if args.generate:
                _fresh_lib()
                sys.modules.pop('lib.vqe', None)
                os.makedirs(CONFIG_DIR, exist_ok=True)
                write_config(config_path, config_from_events(script, orig))
            if not os.path.exists(config_path):
                raise NonStandardScript('no config')
            new = record_runner(config_path, dirs)
        except NonStandardScript as e:
            results['not_covered'].append((script, str(e)))
            continue
        except Exception as e:
            results['error'].append((script, f'{type(e).__name__}: {e}'))
            if os.environ.get('VERIFY_DEBUG'):
                traceback.print_exc()
            continue

        # Each get_dvr_vqe call is compared together with the DVR_VQE
        # arguments in effect for it; potential loads and saves in order.
        diffs = strict_diff(orig, new, 'events')
        (results['mismatch'] if diffs else results['match']).append((script, diffs))

    print(f"Checked {len(scripts)} scripts against {args.ref}\n")
    print(f"MATCH ({len(results['match'])}):")
    for script, _ in results['match']:
        n = sum(1 for e in record_original(read_original(args.ref, script), dirs) if e[0] == 'get_dvr_vqe')
        print(f'  {script:42s} {n} get_dvr_vqe call(s)')
    for key, title in (('mismatch', 'MISMATCH'), ('not_covered', 'NOT COVERED (no config)'),
                       ('error', 'ORIGINAL SCRIPT FAILS')):
        print(f"\n{title} ({len(results[key])}):")
        for script, info in results[key]:
            if isinstance(info, list):
                print(f'  {script}')
                for d in info[:10]:
                    print(f'      {d}')
            else:
                print(f'  {script:42s} {info}')
    return 1 if results['mismatch'] else 0


if __name__ == '__main__':
    sys.exit(main())
