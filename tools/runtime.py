# Configure output paths, temporary caches and access restrictions for a run.

from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'engine'
PROJECT = ENGINE / 'research/project_Part1_v1.7'
SR = PROJECT / 'scripts/surface_revision'


# Create run-local caches and restrict writes to the selected output directory.
def setup(output):
    output = Path(output).resolve()
    if output == ROOT or output.is_relative_to(ENGINE):
        raise ValueError('Choose an output directory outside engine/.')
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(ENGINE)
    sys.dont_write_bytecode = True
    for name in ('matplotlib', 'numba', 'tmp'):
        (output / 'cache' / name).mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(output / 'cache/matplotlib')
    os.environ['NUMBA_CACHE_DIR'] = str(output / 'cache/numba')
    os.environ['TMP'] = os.environ['TEMP'] = str(output / 'cache/tmp')
    for rel in json.loads((ROOT / 'provenance/runtime_directories.json').read_text('utf-8')):
        target = (ENGINE / rel).resolve()
        if not target.is_relative_to(ENGINE):
            raise ValueError('Invalid runtime directory')
        target.mkdir(parents=True, exist_ok=True)
    flags_write = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

    def guard(event, args):
        if event == 'open':
            name, _, flags = args
            if not isinstance(name, (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(name)).resolve()
            if (flags or 0) & flags_write and not path.is_relative_to(output):
                raise PermissionError('Write outside selected output: ' + str(path))
        elif event in ('os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.chmod', 'os.utime'):
            for name in (args[:2] if event == 'os.rename' else args[:1]):
                if isinstance(name, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(name)).resolve()
                    if event == 'os.mkdir' and path.is_dir():
                        continue
                    if not path.is_relative_to(output):
                        raise PermissionError('Mutation outside selected output: ' + str(path))
    sys.addaudithook(guard)
    return output


def path(name):
    directory = SR / name
    sys.path.insert(0, str(directory))


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
