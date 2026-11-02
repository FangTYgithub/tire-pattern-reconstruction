"""Select numerical inputs and a separate output directory for thesis figures."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

PACKAGE = Path(__file__).resolve().parents[1]
_state = None


def configure(group):
    """Parse plotting paths and prevent writes outside the chosen output directory."""
    global _state
    if _state is not None:
        return _state['root'], _state['output']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PACKAGE/'engine')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='+', default=[], help='Figure IDs, e.g. Abb_3_01.')
    parser.add_argument('--verify-reference', action='store_true', help='Require exact archived input hashes.')
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    if output == root or output.is_relative_to(root) or output.is_relative_to(PACKAGE/'engine'):
        raise ValueError('Use an output directory outside the numerical input tree.')
    output.mkdir(parents=True, exist_ok=True)
    cache = output/'cache'; cache.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
    sys.dont_write_bytecode = True
    _state = dict(root=root, output=output, group=group, only=set(args.only), exact=args.verify_reference)
    flags_write = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

    def guard(event, values):
        if event == 'open':
            name, _, flags = values
            paths = [name] if (flags or 0) & flags_write else []
        elif event in ('os.remove','os.rename','os.mkdir','os.rmdir','os.chmod','os.utime'):
            paths = values[:2] if event == 'os.rename' else values[:1]
        else:
            return
        for name in paths:
            if isinstance(name,(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(name)).resolve()
                if event=='os.mkdir' and p.is_dir():
                    continue
                if not p.is_relative_to(output):
                    raise PermissionError('Write outside figure output: '+str(p))
    sys.addaudithook(guard)
    return root, output


def selected(name):
    return not _state['only'] or name in _state['only']


def verify_inputs():
    """Check declared result files; exact historical hashes are optional for reruns."""
    manifest=json.loads((PACKAGE/'provenance/thesis_inventory.json').read_text('utf-8'))
    missing=[]; mismatched=[]
    for item in manifest['figure_inputs']:
        if _state['group'] not in item['readers']:
            continue
        p=_state['root']/item['path']
        if not p.is_file():
            missing.append(item['path'])
        elif _state['exact'] and hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            mismatched.append(item['path'])
    if missing or mismatched:
        raise ValueError('Numerical result inputs required; missing='+str(missing[:8])+', hash mismatch='+str(mismatched[:8]))
