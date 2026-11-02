# Verify the hashes of the packaged engine files and compute the delivery binding.

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# Check packaged file hashes and return the binding used to identify compatible results.
def verify():
    manifest = json.loads((ROOT/'provenance/engine_files.json').read_text('utf-8'))
    failures = []
    for rel, item in manifest.items():
        path = (ROOT/'engine'/rel).resolve()
        if not path.is_relative_to(ROOT/'engine') or not path.is_file():
            failures.append(rel)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            failures.append(rel)
    if failures:
        raise ValueError('Source/input hash mismatch: '+str(failures[:10]))
    digest = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode())

    for file in [ROOT/'run.py', *sorted((ROOT/'tools').glob('*.py'))]:
        digest.update(file.relative_to(ROOT).as_posix().encode())
        digest.update(file.read_bytes())
    return {'files':len(manifest),'binding':digest.hexdigest(),'status':'passed'}
