# Normalize encoding records into the common result schema.

from __future__ import annotations

from pathlib import Path

import numpy as np

import c5_common as K

SUB = dict(decode='dec', eval='eval', diag='diag')


# Read layered result records and check file completeness and content identities.
class FileStore:
    kind = 'files'

    def __init__(self, root):
        self.root = Path(root) if Path(root).is_absolute() else K.ROOT / root

    def path(self, layer, key) -> Path:
        return self.root / SUB[layer] / f'{key}.json'

    def inspect(self, layer, key):
        p = self.path(layer, key)
        if not p.exists():
            return 'absent', None
        try:
            r = K.load_json(p)
        except Exception:
            return 'unreadable', None
        if r.get('complete') is not True:
            return 'not_complete', r
        if r.get('identity') != key:
            return 'identity_mismatch', r
        if r.get('kind') == 'ref':
            src = K.ROOT / r.get('source_npz', '')
            if not (r.get('source_npz') and src.exists() and K.sha_file(src) == r.get('source_sha256')):
                return 'source_npz_hash', r
            if r.get('source_json'):
                if not r.get('source_json_sha256'):
                    return 'source_json_unregistered', r
                sj = K.ROOT / r['source_json']
                if not (sj.exists() and K.sha_file(sj) == r['source_json_sha256']):
                    return 'source_json_hash', r
            return 'ok', r
        z = p.with_suffix('.npz')
        if not (z.exists() and K.sha_file(z) == r.get('npz_sha256')):
            return 'npz_hash', r
        return 'ok', r

    def has_arrays(self, layer, key) -> bool:
        return True

    def arrays(self, layer, key, rec) -> dict:
        p = (K.ROOT / rec['source_npz']) if rec.get('kind') == 'ref' else self.path(layer, key).with_suffix('.npz')
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def other_records(self, layer, expected_keys: set):
        d = self.root / SUB[layer]
        for p in (d.glob('*.json') if d.exists() else []):
            if p.stem not in expected_keys:
                try:
                    yield p.stem, K.load_json(p)
                except Exception:
                    yield p.stem, None

    def fingerprint(self) -> str:
        return K.dir_digest([self.root / s for s in SUB.values()])

    def describe(self) -> dict:
        return dict(kind=self.kind, root=K.rel(self.root))


class MemStore:

    kind = 'memory'

    def __init__(self, records: dict, arrays: dict | None = None, spec_digest: str = ''):
        self.records, self._arrays, self.spec_digest = records, arrays or {}, spec_digest

    def inspect(self, layer, key):
        r = self.records.get((layer, key))
        if r is None:
            return 'absent', None
        if r.get('complete') is not True:
            return 'not_complete', r
        if r.get('identity') != key:
            return 'identity_mismatch', r
        return 'ok', r

    def arrays(self, layer, key, rec) -> dict:
        a = self._arrays.get((layer, key))
        if a is None:
            raise K.Stop('technical', f'test store holds no arrays for {layer}/{key}')
        return a

    def has_arrays(self, layer, key) -> bool:
        return (layer, key) in self._arrays

    def other_records(self, layer, expected_keys: set):
        for (ly, k), r in self.records.items():
            if ly == layer and k not in expected_keys:
                yield k, r

    def fingerprint(self) -> str:
        return self.spec_digest

    def describe(self) -> dict:
        return dict(kind=self.kind, spec_digest=self.spec_digest, n_records=len(self.records), n_arrays=len(self._arrays))
