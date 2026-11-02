# Load complete encoding records with their image and repetition identities.

from __future__ import annotations

from pathlib import Path

import c6_common as K
from c5_store import SUB, FileStore


class FileStore6(FileStore):
    def __init__(self, root, hashes: K.Hashes):
        super().__init__(root); self.H = hashes; self.seen = {}

    def inspect(self, layer, key):
        p = self.path(layer, key)
        if not p.exists():
            return self._note(layer, key, 'absent', None, p)
        try:
            r = K.load_json(p)
        except Exception:
            return self._note(layer, key, 'unreadable', None, p)
        if r.get('complete') is not True:
            return self._note(layer, key, 'not_complete', r, p)
        if r.get('identity') != key:
            return self._note(layer, key, 'identity_mismatch', r, p)
        if r.get('kind') == 'ref':
            src = K.ROOT / r.get('source_npz', '')
            if not (r.get('source_npz') and self.H.sha(src) == r.get('source_sha256')):
                return self._note(layer, key, 'source_npz_hash', r, p, src)
            if r.get('source_json'):
                if not r.get('source_json_sha256'):
                    return self._note(layer, key, 'source_json_unregistered', r, p)
                sj = K.ROOT / r['source_json']
                if self.H.sha(sj) != r['source_json_sha256']:
                    return self._note(layer, key, 'source_json_hash', r, p, sj)
            return self._note(layer, key, 'ok', r, p, src)
        z = p.with_suffix('.npz')
        if self.H.sha(z) != r.get('npz_sha256'):
            return self._note(layer, key, 'npz_hash', r, p, z)
        return self._note(layer, key, 'ok', r, p, z)

    def _note(self, layer, key, st, rec, p, src=None):
        self.seen[(layer, key)] = (st, self.H.sha(p) if p.exists() else None, (self.H.sha(src) if src is not None and Path(src).exists() else None))
        return st, rec

    def fingerprint(self) -> str:

        return K.digest(sorted((l_, k, v[0], v[1], v[2]) for (l_, k), v in self.seen.items()), 24)

    def describe(self) -> dict:
        return dict(kind='files', root=K.rel(self.root), files_hashed_this_run=self.H.n_files)
