# Provide paths, content identities and stores for environment cases.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'scripts' / 'surface_revision'
for _q in ('thesis_reinforce04', 'thesis_reinforce05', 'model_prep02'):
    if str(SR / _q) not in sys.path:
        sys.path.insert(0, str(SR / _q))
import tr4_common as K4

OUT = ROOT / 'outputs' / 'surface_revision' / 'model_prep03'
MP2_OUT = ROOT / 'outputs' / 'surface_revision' / 'model_prep02'
DRAFTS = K4.DRAFTS
TASK = DRAFTS / 'MODEL-PREP-03_protocol.md'
TASK_ID = 'MODEL-PREP-03'
CAPS = dict(forward=520, forward_entry=8, decode=780, decode_D0=260, decode_DF=260, decode_P4=260, evaluation=780)
Stop = K4.Stop
EXIT = K4.EXIT
clean, rel, load_json, load_npz, read_csv = K4.clean, K4.rel, K4.load_json, K4.load_npz, K4.read_csv
sha, canon, digest, content_sha, arrays_digest, now = K4.sha, K4.canon, K4.digest, K4.content_sha, K4.arrays_digest, K4.now


# Resolve run-local paths and perform checked atomic writes.
class Paths:

    def __init__(self, root: Path = OUT):
        self.root = Path(root)
        if OUT.resolve() != self.root.resolve() and OUT.resolve() not in self.root.resolve().parents:
            raise Stop('refused', f'root outside {rel(OUT)}: {self.root}')
        self.run = self.root / 'run.json'
        self.data = self.root / 'data'
        self.figs = self.root / 'figures'
        self.ledger = self.root / 'ledger.jsonl'
        self.sandbox = self.root / '_sandbox'

    def guard(self, p) -> Path:
        q = Path(p).resolve()
        if self.root.resolve() != q and self.root.resolve() not in q.parents:
            raise Stop('refused', f'write outside {self.root} refused: {q}')
        q.parent.mkdir(parents=True, exist_ok=True)
        return q

    def atomic(self, p, write) -> Path:
        q = self.guard(p); t = q.with_name(q.name + f'.tmp{os.getpid()}')
        write(t)
        for _ in range(200):
            try:
                os.replace(t, q)
                return q
            except PermissionError:
                time.sleep(0.05)
        raise Stop('technical', f'cannot replace {q}')

    def save_json(self, p, obj) -> Path:
        return self.atomic(p, lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1, default=str), encoding='utf-8'))

    def save_csv(self, df, p) -> Path:
        return self.atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))

    def save_text(self, p, text: str) -> Path:
        return self.atomic(p, lambda t: t.write_text(text, encoding='utf-8'))

    def save_pack(self, p, meta: dict, arrays: dict) -> str:
        def w(t):
            with open(t, 'wb') as fh:
                np.savez_compressed(fh, _meta=np.array(canon(meta)), **arrays)
        return sha(self.atomic(p, w))

    def run_load(self) -> dict:
        return load_json(self.run) if self.run.exists() else dict(task=TASK_ID, task_book=rel(TASK),   created=now())

    def run_put(self, section: str, obj) -> dict:
        r = self.run_load(); r[section] = clean(obj); r['updated'] = now(); self.save_json(self.run, r)
        return r


P0 = Paths(OUT)


def load_pack(p) -> tuple[dict, dict]:
    return K4.load_pack(p)


def finite_int(x, name: str) -> int:
    if isinstance(x, (bool, np.bool_)) or x is None:
        raise Stop('preflight', f'{name} {x!r} is not an integer')
    try:
        f = float(x)
    except (TypeError, ValueError):
        raise Stop('preflight', f'{name} {x!r} is not a number') from None
    if not np.isfinite(f) or f != round(f):
        raise Stop('preflight', f'{name} {x!r} is not a finite integer')
    return int(round(f))


# Track completed stages and compatible result identities for resuming cases.
class Store:
    def __init__(self, paths: Paths, caps: dict | None = None):
        self.P = paths
        self.caps = dict(CAPS if caps is None else caps)

    @property
    def root(self):
        return self.P.root

    def _append(self, rec: dict) -> None:
        q = self.P.guard(self.P.ledger)
        with open(q, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(clean(rec), ensure_ascii=False, default=str) + chr(10)); fh.flush(); os.fsync(fh.fileno())

    def rows(self) -> list:
        if not self.P.ledger.exists():
            return []
        return [json.loads(x) for x in self.P.ledger.read_text(encoding='utf-8').splitlines() if x.strip()]

    def counts(self) -> dict:
        R = self.rows(); res = {r['id']: r for r in R if r.get('event') == 'reserve'}
        end = {r['id']: r['status'] for r in R if r.get('event') == 'end'}
        out = {}
        for i, r in res.items():
            for grp in r['groups']:
                d = out.setdefault(grp, dict(reserved=0, ok=0, failed=0, interrupted=0, aborted=0, open=0))
                d['reserved'] += 1
                st = end.get(i) or 'open'
                d[st] = d.get(st, 0) + 1
        return out

    def unclosed(self) -> list:
        R = self.rows(); ends = {r['id'] for r in R if r.get('event') == 'end'}
        return [r['id'] for r in R if r.get('event') == 'reserve' and r['id'] not in ends]

    def reserve(self, groups: tuple, key: str, **info) -> str:
        c = self.counts()
        for g in groups:
            used = (c.get(g) or {}).get('reserved', 0)
            if used + 1 > int(self.caps[g]):
                raise Stop('budget', f'{g}: {used} reserved calls, cap {self.caps[g]} (refused before the call)')
        rid = f'{groups[0]}:{key}:{time.time_ns()}'
        self._append(dict(event='reserve', id=rid, groups=list(groups), key=key, t=now(), pid=os.getpid(), **info))
        return rid

    def end(self, rid: str, status: str, **info) -> None:
        self._append(dict(event='end', id=rid, status=status, t=now(), **info))

    def note(self, kind: str, **info) -> None:
        self._append(dict(event='note', kind=kind, t=now(), **info))

    @property
    def index_path(self):
        return self.P.data / 'index.json'

    def index(self) -> dict:
        return load_json(self.index_path) if self.index_path.exists() else {}

    def index_put(self, rel_name: str, h: str, **info) -> None:
        ix = self.index(); ix[rel_name] = dict(sha256=h, t=now(), **info); self.P.save_json(self.index_path, ix)
