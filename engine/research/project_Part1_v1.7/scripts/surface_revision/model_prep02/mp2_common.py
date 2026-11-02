# Store environment-study inputs and results with dependency checks.

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
for _q in ('thesis_reinforce04', 'thesis_reinforce05'):
    if str(SR / _q) not in sys.path:
        sys.path.insert(0, str(SR / _q))
import tr4_common as K4

OUT = ROOT / 'outputs' / 'surface_revision' / 'model_prep02'
RUN = OUT / 'run.json'
FIGS = OUT / 'figures'
DATA = OUT / 'data'
DRAFTS = K4.DRAFTS
TASK = DRAFTS / 'MODEL-PREP-02_protocol.md'
TASK_ID = 'MODEL-PREP-02'
CAPS = dict(forward=20, forward_D1=12, forward_D2=8, decode=36, evaluation=36)
NUMERIC_FILES = ('mp2_common.py', 'mp2_forward.py', 'mp2_cases.py', 'mp2_chain.py', 'mp2_main.py')
ZERO_IMAGING_FILES = ('mp2_params.py', 'mp2_geometry.py', 'mp2_checks.py')
LAYOUT_FILES = ('mp2_figs.py',)
Stop = K4.Stop
EXIT = K4.EXIT
clean, rel, load_json, load_npz, read_csv = K4.clean, K4.rel, K4.load_json, K4.load_npz, K4.read_csv
sha, canon, digest, content_sha, arrays_digest, now = K4.sha, K4.canon, K4.digest, K4.content_sha, K4.arrays_digest, K4.now


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() != q and OUT.resolve() not in q.parents:
        raise Stop('refused', f'write outside {rel(OUT)} refused: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def _atomic(p, write) -> Path:
    q = guard(p); t = q.with_name(q.name + f'.tmp{os.getpid()}')
    write(t)
    for _ in range(200):
        try:
            os.replace(t, q)
            return q
        except PermissionError:
            time.sleep(0.05)
    raise Stop('technical', f'cannot replace {q}')


def save_json(p, obj) -> Path:
    return _atomic(p, lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1, default=str), encoding='utf-8'))


def save_csv(df, p) -> Path:
    return _atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))


def save_text(p, text: str) -> Path:
    return _atomic(p, lambda t: t.write_text(text, encoding='utf-8'))


def save_pack(p, meta: dict, arrays: dict) -> str:
    def w(t):
        with open(t, 'wb') as fh:
            np.savez_compressed(fh, _meta=np.array(canon(meta)), **arrays)
    return sha(_atomic(p, w))


def load_pack(p) -> tuple[dict, dict]:
    return K4.load_pack(p)


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task=TASK_ID, task_book=rel(TASK),   created=now())


def run_put(section: str, obj) -> dict:
    r = run_load(); r[section] = clean(obj); r['updated'] = now(); save_json(RUN, r)
    return r


def code_identity(files=NUMERIC_FILES + ZERO_IMAGING_FILES + LAYOUT_FILES) -> dict:
    return {f: (sha(HERE / f) if (HERE / f).exists() else None) for f in files}


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


# Track completed computation stages and their file identities for resumable execution.
class Store:
    def __init__(self, root: Path, caps: dict | None = None):
        self.root = guard(Path(root) / '.root').parent
        self.data = self.root / 'data'
        self.ledger_path = self.root / 'ledger.jsonl'
        self.index_path = self.data / 'index.json'
        self.caps = dict(CAPS if caps is None else caps)

    def _append(self, rec: dict) -> None:
        q = guard(self.ledger_path)
        with open(q, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(clean(rec), ensure_ascii=False, default=str) + chr(10)); fh.flush(); os.fsync(fh.fileno())

    def rows(self) -> list:
        if not self.ledger_path.exists():
            return []
        return [json.loads(x) for x in self.ledger_path.read_text(encoding='utf-8').splitlines() if x.strip()]

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

    def index(self) -> dict:
        return load_json(self.index_path) if self.index_path.exists() else {}

    def index_put(self, rel_name: str, h: str, **info) -> None:
        ix = self.index(); ix[rel_name] = dict(sha256=h, t=now(), **info); save_json(self.index_path, ix)
