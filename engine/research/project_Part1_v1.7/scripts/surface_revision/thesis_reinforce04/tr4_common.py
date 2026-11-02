# Provide content-bound storage and shared helpers for sub-ray comparisons.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'scripts' / 'surface_revision'
TR3 = SR / 'thesis_reinforce03'
if str(TR3) not in sys.path:
    sys.path.insert(0, str(TR3))
import tr3_common as K3

OUT = ROOT / 'outputs' / 'surface_revision' / 'thesis_reinforce04'
OUT1, OUT2, OUT3 = K3.OUT1, K3.OUT2, K3.OUT
RUN = OUT / 'run.json'
FIGS = OUT / 'figures'
DRAFTS = K3.DRAFTS
TASK = DRAFTS / 'THESIS-REINFORCE-04_protocol.md'
REVIEW3 = DRAFTS / 'THESIS-REINFORCE-03_review.md'
TASK_ID = 'THESIS-REINFORCE-04'
CAPS = dict(render_compat=24, render_3x3=24, render=48, decode_3x3=42, decode_1x1_DF=18, decode=60, evaluation=60, old_common_evaluation=0)
NUMERIC_FILES = ('tr4_common.py', 'tr4_gate.py', 'tr4_prep.py', 'tr4_chain.py', 'tr4_pairs.py', 'tr4_main.py')
LAYOUT_FILES = ('tr4_fig.py',)
DRILL_FILES = ('tr4_drills.py',)
clean, rel, load_json, load_npz, read_csv = K3.clean, K3.rel, K3.load_json, K3.load_npz, K3.read_csv
EXIT = dict(ok=0, usage=2, refused=20, preflight=21, budget=22, compat_stop=30, technical=40, result_integrity=41, interrupted=42)


class Stop(RuntimeError):


    def __init__(self, code: str, msg: str):
        super().__init__(f'[{code}] {msg}')
        self.code = code if code in EXIT else 'technical'
        self.msg = msg
        self.exit_code = EXIT[self.code]


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canon(obj) -> str:
    return json.dumps(clean(obj), sort_keys=True, ensure_ascii=False, default=str)


def digest(obj) -> str:
    return hashlib.sha256(canon(obj).encode('utf-8')).hexdigest()


def content_sha(a) -> str:

    a = np.asarray(a); h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def arrays_digest(arrays: dict) -> str:
    return digest({k: content_sha(arrays[k]) for k in sorted(arrays)})


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() != q and OUT.resolve() not in q.parents:
        raise Stop('refused', f'write outside {rel(OUT)} refused: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def _atomic(p, write) -> Path:
    q = guard(p); t = q.with_name(q.name + f'.tmp{os.getpid()}')
    write(t)
    for _ in range(100):
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


def save_pack(p, meta: dict, arrays: dict) -> str:

    def w(t):
        with open(t, 'wb') as fh:
            np.savez_compressed(fh, _meta=np.array(canon(meta)), **arrays)
    return sha(_atomic(p, w))


def load_pack(p) -> tuple[dict, dict]:
    with np.load(p, allow_pickle=False) as z:
        A = {k: np.asarray(z[k]) for k in z.files}
    meta = json.loads(str(A.pop('_meta')))
    return meta, A


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task=TASK_ID, task_book=rel(TASK),   created=now())


def run_put(section: str, obj) -> dict:
    r = run_load(); r[section] = clean(obj); r['updated'] = now(); save_json(RUN, r)
    return r


def code_identity(files=NUMERIC_FILES) -> dict:
    return {f: sha(HERE / f) for f in files}


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
                d = out.setdefault(grp, dict(reserved=0, ok=0, failed=0, interrupted=0, mismatch=0, open=0))
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


    def index(self) -> dict:
        return load_json(self.index_path) if self.index_path.exists() else {}

    def index_put(self, rel_name: str, h: str, **info) -> None:
        ix = self.index(); ix[rel_name] = dict(sha256=h, t=now(), **info); save_json(self.index_path, ix)
