# Provide execution records and budget accounting for joint-study runs.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_A3 = HERE.parent / 's5_align03'
if str(_A3) not in sys.path:
    sys.path.insert(0, str(_A3))
import s5r_common as K3

ROOT = K3.ROOT
SR = K3.SR
OUT = SR / 's5_align04'
FORMAL_ROOT = SR / 's5_six_formal'
FORMAL_STORE = FORMAL_ROOT / 'store'
LEDGER = FORMAL_ROOT / 'ledger.jsonl'
ISO = OUT / '_isolation'
RUN = OUT / 'run.json'
A3RUN = SR / 's5_align03' / 'run.json'
TASK = K3.DRAFTS / 'S5-ALIGN-04_protocol.md'
REVIEW = K3.DRAFTS / 'S5-ALIGN-03_review.md'
ALLOWED_ROOTS = (OUT, FORMAL_ROOT)
CAPS = dict(render=8, decode_D0=4, decode_DF=4, decode_P4=4, evaluate=12)
Stop = K3.Stop
EXIT = dict(K3.EXIT, quota=14, auth=15, freeze=16, platform=17)
now, sha, digest, rel, clean, load_json, load_npz, read_csv = K3.now, K3.sha, K3.digest, K3.rel, K3.clean, K3.load_json, K3.load_npz, K3.read_csv


def root_of(p) -> Path:

    q = Path(p).resolve()
    for r in ALLOWED_ROOTS:
        if r.resolve() == q or r.resolve() in q.parents:
            return r
    raise Stop('guard', f'write outside the authorised roots refused: {q}')


def save_json(p, obj) -> Path:
    return K3.save_json(p, obj, root_of(p))


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task='S5-ALIGN-04', task_book=rel(TASK),  acceptance='reference (independent)', created=now())


def run_put(section: str, obj) -> dict:
    r = run_load()
    if section in r.get('fixed', []):
        raise Stop('freeze', f'run.json section {section} is fixed (written before the computation)')
    r[section] = obj; r['updated'] = now(); save_json(RUN, r)
    return r


def run_fix(section: str, obj) -> dict:
    r = run_load()
    if section in r:
        if digest(r[section]) != digest(obj):
            raise Stop('freeze', f'{section} already fixed with a different content')
        return r
    r[section] = obj; r.setdefault('fixed', []).append(section); r.setdefault('fixed_at', {})[section] = now(); r['updated'] = now(); save_json(RUN, r)
    return r


def run_append(section: str, item: dict, keep: int = 60) -> None:
    r = run_load(); lst = r.get(section, []); lst.append(clean(item)); r[section] = lst[-keep:]; r['updated'] = now(); save_json(RUN, r)


class Ledger:
    def __init__(self, path: Path, auth_id: str, caps: dict, keys: set):
        self.path = Path(path); self.auth_id = auth_id; self.caps = dict(caps); self.keys = set(keys)
        root_of(self.path)

    def events(self) -> list:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding='utf-8').splitlines() if x.strip()]

    def counts(self) -> dict:
        c = {k: 0 for k in self.caps}; st = {}
        for e in self.events():
            if e.get('auth') != self.auth_id:
                continue
            if e['status'] == 'reserved':
                c[e['kind']] = c.get(e['kind'], 0) + 1
            st[e['status']] = st.get(e['status'], 0) + 1
        return dict(reserved=c, by_status=st)

    def _write(self, e: dict) -> None:
        root_of(self.path); self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(clean(e), ensure_ascii=False) + '\n'); fh.flush(); os.fsync(fh.fileno())

    def reserve(self, kind: str, key: str, manifest_key: str) -> str:
        if manifest_key not in self.keys:
            raise Stop('auth', f'{manifest_key} is not in the authorised key set')
        if kind not in self.caps or self.caps[kind] <= 0:
            raise Stop('quota', f'no quota for {kind}')
        used = self.counts()['reserved'].get(kind, 0)
        if used >= self.caps[kind]:
            raise Stop('quota', f'{kind}: {used} of {self.caps[kind]} attempts already reserved - refused before the call')
        tok = digest(dict(a=self.auth_id, k=kind, key=key, n=used, t=now(), pid=os.getpid()), 16)
        self._write(dict(auth=self.auth_id, kind=kind, key=key, manifest_key=manifest_key, status='reserved', token=tok, n_before=used, t=now(), pid=os.getpid()))
        return tok

    def close(self, tok: str, kind: str, key: str, status: str, info=None) -> None:
        self._write(dict(auth=self.auth_id, kind=kind, key=key, status=status, token=tok, info=info, t=now(), pid=os.getpid()))
