# Provide result records and input checks for environment-study preparation.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'scripts' / 'surface_revision'
for _q in ('model_prep03', 'thesis_reinforce01', 'thesis_reinforce03', 's345_ready', 'sections_adapter'):
    if str(SR / _q) not in sys.path:
        sys.path.insert(0, str(SR / _q))
import mp3_common as K3

K4 = K3.K4
OUT = ROOT / 'outputs' / 'surface_revision' / 'model_prep04'
MP3_OUT = K3.OUT
OUTSR = ROOT / 'outputs' / 'surface_revision'
DRAFTS = K3.DRAFTS
TASK = DRAFTS / 'MODEL-PREP-04_protocol.md'
REVIEW3 = DRAFTS / 'MODEL-PREP-03_review.md'
TASK3 = K3.TASK
TASK_ID = 'MODEL-PREP-04'
DRILL_PREFIX = 'mp4_drill_'
Stop = K3.Stop
EXIT = K3.EXIT
clean, rel, load_json, read_csv = K3.clean, K3.rel, K3.load_json, K3.read_csv
sha, canon, digest, content_sha, now = K3.sha, K3.canon, K3.digest, K3.content_sha, K3.now


CALLS = dict(render=0, decode=0, evaluation=0)
AUX = dict(a4_identity_relabel=0, ray_cast_stored_features=0, pattern_raster=0)


def drill_base() -> Path:
    return Path(tempfile.gettempdir()).resolve()


class Paths:

    def __init__(self, root: Path = OUT):
        self.root = Path(root).resolve()
        real = self.root == OUT.resolve()
        drill = self.root.parent == drill_base() and self.root.name.startswith(DRILL_PREFIX)
        if not (real or drill):
            raise Stop('refused', f'root {self.root} is neither {rel(OUT)} nor an isolated drill root')
        self.is_real = real
        self.run = self.root / 'run.json'
        self.figs = self.root / 'figures'
        self.ledger = self.root / 'ledger.jsonl'

    def guard(self, p) -> Path:
        q = Path(p).resolve()
        if self.root != q and self.root not in q.parents:
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

    def run_load(self) -> dict:
        return load_json(self.run) if self.run.exists() else dict(task=TASK_ID, task_book=rel(TASK),   created=now())

    def run_put(self, section: str, obj) -> dict:
        r = self.run_load(); r[section] = clean(obj); r['updated'] = now(); self.save_json(self.run, r)
        return r


P0 = Paths(OUT)


def code_files() -> list:
    return sorted(HERE.glob('mp4_*.py'))


def code_identity() -> dict:
    return {rel(p): sha(p) for p in code_files()}
