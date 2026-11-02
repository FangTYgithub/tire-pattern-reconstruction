# Define environment-study paths, execution limits and shared storage helpers.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'scripts' / 'surface_revision'
for _q in ('model_prep04', 'model_prep03', 'thesis_reinforce01', 'thesis_reinforce03', 's345_ready', 'sections_adapter', 's5_align04'):
    if str(SR / _q) not in sys.path:
        sys.path.insert(0, str(SR / _q))
import mp3_common as K3

OUT = ROOT / 'outputs' / 'surface_revision' / 'model_prep05'
MP3_OUT = K3.OUT
MP4_OUT = ROOT / 'outputs' / 'surface_revision' / 'model_prep04'
OUTSR = ROOT / 'outputs' / 'surface_revision'
DRAFTS = K3.DRAFTS
REVIEW4 = DRAFTS / 'MODEL-PREP-04_review.md'
TASK4 = DRAFTS / 'MODEL-PREP-04_protocol.md'
TASK_ID = 'MODEL-PREP-05'
TASK_NOTE = 'task book given in the conversation (2026-10-19), not stored as a file; basis: MODEL-PREP-04 reference review'
DRILL_PREFIX = 'mp5_drill_'
CAPS = dict(forward=2450, forward_entry=136, decode=3626, decode_D0=1176, decode_DF=1176, decode_P4=1274, evaluation=3626,
            replay_decode=32, replay_evaluation=32)
REPLAY_MAX_IMAGES = 20
Stop = K3.Stop
EXIT = K3.EXIT
clean, rel, load_json, read_csv = K3.clean, K3.rel, K3.load_json, K3.read_csv
sha, canon, digest, content_sha, arrays_digest, now = K3.sha, K3.canon, K3.digest, K3.content_sha, K3.arrays_digest, K3.now
load_pack = K3.load_pack
Store = K3.Store


def drill_base() -> Path:
    return Path(tempfile.gettempdir()).resolve()


# Resolve output paths and provide checked atomic storage operations.
class Paths:

    def __init__(self, root: Path = OUT):
        self.root = Path(root).resolve()
        real = self.root == OUT.resolve()
        drill = self.root.parent == drill_base() and self.root.name.startswith(DRILL_PREFIX)
        if not (real or drill):
            raise Stop('refused', f'root {self.root} is neither {rel(OUT)} nor an isolated drill root')
        self.is_real = real
        self.run = self.root / 'run.json'
        self.data = self.root / 'data'
        self.figs = self.root / 'figures'
        self.ledger = self.root / 'ledger.jsonl'
        self.sandbox = self.root / '_sandbox'

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

    def save_text(self, p, text: str) -> Path:
        return self.atomic(p, lambda t: t.write_text(text, encoding='utf-8'))

    def save_pack(self, p, meta: dict, arrays: dict) -> str:
        def w(t):
            with open(t, 'wb') as fh:
                np.savez_compressed(fh, _meta=np.array(canon(meta)), **arrays)
        return sha(self.atomic(p, w))

    def run_load(self) -> dict:
        return load_json(self.run) if self.run.exists() else dict(task=TASK_ID, task_note=TASK_NOTE, review_basis=rel(REVIEW4), 
                                                                    created=now())

    def run_put(self, section: str, obj) -> dict:
        r = self.run_load(); r[section] = clean(obj); r['updated'] = now(); self.save_json(self.run, r)
        return r


P0 = Paths(OUT)
