# Provide shared settings and storage paths for extended environment cases.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AR_DIR = ROOT / 'scripts' / 'surface_revision' / 'model_prep06ar'
if str(AR_DIR) not in sys.path:
    sys.path.insert(0, str(AR_DIR))
import r_common as R

K5, K6 = R.K5, R.K6
OUT = ROOT / 'outputs' / 'surface_revision' / 'model_prep06c'
DATA = OUT / 'data'
SANDBOX = OUT / '_sandbox'
LOGS = OUT / 'logs'
SRC5, SRC6, SRCR = R.SRC5, R.SRC6, R.OUT
TASK_ID = 'MODEL-PREP-06C (self-arranged night run)'
FREEZE = OUT / 'freeze.json'
PROTOCOL = OUT / 'protocol.json'
LEDGER = OUT / 'ledger.jsonl'
CAPS = dict(render_compat=4, render_new=3164, render_holdout=384, decode_new=3164, decode_holdout=384, decode_total=3552)
Stop = R.Stop
now, rel, absp, sha_file, undef, num, clean, digest = R.now, R.rel, R.absp, R.sha_file, R.undef, R.num, R.clean, R.digest


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() != q and OUT.resolve() not in q.parents:
        raise Stop('refused', f'write outside {rel(OUT)} refused: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def atomic(p, write) -> Path:
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
    return atomic(p, lambda t: t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'))


def save_csv(df, p) -> Path:
    return atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))


load_json = R.load_json


def protocol() -> dict:
    return load_json(PROTOCOL) if PROTOCOL.exists() else dict(task=TASK_ID,   created=now(),
                                                              authorisation='user request 2026-10-21 ~02:30: self-arranged overnight run to finish the research tasks and converge the conclusions')


def protocol_put(section: str, obj) -> dict:
    p = protocol(); p[section] = clean(obj); p['updated'] = now(); save_json(PROTOCOL, p)
    return p


def set_sandbox_env() -> None:
    for k, s in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
        os.environ[k] = str(SANDBOX / s)
    os.environ['LINE01RB_RENDER_CAP'] = '0'


class Ledger:
    def __init__(self):
        self.n = {}
        if LEDGER.exists():
            for line in LEDGER.read_text(encoding='utf-8').splitlines():
                r = json.loads(line)
                if r.get('event') == 'reserve':
                    self.n[r['group']] = self.n.get(r['group'], 0) + 1

    def _w(self, rec):
        guard(LEDGER)
        with open(LEDGER, 'a', encoding='utf-8') as f:
            f.write(json.dumps(clean(rec), ensure_ascii=False) + '\n')

    def count(self, *g) -> int:
        return sum(self.n.get(x, 0) for x in g)

    def reserve(self, group: str, key: str) -> str:
        caps = {'render_compat': ('render_compat',), 'render_new': ('render_new',), 'render_holdout': ('render_holdout',), 'decode_new': ('decode_new',),
                'decode_holdout': ('decode_holdout',), 'decode_total': ('decode_new', 'decode_holdout')}
        for cap, members in caps.items():
            if group in members and self.count(*members) + 1 > CAPS[cap]:
                raise Stop('budget', f'{group} {key}: cap {cap} = {CAPS[cap]} reached')
        rid = digest(dict(g=group, k=key, t=time.time_ns()))[:16]
        self._w(dict(event='reserve', rid=rid, group=group, key=key, t=now())); self.n[group] = self.n.get(group, 0) + 1
        return rid

    def end(self, rid: str, status: str, **kw):
        self._w(dict(event='end', rid=rid, status=status, t=now(), **kw))


def verify_freeze(stage: str) -> dict:
    if not FREEZE.exists():
        raise Stop('freeze', f'{stage}: no freeze.json')
    fz = load_json(FREEZE)
    bad = sorted(r for r, h in fz['files'].items() if (sha_file(absp(r)) if absp(r).exists() else None) != h)
    if bad:
        raise Stop('freeze', f'{stage}: {len(bad)} frozen file(s) changed since freeze v{fz["version"]}: {bad[:6]}')
    return dict(version=fz['version'], n_files=len(fz['files']), t=now())
