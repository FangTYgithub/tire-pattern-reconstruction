# Provide paths and progress records for point-recovery diagnostics.

import sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'scripts' / 'spotnoise_round3'), str(ROOT / 'scripts' / 'spotnoise_suppA'), str(ROOT / 'scripts' / 'spotnoise')]
import sn_lib as SL

OUT = ROOT / 'outputs' / 'spotnoise_round4'
CROSS_CASES = ROOT / 'outputs' / 'spotnoise_cross' / 'cases'
ROUND3 = ROOT / 'outputs' / 'spotnoise_round3'
save, digest, now = SL.save, SL.digest, SL.now
TASK_START = '2026-10-06T17:18:12+02:00'


def progress(**kw):
    p = OUT / 'progress.json'
    cur = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    cur['updated'] = now(); cur['timezone'] = 'Europe/Berlin (local clock; ISO offsets)'
    save(p, cur)
    return cur


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f'{now()} {msg}'
    with (OUT / 'run.log').open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


def cross_slots():

    return [s for s in json.loads((ROUND3 / 'design' / 'slots.json').read_text(encoding='utf-8'))['slots'] if s['exp'] == 'X']
