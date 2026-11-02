# Provide source identities and storage paths for detector diagnostics.

import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_n, '1')
import sys, json, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'scripts' / 'spotnoise_round4'), str(ROOT / 'scripts' / 'spotnoise_round3'),
                str(ROOT / 'scripts' / 'spotnoise_suppA'), str(ROOT / 'scripts' / 'spotnoise'), str(ROOT / 'src'), str(ROOT)]
import sn_lib as SL

OUT = ROOT / 'outputs' / 'spotnoise_round5'
CROSS_CASES = ROOT / 'outputs' / 'spotnoise_cross' / 'cases'
VAL_DIR = ROOT / 'outputs' / 'spotnoise_round4' / 'validation'
ROUND3 = ROOT / 'outputs' / 'spotnoise_round3'
TASK_START = '2026-10-06T20:00:57+02:00'
GROUPS = ('D0', 'D1', 'D2')
save, now = SL.save, SL.now


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def progress(**kw):
    p = OUT / 'progress.json'
    cur = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    cur['updated'] = now(); cur['timezone'] = 'Europe/Berlin (local clock; ISO offsets)'; cur.setdefault('task_start', TASK_START)
    save(p, cur)
    return cur


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f'{now()} {msg}'
    with (OUT / 'run.log').open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


def vlabel(v):
    return f'{float(1.0 if v is None else v):.2f}'


def sources():

    out = []
    slots = json.loads((ROUND3 / 'design' / 'slots.json').read_text(encoding='utf-8'))['slots']
    for s in slots:
        if s['exp'] != 'X':
            continue
        sp = s['spec']
        out.append(dict(dataset='X', key=s['key'], surface=sp['surface'], phase=int(sp['phase']), r_mm=float(sp['r_mm']), vlabel=vlabel(sp['v']),
                        sigma=float(sp['sigma']), seed=int(sp['seed']), noise_base=int(sp['noise_base']),
                        json=str((CROSS_CASES / f"{s['key']}.json").relative_to(ROOT)), npz=str((CROSS_CASES / f"{s['key']}.npz").relative_to(ROOT))))
    cl = json.loads((VAL_DIR / 'case_list.json').read_text(encoding='utf-8'))
    for sp, k in zip(cl['specs'], cl['keys']):
        out.append(dict(dataset='V', key=k, surface=sp['surface'], phase=int(sp['phase']), r_mm=float(sp['r_mm']), vlabel=vlabel(sp['v']),
                        sigma=float(sp['sigma']), seed=int(sp['seed']), noise_base=int(sp['noise_base']),
                        json=str((VAL_DIR / 'cases' / f'{k}.json').relative_to(ROOT)), npz=str((VAL_DIR / 'cases' / f'{k}.npz').relative_to(ROOT))))
    for o in out:
        o['json'] = o['json'].replace('\\', '/'); o['npz'] = o['npz'].replace('\\', '/')
        o['noise_seed'] = None if o['sigma'] == 0 else int(o['noise_base'] + 10 * o['phase'] + o['seed'])
    return out


def ref_key(surface, phase, r_mm, vl):

    for s in sources():
        if s['dataset'] == 'X' and s['sigma'] == 0 and s['surface'] == surface and s['phase'] == phase and s['r_mm'] == r_mm and s['vlabel'] == vl:
            return s['key']
    return None
