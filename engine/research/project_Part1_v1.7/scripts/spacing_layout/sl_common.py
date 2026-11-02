# Provide storage and checkpoint helpers for spacing and layout experiments.

import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_n, '1')
import sys, json, hashlib, time, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs' / 'spacing_layout_prep'
TASK_START = '2026-10-07T02:07:00+02:00'


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _atomic(p, write):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    write(tmp)
    tmp.replace(p)


def save_json(p, obj):
    _atomic(p, lambda t: t.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding='utf-8'))


def save_text(p, text):
    _atomic(p, lambda t: t.write_text(text, encoding='utf-8'))


def save_csv(df, p, **kw):

    comp = 'gzip' if str(p).endswith('.gz') else None
    _atomic(p, lambda t: df.to_csv(t, index=False, compression=comp, **kw))


def progress(**kw):
    p = OUT / 'progress.json'
    cur = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    cur['updated'] = now(); cur['timezone'] = 'Europe/Berlin (local clock; ISO offsets)'; cur.setdefault('task_start', TASK_START)
    save_json(p, cur)
    return cur


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f'{now()} {msg}'
    with (OUT / 'run.log').open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


class Stage:

    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.monotonic(); self.start = now()
        progress(stage=self.name, stages={self.name: dict(state='running', started=self.start)})
        log(f'stage {self.name}: start')
        return self
    def __exit__(self, et, ev, tb):
        st = 'done' if et is None else 'error'
        progress(stages={self.name: dict(state=st, started=self.start, finished=now(), seconds=round(time.monotonic() - self.t0, 1),
                                         error=None if et is None else f'{et.__name__}: {ev}')})
        log(f'stage {self.name}: {st} ({time.monotonic() - self.t0:.0f} s)')
        return False
