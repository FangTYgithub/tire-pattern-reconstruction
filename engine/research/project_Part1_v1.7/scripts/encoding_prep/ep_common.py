# Provide file, hash and checkpoint helpers for encoding experiments.

import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_n, '1')
import sys, json, hashlib, time, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'scripts' / 'spacing_layout'), str(ROOT / 'scripts' / 'spotnoise_round5'), str(ROOT / 'scripts' / 'spotnoise_round4'),
                str(ROOT / 'scripts' / 'spotnoise_round3'), str(ROOT / 'scripts' / 'spotnoise_suppA'), str(ROOT / 'scripts' / 'spotnoise'),
                str(ROOT / 'scripts' / 'pm015'), str(ROOT / 'scripts' / 'pm014'), str(ROOT / 'scripts' / 'pm013'), str(ROOT / 'src'), str(ROOT)]

OUT = ROOT / 'outputs' / 'encoding_prep'
DOCS = ROOT / 'docs' / 'encoding_prep'
CASES = OUT / 'smoke_cases'
FIRST_TIMESTAMP_UTC = '2026-10-08T13:20:27Z'


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def now_local():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode('utf-8')).hexdigest()


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
    _atomic(p, lambda t: df.to_csv(t, index=False, **kw))


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def progress(**kw):
    p = OUT / 'progress.json'
    cur = load_json(p) if p.exists() else {}
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    cur['updated_utc'] = utcnow(); cur['updated_local'] = now_local(); cur.setdefault('first_timestamp_utc', FIRST_TIMESTAMP_UTC)
    cur.setdefault('note', 'times are real clock readings (UTC + local ISO); stages are marked done only after their outputs exist')
    save_json(p, cur)
    return cur


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f'{utcnow()} {msg}'
    with (OUT / 'run.log').open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')
    print(line, flush=True)


class Stage:

    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.monotonic(); self.start = utcnow()
        progress(stage=self.name, stages={self.name: dict(state='running', started_utc=self.start)})
        log(f'stage {self.name}: start')
        return self
    def __exit__(self, et, ev, tb):
        st = 'done' if et is None else 'error'
        progress(stages={self.name: dict(state=st, started_utc=self.start, finished_utc=utcnow(), seconds=round(time.monotonic() - self.t0, 1),
                                         error=None if et is None else f'{et.__name__}: {ev}')})
        log(f'stage {self.name}: {st} ({time.monotonic() - self.t0:.0f} s)')
        return False
