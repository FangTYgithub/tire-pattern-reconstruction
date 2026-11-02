# Provide common input and output definitions for D0/DF comparisons.

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
HOME = ROOT.parents[1]
SR = ROOT / 'scripts' / 'surface_revision'
OUT = ROOT / 'outputs' / 'surface_revision' / 's3_fe01'
RUN, RESULTS = OUT / 'run', OUT / 'results'
TASK_ID = 'S3-FE-01'
FLOOR_RULE = 'DF'
_GUARD = {'on': False}


class Stop(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f'[{code}] {msg}'); self.code = code


def install_guard() -> None:

    if _GUARD['on']:
        return
    allowed = OUT.resolve()

    def ok(p: Path) -> bool:
        try:
            p.relative_to(allowed); return True
        except ValueError:
            return False

    def guard(event, args):
        if event == 'open':
            raw, mode, flags = args
            if not isinstance(raw, (str, bytes, os.PathLike)):
                return
            p = Path(os.fsdecode(raw)).resolve()
            if p.suffix == '.pyc':
                raise FileNotFoundError('source-only import')
            if (flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC):
                if (p.is_relative_to(ROOT) or p.is_relative_to(HOME / 'outputs')) and not ok(p):
                    raise RuntimeError(f'{TASK_ID}: write outside the task folder denied: {p}')
        elif event in ('os.remove', 'os.rename', 'os.rmdir', 'os.chmod', 'os.utime', 'os.truncate'):
            for raw in (args[:2] if event == 'os.rename' else args[:1]):
                if isinstance(raw, (str, bytes, os.PathLike)):
                    p = Path(os.fsdecode(raw)).resolve()
                    if (p.is_relative_to(ROOT) or p.is_relative_to(HOME / 'outputs')) and not ok(p):
                        raise RuntimeError(f'{TASK_ID}: mutation outside the task folder denied: {p}')
    sys.addaudithook(guard); _GUARD['on'] = True


def paths_s12() -> None:

    p = str(SR / 's3_code04')
    if p not in sys.path:
        sys.path.insert(0, p)
    import c4_common
    for q in ('s345_ready', 's3_code01', 's3_code02'):
        s = str(SR / q)
        if s not in sys.path:
            sys.path.append(s)


def paths_s4() -> None:
    for q in ('s4_real07', 's4_real02', 's345_ready', 'sections_adapter', 's12_resource01'):
        s = str(SR / q)
        if s not in sys.path:
            sys.path.insert(0, s)
    import s4r7_common


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def digest(x, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(clean(x), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:n]


def sha_file(p) -> str:
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_json(p) -> dict:
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def load_npz(p) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_json(p, obj) -> None:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'); os.replace(tmp, p)


def save_npz(p, **arrays) -> str:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays); os.replace(tmp, p); return sha_file(p)


def save_csv(df, p) -> None:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False, encoding='utf-8')


def ledger(stage: str, kind: str, key: str, status: str, **kw) -> None:
    d = RUN / stage; d.mkdir(parents=True, exist_ok=True)
    with open(d / f'ledger_w{os.getpid()}.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(dict(t=now(), kind=kind, key=key, status=status, **clean(kw)), ensure_ascii=False) + '\n')


def ledger_counts(stage: str) -> dict:
    d = RUN / stage; c: dict = {}
    if d.exists():
        for p in d.glob('ledger_w*.jsonl'):
            for line in p.read_text(encoding='utf-8').splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = f"{r.get('kind')}:{r.get('status')}"; c[k] = c.get(k, 0) + 1
    return c


def eq_arr(a, b) -> bool:
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        try:
            a = a.astype(b.dtype)
        except (TypeError, ValueError):
            return False
    if np.issubdtype(b.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def metric_diffs(m_new: dict, m_old: dict, keys) -> dict:
    out = {}
    for k in keys:
        a, b = m_new.get(k), m_old.get(k)
        if a is None and b is None:
            continue
        if isinstance(a, (int, float, np.generic)) and isinstance(b, (int, float, np.generic)):
            fa, fb = float(a), float(b)
            if (np.isnan(fa) and np.isnan(fb)) or abs(fa - fb) <= 1e-9:
                continue
        elif a == b:
            continue
        out[k] = (b, a)
    return out


METRICS = ('state', 'n_out', 'n_accepted', 'n_valid', 'n_selected', 'n_detected', 'n_candidates', 'n_matched', 'n_ambiguous', 'n_duplicate', 'n_no_correspondence',
           'n_confirmable_features', 'id_correct', 'id_wrong', 'id_unconf', 'img_recovery', 'glob_n_valid', 'glob_Ctau', 'glob_Cvalid', 'glob_cov_interp', 'glob_recovery',
           'glob_pt_rmse', 'glob_pt_p95', 'glob_pt_max', 'glob_surf_rmse', 'glob_gross_rate', 'loc_n_valid', 'loc_Ctau', 'loc_Cvalid', 'loc_recovery', 'loc_pt_rmse', 'loc_pt_p95',
           'glob_n_nodes')

CHECK = ('state', 'n_accepted', 'n_valid', 'id_correct', 'id_wrong', 'id_unconf', 'glob_Ctau', 'glob_Cvalid', 'glob_cov_interp', 'glob_recovery', 'glob_pt_rmse', 'glob_pt_p95',
         'glob_surf_rmse', 'loc_Ctau', 'loc_Cvalid', 'loc_pt_rmse')
