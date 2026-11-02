# Provide shared storage, dependency and execution-budget helpers for layout cases.

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / 'scripts' / 'surface_revision' / 's4_real02', ROOT / 'scripts' / 'surface_revision' / 's4_real05', HERE,
           ROOT / 'scripts' / 'spatial_layout' / 'spat03', ROOT / 'scripts' / 'spatial_layout' / 'spat01r', ROOT / 'scripts' / 'spatial_layout'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import r2_common as K

SR = ROOT / 'outputs' / 'surface_revision'
OUT = SR / 's4_real06'
RUN = OUT / 'run.json'
FIGS, RUNDIR, IMG6, DEC6, EVL6, ISO = OUT / 'figures', OUT / 'run', OUT / 'images', OUT / 'decodes', OUT / 'evals', OUT / '_isolation'
R2, R5 = SR / 's4_real02', SR / 's4_real05'
DRAFTS = ROOT.parents[1] / 'protocols'
TASK = DRAFTS / 'S4-REAL-06_protocol.md'
REVIEW = DRAFTS / 'S4-REAL-05_review.md'
reference_R05 = {'outputs/surface_revision/s4_real05/report_zh.md': 'ffd936d3ff5064ade0c14362d944a4cf4a45e4d56aa11899ede0e9ac2e0c52d0',
             'outputs/surface_revision/s4_real05/run.json': '57db05c7a01574d3475b187ab20054405c0bca2a6b0ca81aeeb56f3febca5129',
             'outputs/surface_revision/s4_real05/replay_metrics.csv': '07475507b6c15bbdb68755e9b2dfaa05f347b8ef024967b6ea811acf755e12f6',
             'outputs/surface_revision/s4_real05/checks.csv': '62b9c3f8fc533b63cf953fcb6c857e04c4eb62d091a892ed555b4aa97a073eb7',
             'outputs/surface_revision/s4_real05/plan.csv': 'f42a26e2cd94e3820c5ce8e483c9e2dd04694a19c1ad934398913baaf876d14e'}
reference_R02 = {'outputs/surface_revision/s4_real02/manifest.csv': 'c9689e279f089ba7f42e337a1300394131fee7a3add8115a9392e7d9956980a1',
             'outputs/surface_revision/s4_real02/layouts.json': '20f28d4bc275c615224a9ae89135028b9815d96f3444947ee085a40a58728f16'}
Q = 11695.0
CAP_RENDER = 90
CAP_DECODE = 180
DECODE_PURPOSE = dict(science_dot=48, science_p4=36, science_clip=72, entry_regression=12, technical=12)
RENDER_PURPOSE = dict(science=84, retry=6)
GROSS_MM = 1.0

READ: dict = {}


class Stop(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f'[{code}] {msg}'); self.code = code; self.msg = msg


def reg(p) -> Path:
    p = Path(p); r = K.rel(p)
    if r not in READ and p.exists():
        READ[r] = K.sha_file(p)
    return p


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in q.parents and q != OUT.resolve():
        raise Stop('guard', f'write outside {OUT} refused: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def _atomic(p, write) -> Path:
    q = guard(p); t = q.with_name(q.name + f'.tmp{os.getpid()}'); write(t)
    for _ in range(100):
        try:
            os.replace(t, q); return q
        except PermissionError:
            time.sleep(0.05)
    raise Stop('technical', f'atomic replace failed for {q}')


def save_json(p, obj) -> Path:
    return _atomic(p, lambda t: t.write_text(json.dumps(K.clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'))


def save_csv(df: pd.DataFrame, p) -> Path:
    return _atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))


def save_npz(p, **arrays) -> str:
    _atomic(p, lambda t: np.savez_compressed(open(t, 'wb'), **arrays))
    return K.sha_file(p)


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def complete(p, identity: str, required=()) -> bool:

    p = Path(p)
    if not p.exists():
        return False
    try:
        r = load_json(p)
    except Exception:
        return False
    if r.get('complete') is not True or r.get('identity') != identity or any(k not in r for k in required):
        return False
    z = p.with_suffix('.npz')
    return z.exists() and K.sha_file(z) == r.get('npz_sha256')


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task='S4-REAL-06', task_book=K.rel(TASK),  acceptance='reference (independent)', created=K.now_local())


def run_put(section: str, obj, freeze: bool = False) -> dict:
    r = run_load(); fz = r.get('frozen', {})
    if section in fz:
        raise Stop('freeze', f'run.json section {section} is frozen; never rewritten')
    if freeze:
        r.setdefault('frozen', {})[section] = dict(digest=K.digest(obj, 32), t_local=K.now_local())
    r[section] = obj; r['updated'] = K.now_local(); save_json(RUN, r)
    return r


def run_verify_frozen() -> dict:
    r = run_load(); bad = [s for s, v in r.get('frozen', {}).items() if K.digest(r.get(s), 32) != v['digest']]
    if bad:
        raise Stop('freeze', f'frozen run.json sections changed: {bad}')
    return dict(ok=True, sections=sorted(r.get('frozen', {})))


def append(section: str, rec: dict) -> None:
    r = run_load(); r.setdefault(section, []).append(K.clean(rec)); r['updated'] = K.now_local(); save_json(RUN, r)


def flush_reads(section: str) -> None:
    r = run_load(); ir = r.setdefault('inputs_read', {})
    for k, v in READ.items():
        if k in ir and ir[k] != v:
            raise Stop('input', f'{k} changed during the run ({section})')
        ir.setdefault(k, v)
    r['updated'] = K.now_local(); save_json(RUN, r)


def part(name: str, df: pd.DataFrame) -> None:
    save_csv(df, OUT / '_parts' / f'{name}.csv')


class Budget:
    def __init__(self, kind: str, cap: int, purposes: dict, base: Path | None = None):
        self.kind, self.cap, self.purposes = kind, cap, purposes
        b = base or RUNDIR
        self.f = b / f'budget_{kind}.json'; self.a = b / f'attempts_{kind}.jsonl'

    def state(self) -> dict:
        return load_json(self.f) if self.f.exists() else dict(kind=self.kind, cap=self.cap, used=0, by_purpose={})

    def reserve(self, key: str, purpose: str) -> int:
        b = self.state()
        if int(b['cap']) != self.cap:
            raise Stop('budget', f"{self.kind} cap {b['cap']} != {self.cap}")
        if purpose not in self.purposes:
            raise Stop('budget', f'{self.kind}: unknown purpose {purpose}')
        if b['used'] >= b['cap']:
            raise Stop('budget', f"{self.kind} cap exhausted ({b['used']} / {b['cap']})")
        if b['by_purpose'].get(purpose, 0) >= self.purposes[purpose]:
            raise Stop('budget', f'{self.kind} purpose {purpose} cap {self.purposes[purpose]} exhausted')
        b['used'] += 1; b['by_purpose'][purpose] = b['by_purpose'].get(purpose, 0) + 1
        save_json(self.f, b); self.log(dict(n=b['used'], key=key, purpose=purpose, event='reserve'))
        return b['used']

    def log(self, rec: dict) -> None:
        guard(self.a)
        with open(self.a, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(K.clean(dict(rec, t=K.now_local())), ensure_ascii=False) + '\n')


def render_budget(base=None) -> Budget:
    return Budget('render', CAP_RENDER, RENDER_PURPOSE, base)


def decode_budget(base=None) -> Budget:
    return Budget('decode', CAP_DECODE, DECODE_PURPOSE, base)


DECODE_DEPS = {'D0': ['scripts/surface_revision/s345_ready/rdy_decode.py', 'src/m4_reconstruction.py', 'scripts/spotnoise_round5/r5_detect.py', 'scripts/surface_revision/sections_adapter/adp_surf02.py',
                      'scripts/surface_revision/sections_adapter/adp_binding.py', 'scripts/surface_revision/surf02/s2_common.py', 'scripts/spatial_layout/spat01r/sp2_common.py'],
               'P4TABLE': ['scripts/surface_revision/s345_ready/rdy_decode.py', 'src/m4_lines.py', 'scripts/spatial_layout/sp_regression.py', 'scripts/spatial_layout/spat01r/sp2_lines_table.py',
                           'scripts/spatial_layout/sp_lines_table.py', 'scripts/surface_revision/s345_ready/rdy_platform.py', 'scripts/spatial_layout/spat01r/sp2_common.py'],
               'CLIP': ['scripts/surface_revision/s4_real05/s4r5_clip.py', 'scripts/spatial_layout/spat01r/sp2_lines_table.py', 'scripts/spatial_layout/sp_lines_table.py', 'src/m4_lines.py',
                        'scripts/spatial_layout/spat01r/sp2_common.py', 'scripts/spatial_layout/sp_regression.py', 'scripts/surface_revision/s345_ready/rdy_platform.py',
                        'scripts/surface_revision/s345_ready/rdy_decode.py', 'scripts/surface_revision/s4_real02/r2_layouts.py']}
EVAL_DEPS = ['scripts/surface_revision/s4_real02/r2_eval.py', 'scripts/surface_revision/s4_real02/r2_eval_snap.py', 'scripts/surface_revision/s345_ready/rdy_eval.py',
             'scripts/surface_revision/sections_adapter/s1r_eval.py', 'scripts/surface_revision/sections_adapter/s1r_common.py', 'src/surface_eval_r3.py', 'scripts/spacing_layout/sl_lib.py',
             'scripts/spotnoise/sn_lib.py', 'scripts/spatial_layout/spat01r/sp2_lines_table.py', 'scripts/line01/l01_truth.py', 'scripts/surface_revision/surf02/s2_common.py',
             'scripts/surface_revision/s345_ready/rdy_platform.py', 'scripts/surface_revision/s4_real06/s4r6_eval.py']


def code_hashes(paths) -> dict:
    return {p: (K.sha_file(ROOT / p) if (ROOT / p).exists() else 'MISSING') for p in paths}


def own_code() -> dict:
    return {K.rel(p): K.sha_file(p) for p in sorted(HERE.glob('*.py'))}


def decode_key(image_key: str, image_sha: str, raster_key: str, table_hash, method: str, option, params, cfg) -> str:
    return 'R6D-' + K.digest(dict(image=image_key, image_npz=image_sha, raster=raster_key, table=table_hash, method=method, option=option, params=K.clean(params), cfg=K.clean(cfg),
                                  code=code_hashes(DECODE_DEPS[method])), 24)


def eval_key(dkey: str, dec_sha: str, eval_deps: dict, domains: dict, mesh_hash: str, ident_rule: str, ref_image=None) -> str:
    return 'R6E-' + K.digest(dict(decode=dkey, decode_npz=dec_sha, code=eval_deps, domains=domains, mesh=mesh_hash, identity=ident_rule, d=K.D_SUPPORT, tau=K.TAU, gross=GROSS_MM, ref=ref_image), 24)
