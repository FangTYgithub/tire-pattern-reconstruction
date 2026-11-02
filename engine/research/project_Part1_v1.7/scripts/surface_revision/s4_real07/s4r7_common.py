# Provide storage, result identities and execution limits for held-out layouts.

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
for _p in (ROOT / 'scripts' / 'surface_revision' / 's4_real02', ROOT / 'scripts' / 'surface_revision' / 's4_real05', ROOT / 'scripts' / 'surface_revision' / 's4_real06', HERE,
           ROOT / 'scripts' / 'spatial_layout' / 'spat03', ROOT / 'scripts' / 'spatial_layout' / 'spat01r', ROOT / 'scripts' / 'spatial_layout'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import r2_common as K

SR = ROOT / 'outputs' / 'surface_revision'
OUT = SR / 's4_real07'
RUN = OUT / 'run.json'
FIGS, RUNDIR, IMG7, DEC7, EVL7, ISO, ENTRY = OUT / 'figures', OUT / 'run', OUT / 'images', OUT / 'decodes', OUT / 'evals', OUT / '_isolation', OUT / '_entry'
R2, R5, R6 = SR / 's4_real02', SR / 's4_real05', SR / 's4_real06'
DRAFTS = ROOT.parents[1] / 'protocols'
TASK = DRAFTS / 'S4-REAL-07_protocol.md'
REVIEW = DRAFTS / 'S4-REAL-06_review.md'
Q = 11695.0
GROSS_MM = 1.0
CAP_RENDER, CAP_DECODE = 660, 1020
RENDER_PURPOSE = dict(science=648, retry=12)
DECODE_PURPOSE = dict(science=996, regression=12, technical=12)
NOISE_SEED_BASE_R7 = 3_200_000_000
SURF = ('B_curv', 'M10_curv', 'M24_curv')

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
    return load_json(RUN) if RUN.exists() else dict(task='S4-REAL-07', task_book=K.rel(TASK),  acceptance='reference (independent)', created=K.now_local())


def run_put(section: str, obj, freeze: bool = False) -> dict:
    r = run_load(); fz = r.get('frozen', {})
    if section in fz:
        raise Stop('freeze', f'run.json section {section} is frozen; never rewritten')
    if freeze:
        r.setdefault('frozen', {})[section] = dict(digest=K.digest(obj, 32), t_local=K.now_local())
    r[section] = obj; r['updated'] = K.now_local(); save_json(RUN, r)
    return r


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
        if int(b['cap']) != self.cap or purpose not in self.purposes:
            raise Stop('budget', f'{self.kind}: cap / purpose mismatch ({purpose})')
        if b['used'] >= b['cap'] or b['by_purpose'].get(purpose, 0) >= self.purposes[purpose]:
            raise Stop('budget', f"{self.kind} budget exhausted for {purpose} ({b['by_purpose'].get(purpose, 0)} / {self.purposes[purpose]}, total {b['used']} / {b['cap']})")
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


FORWARD_DEPS = ['scripts/surface_revision/pix_bound01/pb_m3.py', 'src/m3_virtual_camera.py', 'scripts/surface_revision/sections_adapter/adp_surf02.py', 'scripts/surface_revision/surf02/s2_common.py',
                'scripts/line01/l01_common.py', 'scripts/surface_revision/s4_real02/r2_layouts.py', 'scripts/spatial_layout/sp_dots_columns.py', 'scripts/spatial_layout/spat01r/sp2_lines_table.py',
                'scripts/spatial_layout/spat01r/sp2_common.py', 'scripts/surface_revision/s345_ready/rdy_inputs.py', 'scripts/spatial_layout/sp_lines_table.py', 'src/m2_pattern.py',
                'scripts/surface_revision/s4_real07/s4r7_phase.py']
GEN_DEPS = ['scripts/spatial_layout/spat03/s3_generate.py', 'scripts/spatial_layout/spat03/s3_common.py', 'scripts/spatial_layout/spat01r/sp2_generator.py', 'scripts/spatial_layout/sp_geometry.py',
            'scripts/surface_revision/s4_real07/s4r7_gen.py']
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
             'scripts/surface_revision/s345_ready/rdy_platform.py', 'scripts/surface_revision/s4_real07/s4r7_eval.py']
RUN_DEPS = ['scripts/surface_revision/s4_real07/s4r7_common.py', 'scripts/surface_revision/s4_real07/s4r7_run.py', 'scripts/surface_revision/s4_real07/s4r7_bind.py',
            'scripts/surface_revision/s4_real07/s4r7_analysis.py', 'scripts/surface_revision/s4_real07/s4r7_main.py']
DATA_DEPS = ['outputs/surface_revision/pix_bound01/domains/omega1.npz', 'outputs/surface_revision/pix_bound01/domains/omega1.json']
DISPLAY_DEPS = ['scripts/surface_revision/s4_real07/s4r7_figs.py']
SCIENCE_FILES = ['outputs/surface_revision/s4_real07/manifest.csv', 'outputs/surface_revision/s4_real07/_parts/results_plan.csv', 'outputs/surface_revision/s4_real07/generation.json',
                 'outputs/surface_revision/s4_real07/_parts/views.csv', 'outputs/surface_revision/s4_real07/_parts/rasters.csv']


def code_hashes(paths, hasher=None) -> dict:
    h = hasher or K.sha_file
    return {p: (h(ROOT / p) if (ROOT / p).exists() else 'MISSING') for p in paths}


def own_code() -> dict:
    return {K.rel(p): K.sha_file(p) for p in sorted(HERE.glob('*.py'))}


def decode_key(image_key, image_sha, raster_key, table_hash, method, option, params, cfg, hasher=None) -> str:
    return 'R7D-' + K.digest(dict(image=image_key, image_npz=image_sha, raster=raster_key, table=table_hash, method=method, option=option, params=K.clean(params), cfg=K.clean(cfg),
                                  code=code_hashes(DECODE_DEPS[method], hasher)), 24)


def eval_key(dkey, dec_sha, eval_deps: dict, domains: dict, mesh_hash: str, ident_rule: str, ref_image=None) -> str:
    return 'R7E-' + K.digest(dict(decode=dkey, decode_npz=dec_sha, code=eval_deps, domains=domains, mesh=mesh_hash, identity=ident_rule, d=K.D_SUPPORT, tau=K.TAU, gross=GROSS_MM, ref=ref_image), 24)


def science_snapshot(hasher=None, data_digests: dict | None = None) -> dict:

    return dict(science_files=code_hashes(SCIENCE_FILES, hasher), forward=code_hashes(FORWARD_DEPS, hasher), generation=code_hashes(GEN_DEPS, hasher),
                decode={m: code_hashes(v, hasher) for m, v in DECODE_DEPS.items()}, evaluation=code_hashes(EVAL_DEPS, hasher), run=code_hashes(RUN_DEPS, hasher),
                data=code_hashes(DATA_DEPS, hasher), data_digests=data_digests or {})


def freeze_check(stage: str, hasher=None, data_digests: dict | None = None, record: bool = True) -> dict:

    r = run_load()
    if 'config' not in r.get('frozen', {}):
        raise Stop('freeze', 'science configuration not frozen')
    if K.digest(r['config'], 32) != r['frozen']['config']['digest']:
        raise Stop('freeze', 'frozen config section changed')
    fz = r['config']['snapshot']; now = science_snapshot(hasher, data_digests if data_digests is not None else fz['data_digests'])
    diff = []
    for grp in fz:
        a, b = fz[grp], now.get(grp)
        if isinstance(a, dict) and grp == 'decode':
            for m in a:
                diff += [f'decode/{m}/{k}' for k in a[m] if a[m][k] != (b or {}).get(m, {}).get(k)]
        elif isinstance(a, dict):
            diff += [f'{grp}/{k}' for k in a if a[k] != (b or {}).get(k)]
    out = dict(stage=stage, ok=not diff, differences=diff, t_local=K.now_local())
    if record:
        rr = run_load(); rr.setdefault('freeze_checks', []).append(out); rr['updated'] = K.now_local(); save_json(RUN, rr)
    if diff:
        raise Stop('freeze', f'{stage}: science inputs changed after the freeze: {diff[:6]}')
    return out
