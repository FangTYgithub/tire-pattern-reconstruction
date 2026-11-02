# Define analysis settings and output paths for PSF and illumination diagnostics.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'scripts' / 'surface_revision'
MP5_DIR = SR / 'model_prep05'
for _q in (MP5_DIR,):
    if str(_q) not in sys.path:
        sys.path.insert(0, str(_q))
import mp5_common as K5

OUT = ROOT / 'outputs' / 'surface_revision' / 'model_prep06a'
SANDBOX = OUT / '_sandbox'
SRC5 = ROOT / 'outputs' / 'surface_revision' / 'model_prep05'
SRC5C = ROOT / 'outputs' / 'surface_revision' / 'model_prep05c'
S5A = ROOT / 'outputs' / 'surface_revision' / 's5_align05'
TR4 = ROOT / 'outputs' / 'surface_revision' / 'thesis_reinforce04'
P4_CFG = ROOT / 'outputs' / 'line01r' / 'c3' / 'config' / 'paths_c3_validation.json'
DRAFTS = ROOT.parents[1] / 'protocols'
TASK_BOOK = DRAFTS / 'MODEL-PREP-06A_protocol.md'
REVIEWS = (DRAFTS / 'MODEL-PREP-05_review.md', DRAFTS / 'MODEL-PREP-05C_review.md')
TASK_ID = 'MODEL-PREP-06A'

INPUT_FILES = ('outputs/surface_revision/model_prep05/formal_manifest.csv', 'outputs/surface_revision/model_prep05/results.csv',
               'outputs/surface_revision/model_prep05/effects.csv', 'outputs/surface_revision/model_prep05/data/index.json',
               'outputs/surface_revision/model_prep05/run.json', 'outputs/surface_revision/model_prep05c/report_zh.md',
               'outputs/surface_revision/model_prep05c/candidate_adjustments.csv', 'outputs/line01r/c3/config/paths_c3_validation.json',
               'outputs/surface_revision/s5_align05/condition_summary.csv', 'outputs/surface_revision/s5_align05/case_results.csv',
               'outputs/surface_revision/s5_align05/run.json', 'outputs/surface_revision/thesis_reinforce04/run.json')
CODE_FILES = ('src/m4_lines.py', 'src/m3_virtual_camera.py', 'src/m4_reconstruction.py', 'src/surface_eval_r3.py',
              'scripts/surface_revision/model_prep03/mp3_forward.py', 'scripts/surface_revision/model_prep03/mp3_chain.py',
              'scripts/surface_revision/model_prep03/mp3_common.py', 'scripts/surface_revision/model_prep05/mp5_inputs.py',
              'scripts/surface_revision/model_prep05/mp5_chain.py', 'scripts/surface_revision/model_prep05/mp5_common.py',
              'scripts/surface_revision/model_prep05/mp5_analysis.py', 'scripts/surface_revision/s5_align04/s5t_platform.py',
              'scripts/surface_revision/s345_ready/rdy_decode.py', 'scripts/surface_revision/s345_ready/rdy_eval.py',
              'scripts/surface_revision/s345_ready/rdy_platform.py', 'scripts/surface_revision/s345_ready/rdy_figs.py',
              'scripts/surface_revision/sections_adapter/adp_surf02.py', 'scripts/surface_revision/sections_adapter/s1r_eval.py',
              'scripts/surface_revision/sections_adapter/s1r_common.py', 'scripts/surface_revision/surf02/s2_common.py',
              'scripts/surface_revision/thesis_reinforce05/tr5_chain.py', 'scripts/surface_revision/thesis_reinforce04/tr4_gate.py',
              'scripts/surface_revision/s3_code01/s3c_dev.py')


REPRESENTATIVES = ('D_r1_p8x6', 'D_r1.5_p8x4', 'L_w0.5_s8', 'L_w1.5_s8')
DOTS5 = ('D_r1_p8x4', 'D_r1_p8x6', 'D_r1.5_p8x4', 'D_r1.5_p8x6')
LINES5 = ('L_w0.5_s6.3', 'L_w0.5_s8', 'L_w1.5_s6.3', 'L_w1.5_s8')
LINES8 = ('L_w0.5_s8', 'L_w1.5_s8')
SURFS = ('B_curv', 'M10_curv', 'M24_curv')
BLOCKS, REPS = (0, 1, 2, 3), (1, 2)
TAUS = (0.10, 0.15, 0.20, 0.30)
DS = (5.0, 6.0)
TAU0, D0_SUPPORT = 0.15, 6.0
REPRO_TOL = 1e-10
PSF_ENVS = (('NOMINAL', 1.0), ('BLUR1_5', 1.5), ('BLUR2', 2.0), ('BLUR3', 3.0))
SCALE_TIE_REL = 1e-12
SCALE_MATCH_TOL = 1e-10
CAPS = dict(decode_A=1184, render_compat=6, render_3x3=96, render_B=102, decode_B=96, decode_total=1280)
SELECTION = dict(
    status='research screening convention registered by the task book (not a device / supervisor acceptance criterion)',
    main=dict(tau_mm=0.15, d_mm=6.0),
    feasible=dict(ctau_global_min=0.50, ctau_local_min=0.50, per_image_global_pt_rmse_max_mm=0.30, per_image_w_max=0, per_image_u_share_max=0.05,
                  per_image_c_plus_w_min=1, no_output_or_undefined='fails'),
    sensitivity=dict(ctau_floor_alt=0.70, taus=[0.10, 0.20, 0.30], other_requirements='unchanged'),
    aggregation='mean of the noise repetitions per phase block -> equal-weight mean over phase blocks; sigma 0 separate (one image per phase, never mixed)',
    tie_band_pp=2.0,
    three_surface_rule='candidate must pass on all three surfaces; score = min over surfaces of the per-surface global C_tau mean; max wins',
    tie_breaks=['keep fixed nominal reference (dots D_r1_p8x6, lines L_w0.5_s8) inside the tie band', 'max of worst-surface local C_tau',
                'min of per-image max global point RMSE', 'lexicographic configuration id'],
    no_candidate='report "no qualified candidate under this requirement"; best failing reference may be listed, never called a recommendation',
    families='dots and lines separately; only DF / P4; A3 diagnostic arms never enter; only jointly measured conditions, no interpolation',
    blocks_not_ci='four-block range is not a confidence interval; old and new phases are never pooled')
HOLDOUT = dict(
    conditions=[dict(name='H1', psf_px=1.0, sigma=40.0, gain=1.0), dict(name='H2', psf_px=2.0, sigma=20.0, gain=1.0),
                dict(name='H3', psf_px=3.0, sigma=40.0, gain=1.0), dict(name='H4', psf_px=1.0, sigma=40.0, gain=0.5)],
    slots='per condition and family at most 2 configurations: the three-surface preferred configuration (§6) + the fixed nominal reference '
          '(dots D_r1_p8x6, lines L_w0.5_s8); identical -> one; none qualified -> reference only, flagged',
    images='<= 4 conditions x 2 families x 2 configurations x 3 surfaces x 4 new phases x 2 new noise realisations = 384',
    claim='replication in the tested conditions with new phases of the same position domain and new noise; not out-of-domain, not a real experiment, not a continuous optimum',
    verdict='same screening level; difference to the reference < -2 pp -> advantage not replicated; no replacement candidate; no regret bound to the full grid')
A3_RULE = ('PSF = 1: s* = s0 (negative control). PSF > 1: among the frozen five P4_C3_cal nodes >= s0 choose the node maximising '
           'unit-peak analytic ridge response R(w_px, node, psf_true) / noise coefficient c(node); ties (relative 1e-12) -> smaller node. '
           'Inputs: registered line width and PSF only (never C_tau, truth error or wrong ids). Diagnostic rule on finite nodes, not an optimal detector.')
ARMS = (('N', 's0', 'nominal'), ('T', 's0', 'true'), ('S', 'star', 'nominal'), ('ST', 'star', 'true'))
SUBRAY = dict(configs=LINES8, envs=PSF_ENVS, sigmas=(20.0, 40.0), surfaces=SURFS, blocks=(0, 2), rep=1, rays_per_pixel=3,
              compat=dict(configs=LINES8, sigmas=(0.0, 20.0, 40.0), surface='M10_curv', block=0, env='NOMINAL'),
              warning='|Delta_w(1x1)| >= 2 pp and 3x3 inside +-2 pp or sign change; or |change of Delta_w| >= 5 pp; or a new confirmed wrong id')


class Stop(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(f'{code}: {msg}'); self.code = code; self.msg = msg


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def undef(x) -> bool:
    return x is None or (isinstance(x, (float, np.floating)) and math.isnan(float(x)))


def num(x):
    return None if undef(x) else float(x)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not math.isfinite(v) else v
    return o


def digest(o) -> str:
    return hashlib.sha256(json.dumps(clean(o), sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


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


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save_csv(df, p) -> Path:
    return atomic(p, lambda t: df.to_csv(t, index=False, encoding='utf-8-sig'))


def save_npz(p, meta: dict, arrays: dict) -> str:
    def w(t):
        with open(t, 'wb') as fh:
            np.savez_compressed(fh, _meta=np.array(json.dumps(clean(meta), sort_keys=True, ensure_ascii=False)), **arrays)
    return sha_file(atomic(p, w))


PROTOCOL = OUT / 'protocol.json'


def protocol() -> dict:
    return load_json(PROTOCOL) if PROTOCOL.exists() else dict(task=TASK_ID, task_book=str(TASK_BOOK),   created=now())


def protocol_put(section: str, obj) -> dict:
    p = protocol(); p[section] = clean(obj); p['updated'] = now(); save_json(PROTOCOL, p)
    return p


def rules_registered() -> dict:
    p = protocol()
    if 'rules' not in p:
        raise Stop('preflight', 'rules are not registered in protocol.json (stage bind first)')
    return p['rules']


LEDGER = OUT / 'ledger.jsonl'


class Ledger:
    def __init__(self):
        self.n = {}
        if LEDGER.exists():
            for line in LEDGER.read_text(encoding='utf-8').splitlines():
                r = json.loads(line)
                if r.get('event') == 'reserve':
                    self.n[r['group']] = self.n.get(r['group'], 0) + 1

    def _write(self, rec: dict) -> None:
        guard(LEDGER)
        with open(LEDGER, 'a', encoding='utf-8') as f:
            f.write(json.dumps(clean(rec), ensure_ascii=False) + '\n')

    def count(self, *groups) -> int:
        return sum(self.n.get(g, 0) for g in groups)

    def reserve(self, group: str, key: str, cap_groups: dict) -> str:
        for cap_name, members in cap_groups.items():
            if self.count(*members) + (1 if group in members else 0) > CAPS[cap_name]:
                raise Stop('budget', f'{group} {key}: cap {cap_name} = {CAPS[cap_name]} reached')
        rid = digest(dict(group=group, key=key, t=time.time_ns()))[:16]
        self._write(dict(event='reserve', rid=rid, group=group, key=key, t=now())); self.n[group] = self.n.get(group, 0) + 1
        return rid

    def end(self, rid: str, status: str, **kw) -> None:
        self._write(dict(event='end', rid=rid, status=status, t=now(), **kw))


CAP_A = dict(decode_A=('regress_P4', 'decode_arm'), decode_total=('regress_P4', 'decode_arm', 'decode_3x3'))
CAP_B_RENDER = dict(render_compat=('render_compat',), render_3x3=('render_3x3',), render_B=('render_compat', 'render_3x3'))
CAP_B_DECODE = dict(decode_B=('decode_3x3',), decode_total=('regress_P4', 'decode_arm', 'decode_3x3'))


_IDX: dict = {}


def pack_rel(case_key: str) -> str:
    return f"outputs/surface_revision/model_prep05/data/{case_key.replace('|', '__')}.npz"


def read_pack(case_key: str, prefixes=None, check_hash: bool = True) -> tuple[dict, dict]:
    if not _IDX:
        _IDX.update(load_json(SRC5 / 'data' / 'index.json'))
    rp = pack_rel(case_key)
    if rp not in _IDX:
        raise Stop('identity', f'{case_key}: pack not in MODEL-PREP-05 data/index.json')
    if check_hash and sha_file(ROOT / rp) != _IDX[rp]['sha256']:
        raise Stop('identity', f'{case_key}: pack sha256 differs from data/index.json')
    with np.load(ROOT / rp, allow_pickle=False) as z:
        names = [n for n in z.files if n != '_meta' and (prefixes is None or any(n.startswith(p) for p in prefixes))]
        A = {n: z[n] for n in names}
        meta = json.loads(str(z['_meta']))
    return meta, A


def sub(A: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in A.items() if k.startswith(prefix)}


def set_sandbox_env() -> None:

    for k, s in (('LINE01RB_OUT_DIR', 'line_out'), ('LINE01RB_CASES_DIR', 'line_cases'), ('LINE01RB_LEDGER_DIR', 'line_ledger')):
        os.environ[k] = str(SANDBOX / s)
    os.environ['LINE01RB_RENDER_CAP'] = '0'
