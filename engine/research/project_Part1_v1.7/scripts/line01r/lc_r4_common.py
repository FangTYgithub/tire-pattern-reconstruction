# Load width-code settings and track previously used phases and rasters.

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'line01')); sys.path.insert(0, str(HERE.parents[1] / 'src'))
import l01rb_common as B

ROOT = B.ROOT; R2 = ROOT / 'outputs' / 'line01r' / 'night01r2'; R2CASES = R2 / 'lc' / 'cases'; R3 = ROOT / 'outputs' / 'line01r' / 'night01r3'
R4 = ROOT / 'outputs' / 'line01r' / 'night01r4'; CFG = R4 / 'config'; CASES = R4 / 'cases'; REH = R4 / 'rehearsal'; AN = R4 / 'analysis'; FIG = R4 / 'figures'; CHK = R4 / 'checks'; TESTS = R4 / 'tests'
REASON = {0: '', 1: 'ok', 2: 'window_truncated', 3: 'no_contrast', 4: 'peak_not_central', 5: 'half_level_not_reached', 6: 'neighbour_contamination', 7: 'other'}
REASON_CODE = {v: k for k, v in REASON.items()}
ARMS = ('ignore', 'use', 'R', 'A')
SEED_BASE = 10700000
PHASE_CANDIDATES_X32 = (-16, -8, 8, 15, 16, -2)


def config() -> dict:
    return B.load_json(CFG / 'r4_config.json')


def snapshot() -> dict:
    return B.load_json(R2 / 'config' / 'snapshot_line_encoding_dev_v1.json')


def runtime_params() -> dict:

    snap = snapshot(); r3 = B.load_json(R3 / 'config' / 'r3_config.json')
    pp = dict(snap['params']['LC']); pp['steger_noise_coef'] = {float(k): float(v) for k, v in pp['steger_noise_coef'].items()}
    p4 = dict(snap['params']['P4_ref']); p4['steger_noise_coef'] = {float(k): float(v) for k, v in p4['steger_noise_coef'].items()}
    ro = dict(snap['readout']); ro['threshold_px'] = float(r3['inputs']['old_threshold_px'])
    rel = dict(r3['reliability_rule']); rel.update(margin_px=0.5, min_rows=5)
    return dict(LC=pp, P4_ref=p4, readout=ro, reliability=rel, old_threshold_px=float(r3['inputs']['old_threshold_px']))


def seen_phases() -> set:
    import l01rc1_matrix as M1
    return set(M1.SEEN_PHASES) | set(M1.FORMAL_PHASES) | set(M1.C2_PHASES) | {-13 / 32, -5 / 32, 3 / 32, 11 / 32, -1 / 32, 6 / 32} | set(B.PHASES)


def seen_rasters_and_keys():
    import pandas as pd
    r, k = set(), set()
    for p in (ROOT / 'outputs/line01r/c3/night/formal_manifest_4928_c3.csv', ROOT / 'outputs/line01r/c2/manifest_c2.csv', ROOT / 'outputs/line01r/c3/manifest_c3.csv', R2 / 'lc' / 'manifest_lc.csv'):
        d = pd.read_csv(p, low_memory=False); r |= set(d.raster_sha256); k |= set(d.forward_key)
    bm = pd.DataFrame(B.manifest()); r |= set(bm.raster_sha256); k |= set(bm.forward_key)
    return r, k


def load_case(cases_dir: Path, cid: str):
    rec = B.load_json(cases_dir / f'{cid}.json')
    with np.load(cases_dir / f'{cid}.npz') as z:
        arr = {k: np.asarray(z[k]) for k in z.files}
    return rec, arr


def status(**kw) -> dict:
    p = R4 / 'status.json'; st = B.load_json(p) if p.exists() else {}
    st.update(kw); st['updated_local'] = B.now_local(); B.save_json(p, st); return st


def ensure_dirs():
    for d in (R4, CFG, CASES, REH, AN, FIG, CHK, TESTS):
        d.mkdir(parents=True, exist_ok=True)
