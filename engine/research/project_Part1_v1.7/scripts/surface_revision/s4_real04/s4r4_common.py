# Provide input registration and guarded storage for layout mechanism analysis.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / 'scripts' / 'surface_revision' / 's4_real02', ROOT / 'scripts' / 'surface_revision' / 's4_real03'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import r2_common as K

SR = ROOT / 'outputs' / 'surface_revision'
OUT = SR / 's4_real04'
RUN = OUT / 'run.json'
FIGS = OUT / 'figures'
R3 = SR / 's4_real03'
S12 = SR / 's12_resource01'
S345 = SR / 's345_ready'
DRAFTS = ROOT.parents[1] / 'protocols'
TASK = DRAFTS / 'S4-REAL-04_protocol.md'
reference_R02 = {'run.json': 'd9020731a1dabc847df284906f6ae528ef9180ed11ba85efec73c18db8c06e38', 'manifest.csv': 'c9689e279f089ba7f42e337a1300394131fee7a3add8115a9392e7d9956980a1',
             'case_metrics.csv': '9c10c7fcc96649b59fd97463e1017f26ec3026385264c9481ab0a924e78e0bf6', 'paired_summary.csv': '667515e83ca18aa7cf142cf855b76bc796947af75861f28eceae49d3ee261618',
             'layouts.json': '20f28d4bc275c615224a9ae89135028b9815d96f3444947ee085a40a58728f16', 'report_zh.md': '00ed41b0ab81923b7ac1c6676ea6e233ce63de867907a645666d43b9c2da364d'}
reference_R03 = {'outputs/surface_revision/s4_real03/report_zh.md': 'a7bd88d05b17ee9c6758f321ae247739882f2c8b63c584adbf85880e7adb4849',
             'outputs/surface_revision/s4_real03/diagnostics.csv': '954c82fc0ceaff4b9a3367a08a5c03ecb301dfd1a052762bc8933bdcb0a2ba5d',
             'outputs/surface_revision/s4_real03/run.json': '1912b95cbb6deb60033b1f3e866abe2c59616df11cfedd61934d96223014b284',
             'scripts/surface_revision/s4_real03/r3_diag.py': '1469dca98abdf32b613e19ff3816690e99068f8826ea56b9c1365cf76c463249',
             'scripts/surface_revision/s4_real03/r3_figs.py': 'f962f5520b32b147f2540c561062485c649ef1cd013998529e404eeb7cc2e25b'}

READ: dict = {}


def reg(p) -> Path:

    p = Path(p); r = K.rel(p)
    if r not in READ and p.exists():
        READ[r] = K.sha_file(p)
    return p


def _guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in q.parents and q != OUT.resolve():
        raise SystemExit(f'write outside {OUT} refused: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def save_json(p, obj):
    q = _guard(p); t = q.with_name(q.name + '.tmp'); t.write_text(json.dumps(K.clean(obj), ensure_ascii=False, indent=1), encoding='utf-8'); os.replace(t, q)


def save_csv(df: pd.DataFrame, p):
    q = _guard(p); t = q.with_name(q.name + '.tmp'); df.to_csv(t, index=False, encoding='utf-8-sig'); os.replace(t, q)


def run_load() -> dict:
    return json.loads(RUN.read_text(encoding='utf-8')) if RUN.exists() else dict(task='S4-REAL-04', task_book=K.rel(TASK),  acceptance='reference (independent)', created=K.now_local())


def run_put(section: str, obj) -> None:
    r = run_load(); r[section] = obj; r['updated'] = K.now_local(); save_json(RUN, r)


def flush_reads(section: str) -> None:
    r = run_load(); have = r.get('inputs_read', {}); have.update({k: v for k, v in READ.items() if k not in have}); r['inputs_read'] = have; r['updated'] = K.now_local(); save_json(RUN, r)


CONFIG = dict(
    scope='new camera imaging 0, full re-decoding 0, new detection 0; stored records, pure geometry, candidate geometry from stored features, limited point-cloud re-evaluation',
    A=dict(anchor_eval='R2E-fbaf24dfc297c37eeabea499', pairing='line M10 H0 sigma 0: S1 (8 blocks) and S3 vs S0 from case_metrics (stored)',
           phase_sensitivity='line M10 sigma 40 S3-S0 glob C_tau: all phases vs H1-H3 (post-hoc stability check, never a replacement)',
           source_grading='all 76 S4-REAL-02 images with accepted clipped-source points: unclipped projector coordinates; direct = inside [-0.5, 511.5)^2 and the rounded pixel lit; '
                          'out_of_frame; inside_unlit (adjacent to the clipped band, PSF / chain compatible)',
           dot_area='384 sigma >= 80 dot images: share of the candidate area below 0.5 x sigma-0 dot area (stored cand_area), exception vs the rest'),
    B1=dict(stats='L6.3 S0 regular, B / M10 / M24 x H0-H3 at sigma 0 (12 images, stored n_candidates / status / segment_id / guard_reason / in_global / in_local); '
                  'noise check: the same 24 images at sigma 20 / 40 (stored fields only)',
            geometric_cases_rule='per surface the sigma-0 S0 phase with the largest and the smallest segment_unknown share (ties: lower phase) -> 6 images; candidate geometry '
                                 '(sp2_lines_table.line_candidates_table on stored features, registered cfg / centrelines) + vote counts per stored segment; truth only for diagnosis',
            height_bins_mm=1.0),
    B2=dict(inventory='sections 1 / 2 (s12_resource01 inputs + inputs_metrics, Gaussian supplement listed separately) and section 5 (S5-main cases): all stored fields; '
                      'line rasters regenerated with the registered generators (no imaging) for clipped_lines; dot rasters for border components',
            trace_rule='at most 8 unique inputs per section, fixed from the stored metadata before any tracing: lines: per layout with a lit clipped line the input with the most '
                       'accepted unconfirmable points (largest first, distinct layouts), one 45 deg input with confirmed wrong identities if present, negative controls: one input with '
                       'a lit clipped line and zero unconfirmable points and one without clipped lines (sigma 0, first by key); dots: the input with the most confirmed wrong identities and '
                       'the one with the most unconfirmable points at sigma <= 40, else at sigma >= 80; Gaussian surface excluded'),
    C1=dict(families=['R8x4', 'L6.3'], j=[0.0, 0.10, 0.15, 0.20, 0.25, 0.30], alpha=[0.70, 0.75, 0.80], beta=[1.10, 1.15, 1.20, 1.25], blocks=list(range(16)), proposals_per_block=1000,
            seed_namespace='s3_common.seed_s3 (registered); blocks 0, 3, 6, 9, 12, 15 coincide with the S4-REAL-02 imaged blocks',
            selection='first candidate (registered order) satisfying the arm mask; S1 = candidate 0; failure = no candidate in 1000',
            rms_matched_control='per block: a candidate of the same pool that violates the S3 mask with |RMS - RMS(S3 selected)| <= 0.01 p0 (availability only, no new arm)',
            grid_reason='j above (1 - alpha) / 2 activates S2 (old D2 scan); beta 1.10 / 1.20 between the old 1.05 / 1.15 / 1.25; j <= 0.30 keeps the coordinate edge rule (old D2); '
                        'alpha 0.75 is new; 1000 proposals replace the old 512 vectors',
            raster_checks='<= 24: per family the first S1 / S2 / S3 / S4 layouts of block 0 at j 0.10 / 0.20 / 0.30 (alpha 0.70, beta 1.15), duplicates removed'),
    C2=dict(groups='per family x surface (B / M10 / M24), sigma 0, H0: (i) display block (dots B021, lines B000) S0 / S1 / S3; (ii) the S1 / S3 layout pair with the smallest '
                   '|RMS/p0 difference| among S3 != S1 (any blocks), qualified only if <= 0.01; S0 shared -> <= 30 clouds',
            d_mm=[3.5, 4.0, 5.0, 6.0], tau_mm=0.15, main='d = 6 mm, tau = 0.15 mm unchanged; other d diagnostic only',
            layers='L0 nominal h; L1 reference sampling on the true surface (lines: stored ref_xyz rows, step 1 px; dots: stored exp_xy on the true surface); '
                   'L2 decoded points with confirmed correct identity; L3 decoded valid cloud (official)'),
)
