# Define Chapter 4 candidates, imaging environments, surfaces and noise levels.

from __future__ import annotations
import os, sys, json, itertools
from pathlib import Path
for _k in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[_k] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HOME = HERE.parents[1]
for _p in [HOME / 'scripts/mp06d_blur', HOME / 'scripts/mp06f_lighting']:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import numpy as np
import pandas as pd
import core as C
OUT = HOME / 'outputs/ch4_bridge01'
C.OUT = OUT
import validation as V
V.install()
import backend as B
B.C.OUT = OUT

PROJECT = C.PROJECT
BASE = PROJECT / 'outputs/surface_revision'
Q = 11695.0  # Sum of normalized projector pixel intensities; peak is capped separately.
SURFS = ['B_curv', 'M10_curv', 'M24_curv']
SIGMAS = [0, 20, 40, 60, 80, 100]  # Registered noise standard deviations [8-bit gray levels] for this experiment.
OLD_DOTS = ['D_r1_p8x' + x for x in ['2.5', '3', '4', '6', '8']] + ['D_r1.5_p8x4', 'D_r1.5_p8x6']
OLD_LINES = ['L_w' + x + '_s8' for x in ['0.5', '0.75', '1', '1.25', '1.5']]
NEW = {'N1': 'D_r0.75_p8x4', 'N2': 'D_r0.75_p8x6', 'N3': 'D_r1_T8x4', 'N4': 'L_w1_s10', 'N5': 'L_w0.5_s10'}
NEW_CFGS = list(NEW.values())
DOTS = OLD_DOTS + [NEW['N1'], NEW['N2'], NEW['N3']]
LINES = OLD_LINES + [NEW['N4'], NEW['N5']]
ALL = DOTS + LINES
BLUR_ENVS = ['NOMINAL', 'BLUR1_5', 'BLUR2', 'BLUR2_5', 'BLUR3']
LIGHT_ENVS = ['NOMINAL', 'UNIFORM', 'G05', 'GRAD_X', 'GRAD_Y', 'GRAD_Y_REV']
ENVS = BLUR_ENVS + LIGHT_ENVS[1:]
PSF = dict(NOMINAL=1., BLUR1_5=1.5, BLUR2=2., BLUR2_5=2.5, BLUR3=3., UNIFORM=1., G05=1., GRAD_X=1., GRAD_Y=1., GRAD_Y_REV=1.)  # Environment-specific Gaussian PSF standard deviations [px].
# Normalized illumination gains; gradients vary with world position in millimetres.
FIELDS = dict(NOMINAL={'kind': 'const', 'value': 1.0}, UNIFORM={'kind': 'const', 'value': 0.75}, G05={'kind': 'const', 'value': 0.5},
              GRAD_X=dict(amp=.25, kind='gradient_x', mean=.75, x_half_mm=46.08, y_half_mm=46.08),
              GRAD_Y=dict(amp=.25, kind='gradient_y', mean=.75, x_half_mm=46.08, y_half_mm=46.08),
              GRAD_Y_REV=dict(amp=.25, kind='gradient_y_reversed', mean=.75, x_half_mm=46.08, y_half_mm=46.08))
for _e in BLUR_ENVS[1:]:
    FIELDS[_e] = {'kind': 'const', 'value': 1.0}
SEED_BASE = dict(point=9_870_000, line=9_880_000)
CAPS = dict(main=6600, subray=36, holdout=288)
OLD = dict(blur_manifest=HOME / 'outputs/mp06c_zero_close_2026-10-21/blur_manifest_7920.csv',
           light_manifest=HOME / 'outputs/mp06f_lighting/manifest.csv',
           blur_results=HOME / 'outputs/mp06d_blur/analysis/results.csv',
           blur_sens=HOME / 'outputs/mp06d_blur/analysis/sensitivity.csv.gz',
           light_results=HOME / 'outputs/mp06f_lighting/results.csv',
           light_sens=HOME / 'outputs/mp06f_lighting/sensitivity.csv.gz',
           blur_store=HOME / 'outputs/mp06d_blur/store.sqlite', light_store=HOME / 'outputs/mp06f_lighting/store.sqlite',
           holdout_protocol=HOME / 'outputs/mp06e_holdout/protocol.json', holdout_manifest=HOME / 'outputs/mp06e_holdout/manifest.csv',
           light_freeze=HOME / 'outputs/mp06f_lighting/freeze.json')
Stop = C.Stop
require = C.require


def fam_of(cfg):
    return 'point' if cfg.startswith('D_') else 'line'


def method_of(cfg):
    return 'DF' if fam_of(cfg) == 'point' else 'P4'


def seed_of(fam, surface, block, rep, base=None):
    return int((base or SEED_BASE)[fam] + 100 * SURFS.index(surface) + 10 * int(block) + int(rep))


def case_key(cfg, sf, b, s, rep, env):
    return f'{cfg}|{sf}|b{b}|s{int(s)}|r{rep}|{env}'


def label(cfg, unit=True):

    u = ' mm' if unit else ''
    if cfg.startswith('D_r'):
        r, p = cfg[3:].split('_', 1)
        lat = ('versetzt ' + p[1:] if p.startswith('T') else p[1:]).replace('x', '×')
        return f'r = {r.replace(".", ",")}{u}, {lat}'
    w, s = cfg[3:].split('_s')
    return f'w = {w.replace(".", ",")} / s = {s.replace(".", ",")}{u}'


def rowdict(r):
    return C.rowdict(r)


def save_json(p, x):
    return C.save(p, x)


def load_json(p):
    return C.load(p)


def expected_keys(cfgs, envs=ENVS):
    return {case_key(cfg, sf, b, s, rep, e) for cfg, e, sf, b, s in itertools.product(cfgs, envs, SURFS, range(4), SIGMAS) for rep in ([0] if s == 0 else [1, 2])}
