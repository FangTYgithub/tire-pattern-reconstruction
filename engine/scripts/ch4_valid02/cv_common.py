# Define validation phases, seeds and candidate identities for Chapter 4.

from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
HOME = HERE.parents[1]
sys.path.insert(0, str(HOME / 'scripts/ch4_bridge01'))
import cb_common as G
C, B = G.C, G.B
OUT = HOME / 'outputs/ch4_valid02'
C.OUT = OUT
B.C.OUT = OUT
BRIDGE = HOME / 'outputs/ch4_bridge01'
SEED_BASE = dict(point=26910000, line=26920000)
N_PHASES = 4  # Number of validation phase blocks per configuration.
CAP = 22440
RETRY_CAP = 60
Stop, require = C.Stop, C.require
ALL, DOTS, LINES, ENVS, SURFS, SIGMAS = G.ALL, G.DOTS, G.LINES, G.ENVS, G.SURFS, G.SIGMAS
fam_of, method_of, seed_of, label = G.fam_of, G.method_of, G.seed_of, G.label


def case_key(cfg, sf, b, s, rep, env):
    return G.case_key(cfg, sf, f'V{b}', s, rep, env)
