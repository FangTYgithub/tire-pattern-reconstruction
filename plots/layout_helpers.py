# Paths and helpers for the supplementary layout figures.
from pathlib import Path
import json
import hashlib
from plot_io import configure

ROOT, OUT = configure('layout_figures')
FIG = OUT
BUILD = OUT
RESEARCH = ROOT/'research/project_Part1_v1.7'
SR = RESEARCH/'outputs/surface_revision'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def js(p):
    return json.loads(Path(p).read_text('utf-8'))

def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
