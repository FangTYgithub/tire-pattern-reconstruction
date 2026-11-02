# Register supplementary inputs and their source dependencies.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'scripts' / 'surface_revision'
TR1, TR2 = SR / 'thesis_reinforce01', SR / 'thesis_reinforce02'
for _p in (str(TR2), str(TR1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import tr2_common as K2

T1 = K2.T1
OUT = ROOT / 'outputs' / 'surface_revision' / 'thesis_reinforce03'
OUT1, OUT2 = K2.OUT1, K2.OUT
RUN = OUT / 'run.json'
FIGS = OUT / 'figures'
DRAFTS = K2.DRAFTS
TASK = DRAFTS / 'THESIS-REINFORCE-03_protocol.md'

Stop = K2.Stop
clean, rel, load_json, load_npz, read_csv, sha, now = K2.clean, K2.rel, K2.load_json, K2.load_npz, K2.read_csv, K2.sha, K2.now
CALLS = dict(camera_render=0, decode=0, domain_evaluation=0, pattern_raster=0)


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() != q and OUT.resolve() not in q.parents:
        raise Stop('guard', f'write outside {rel(OUT)} refused: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def save_json(p, obj) -> Path:
    q = guard(p); t = q.with_name(q.name + '.tmp'); t.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1, default=str), encoding='utf-8'); t.replace(q)
    return q


def save_csv(df, p) -> Path:
    q = guard(p); t = q.with_name(q.name + '.tmp'); df.to_csv(t, index=False, encoding='utf-8-sig'); t.replace(q)
    return q


def run_load() -> dict:
    return load_json(RUN) if RUN.exists() else dict(task='THESIS-REINFORCE-03', task_book=str(TASK),   created=now())


def run_put(section: str, obj) -> dict:
    r = run_load(); r[section] = clean(obj); r['updated'] = now(); save_json(RUN, r)
    return r


def code_identity() -> dict:
    return {p.name: sha(p) for p in sorted(HERE.glob('tr3_*.py'))}


def require_registered(section: str) -> dict:
    reg = (run_load().get('registration') or {}).get('history') or []
    if not reg:
        raise Stop('refused', f'{section}: no code registration (run `tr3_main.py register` first)')
    cur = code_identity(); last = reg[-1]['code']
    diff = sorted(f for f in set(cur) | set(last) if cur.get(f) != last.get(f) and f != 'tr3_fig.py')
    if diff:
        raise Stop('refused', f'{section}: code changed after the last registration: {diff}')
    return dict(registration_t=reg[-1]['t'], code=cur)


def inputs_used() -> dict:
    R = ROOT / 'outputs' / 'surface_revision'
    files = {
        'tr01_identity_geometry': OUT1 / 'identity_geometry.csv', 'tr01_cases_pairs': OUT1 / 'cases_pairs.csv', 'tr01_layers_conditions': OUT1 / 'layers_conditions.csv',
        'tr01_run': OUT1 / 'run.json', 'tr02_corrections': OUT2 / 'corrections.csv', 'tr02_plan': OUT2 / 'plan.csv', 'tr02_run': OUT2 / 'run.json',
        'tr02_report': OUT2 / 'report_zh.md', 'tr02_K1': OUT2 / 'figures' / 'K1_Zustaende_Kandidatenklasse.png', 'tr02_K2': OUT2 / 'figures' / 'K2_Regulaer_versetzt_gepaart.png',
        's12_views': R / 's12_resource01' / 'results' / 'views.csv', 's3_manifest': R / 's3_code04' / 'formal' / 'manifest.csv',
        's3_expected_keys': R / 's3_code07' / 'plan' / 'expected_keys_formal.csv', 'c4_dots_cases': R / 's3_code04' / 'results' / 'C1_dots_cases.csv',
        'c4_frozen': R / 's3_code04' / 'formal' / 'frozen.json', 's1_dependencies': R / 's1_real01' / 'config' / 's1_dependencies.json',
        'surf02_release': R / 'surf02' / 'release' / 'release_surf02.json', 'surf02_dependencies': R / 'surf02' / 'release' / 'dependencies_surf02_v1_1.json',
        'surf02_checks': R / 'surf02' / 'checks' / 'surf02_checks.json', 'usage_s5_03': ROOT / 'outputs/size_spacing/s5_03/usage/used_phases_seeds.csv',
        'usage_s5_04': ROOT / 'outputs/size_spacing/s5_04/usage/used_phases_seeds_supplement.csv', 'usage_s5_04_scope': ROOT / 'outputs/size_spacing/s5_04/usage/unseen_scope_check.json',
        'candidates_s1s5': R / 'sections_adapter' / 'candidates_s1s5.csv'}
    for p in sorted(R.glob('*/manifest*.csv')) + sorted(R.glob('*/*/manifest*.csv')) + sorted(R.glob('*/*/*/manifest*.csv')):
        files[f'manifest:{rel(p)}'] = p
    for p in sorted((OUT1 / 'shards').glob('*.npz')):
        files[f'tr01_shard:{p.name}'] = p
    return files


def recorded_hashes() -> dict:

    rec = {}
    f1 = load_json(OUT1 / 'run.json').get('final') or {}
    for n, h in (f1.get('csv') or {}).items():
        rec[rel(OUT1 / n)] = h
    for n, h in (f1.get('shards') or {}).items():
        rec[rel(OUT1 / 'shards' / n)] = h
    f2 = load_json(OUT2 / 'run.json').get('final') or {}
    for n, h in (f2.get('outputs') or {}).items():
        rec[rel(OUT2 / n)] = h
    return rec


def bind_inputs() -> dict:
    files = inputs_used(); rec = recorded_hashes(); out = {}; diff = []
    for k, p in files.items():
        h = sha(p) if p.exists() else None
        r = rec.get(rel(p))
        out[k] = dict(path=rel(p), sha256=h, recorded=r, status=('missing' if h is None else ('matches_recorded' if r == h else ('differs_from_recorded' if r else 'no_prior_record'))))
        if out[k]['status'] in ('missing', 'differs_from_recorded'):
            diff.append(dict(k=k, **out[k]))
    return dict(t=now(), n=len(out), differs_or_missing=diff, files=out)
