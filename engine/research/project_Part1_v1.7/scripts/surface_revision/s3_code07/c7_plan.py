# Prepare the encoding execution sequence and initial entry checks.

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import c7_common as K

MAN = K.O4 / 'formal' / 'manifest.csv'
MID = dict(dots='R6x4', lines='5.6')
FIXED = [dict(group='fixed', family='dots', layout='R6x4', pattern='mid', surface='M24_curv', phase_id='N3', rep=1, rule='DF', arm='use', sigmas=[0.0, 40.0, 100.0]),
         dict(group='fixed', family='lines', layout='5.6', pattern='strong', surface='M24_curv', phase_id='N5', rep=0, rule='R@1.1247', arm='R@1.1247', sigmas=[0.0, 20.0, 60.0])]
SUPP = [dict(group='supplement', family='dots', layout='R6x4', pattern='mid', surface='B_curv', phase_id='N3', rep=1, rule='DF', arm='use', sigmas=[0.0, 40.0]),
        dict(group='supplement', family='dots', layout='R6x4', pattern='mid', surface='M10_curv', phase_id='N3', rep=1, rule='DF', arm='use', sigmas=[0.0, 40.0]),
        dict(group='supplement', family='lines', layout='5.6', pattern='strong', surface='M24_curv', phase_id='N5', rep=0, rule='R@1.1247', arm='R@1.1247', sigmas=[40.0]),
        dict(group='supplement', family='lines', layout='5.6', pattern='strong', surface='B_curv', phase_id='N5', rep=0, rule='R@1.1247', arm='R@1.1247', sigmas=[0.0, 40.0]),
        dict(group='supplement', family='lines', layout='5.6', pattern='strong', surface='M10_curv', phase_id='N5', rep=0, rule='R@1.1247', arm='R@1.1247', sigmas=[0.0, 40.0])]
POSTHOC = ['risk', 'collapse']


def _torus(a, b):
    d = np.abs(a - b); return np.minimum(d, 1.0 - d)


def subset(M: pd.DataFrame) -> pd.DataFrame:
    S = M[(M.sigma == 0) & (M.pattern.isin(['plain', 'strong'])) & (M.phase_id.isin(['N0', 'N7'])) & (M.layout.astype(str) == M.family.map(MID))]
    return S[['image_key', 'family', 'surface', 'layout', 'pattern', 'phase_id', 'code_parity', 'raster_key', 'sigma', 'rep']].sort_values(['family', 'surface', 'pattern', 'phase_id'])


def expected_pairs(E: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    Ev = E[E.layer == 'eval'].merge(M[['image_key', 'pattern']], on='image_key')
    rows = []
    D, L = Ev[Ev.family == 'dots'], Ev[Ev.family == 'lines']
    for rule in ('B0', 'DF', 'DS'):
        rows.append(('E1', rule, int(((D.rule == rule) & (D.arm == 'use')).sum())))
        rows.append(('E2', rule, int(((D.rule == rule) & (D.arm == 'use')).sum())))
    for x, y in (('DF', 'B0'), ('DS', 'B0'), ('DF', 'DS')):
        for arm in ('use', 'ignore', 'single'):
            rows.append(('E3', f'{x}-{y} {arm}', int(((D.rule == x) & (D.arm == arm)).sum())))
    for rule in ('R@1.1247', 'LB@1.0000'):
        rows.append(('E1L', rule, int((L.rule == rule).sum())))
    rows.append(('E2L', 'R-P4', int((L.rule == 'R@1.1247').sum())))
    return pd.DataFrame(rows, columns=['pair_type', 'comparison', 'n_pairs'])


def n_material(M: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    O1, O2 = K.K4.O1, K.K4.O2
    nf = pd.read_csv(O1 / 'results' / 'precheck_new_phases.csv')
    sd = pd.read_csv(O1 / 'results' / 'precheck_seen_dot_phases.csv'); sl = pd.read_csv(O1 / 'results' / 'precheck_seen_line_phases.csv')
    bc = pd.read_csv(O2 / 'results' / 'plan_phases.csv'); x6 = pd.read_csv(K.O6 / 'plan' / 'B_phases.csv')
    sets = dict(s3_code01_seen=(sd[['u', 'v']], sl[['phase']].rename(columns={'phase': 'phase_frac'})),
                s3_code02_BC=(bc[bc.family == 'dots'][['u', 'v']], bc[bc.family == 'lines'][['phase_frac']]),
                s3_code06_X=(x6[x6.family == 'dots'][['u', 'v']], x6[x6.family == 'lines'][['phase_frac']]))
    ph = M.drop_duplicates(['family', 'phase_id'])[['family', 'phase_id', 'u', 'v', 'phase_frac']]
    rows = []
    for r in ph.itertuples():
        d = dict(family=r.family, phase_id=r.phase_id, u=r.u, v=r.v, phase_frac=r.phase_frac)
        for nm, (dots, lines) in sets.items():
            if r.family == 'dots':
                dist = np.hypot(_torus(dots.u.to_numpy(float), float(r.u)), _torus(dots.v.to_numpy(float), float(r.v)))
            else:
                dist = _torus(lines.phase_frac.to_numpy(float) + 0.5, float(r.phase_frac) + 0.5)
            d[f'min_dist_frac_{nm}'] = float(dist.min()) if len(dist) else None
        rows.append(d)
    N = pd.DataFrame(rows)

    used_r = set(pd.read_csv(K.ROOT / 'outputs/surface_revision/s12_resource01/inputs.csv', low_memory=False, usecols=['raster_key']).raster_key)
    used_r |= set(pd.read_csv(O2 / 'results/plan_manifest.csv', low_memory=False).raster_key) | set(pd.read_csv(K.O6 / 'plan' / 'manifest_b.csv').raster_key)
    used_s = set()
    for p, col in ((K.K4.K3.K2.K1.S345_PLAN, 'noise_seed'), (K.ROOT / 'outputs/surface_revision/s12_resource01/manifest.csv', 'noise_seed'), (O2 / 'results/plan_manifest.csv', 'seed'),
                   (K.O6 / 'plan' / 'manifest_b.csv', 'seed')):
        used_s |= set(pd.read_csv(p, low_memory=False, usecols=[col])[col].dropna().astype(np.int64))
    stored = set()
    for p in (K.ROOT / 'outputs' / 'surface_revision').rglob('*.json'):
        if p.parent.name == 'images' and 's3_code07' not in p.parts:
            stored.add(p.stem)
    fk = set(M.image_key)
    info = dict(formal_rasters_in_executed=sorted(set(M.raster_key) & used_r)[:10], n_formal_rasters=int(M.raster_key.nunique()), n_executed_rasters=len(used_r),
                formal_seeds_in_executed=sorted(set(M.seed.dropna().astype(np.int64)) & used_s)[:10], n_executed_seeds=len(used_s),
                stored_image_records_scanned=len(stored), formal_image_keys_with_stored_image=sorted(fk & stored)[:10],
                formal_run_dirs=[K.rel(p) for p in (K.ROOT / 'outputs' / 'surface_revision').glob('s3_code0*/formal_run')],
                note='CODE-04 registered the N list (manifest) but no N image was executed; positions compared with every executed / registered set')
    return N, info


# Check expected pair counts and prepare the encoding execution plan.
def run(write: bool = False) -> dict:
    import c4_collect as CO
    gen = {}
    import c7_config as CF
    M = pd.read_csv(MAN, low_memory=False); M['layout'] = M.layout.astype(str)
    fz = K.load_json(K.O4 / 'formal' / 'frozen.json'); fps = {l: K.K4.fingerprint(l) for l in ('decode', 'eval', 'diag')}
    E = CO.expected(M, fps); gen['expected_keys_formal.csv'] = E
    EP = expected_pairs(E, M); gen['pairs_expected.csv'] = EP
    S = subset(M); gen['startup_subset.csv'] = S
    gen['cases_formal.json'] = dict(cases=FIXED + SUPP, posthoc=POSTHOC, rules=dict(
        fixed='inherited from S3-CODE-06 configs/formal.json (unchanged); sigma 0 -> rep 0, sigma > 0 -> registered rep',
        supplement='same layout / level / phase / repetition on B_curv and M10_curv (sigma 0 and 40); lines M24 also sigma 40',
        risk='main coded method records (dots DF use, lines R) with >= 20 accepted points: max w / (c + w); ties larger w, then eval_key ascending',
        collapse='main coded method records at sigma 20: max drop of global C_tau vs the sigma-0 record of the same family / surface / layout / pattern / phase / rule / arm; ties eval_key ascending'),
        note='frozen before imaging (package freeze); never replaced by better-looking cases')
    N, mat = n_material(M); gen['n_material.csv'] = N
    files = {}
    for f, obj in gen.items():
        p_ = K.PLAN / f
        if write:
            (K.save_json(p_, obj) if f.endswith('.json') else K.save_csv(obj, p_))
        if f.endswith('.json'):
            same = p_.exists() and K.load_json(p_) == json.loads(json.dumps(K.clean(obj), ensure_ascii=False))
        else:
            tmp = pd.read_csv(p_, low_memory=False) if p_.exists() else None
            new = pd.read_csv(__import__('io').StringIO(obj.to_csv(index=False)), low_memory=False)
            same = tmp is not None and tmp.shape == new.shape and bool((tmp.astype(object).fillna('<NA>').map(str) == new.astype(object).fillna('<NA>').map(str)).all().all())
        files[f] = dict(equal_to_frozen_file=bool(same), sha256=(K.sha_file(p_) if p_.exists() else None))
    seeds = M[M.sigma > 0]
    raw_96 = sum(p.stat().st_size for p in (K.O6 / 'real' / 'records').rglob('*') if p.is_file()) / 2 ** 20
    disk = shutil.disk_usage(K.ROOT).free / 2 ** 30
    chk = dict(manifest=K.ref(MAN), class_map=K.ref(K.O4 / 'formal' / 'class_map.csv'), freeze_s3_code04=dict(file=K.rel(K.O4 / 'formal' / 'frozen.json'), closure_sha256=fz['closure_sha256'],
               mismatches=len(K.K4.verify_closure(fz['closure']))),
               fingerprints_now=fps, fingerprints_s3_code04=fz['fingerprints'], fingerprints_equal=fps == fz['fingerprints'],
               images=len(M), unique_images=int(M.image_key.nunique()), by_family=M.groupby('family').size().to_dict(), sigma0=int((M.sigma == 0).sum()),
               by_sigma=M.groupby('sigma').size().to_dict(), by_surface=M.groupby('surface').size().to_dict(), by_layout=M.groupby(['family', 'layout']).size().astype(int).to_dict(),
               phases=sorted(M.phase_id.unique()), reps=seeds.groupby(['family', 'surface', 'layout', 'pattern', 'phase_id', 'sigma']).rep.apply(lambda x: sorted(x)).astype(str).value_counts().to_dict(),
               expected=E.groupby('layer').size().to_dict(), expected_by_family=E.groupby(['layer', 'family']).size().astype(int).to_dict(), expected_total=len(E),
               expected_unique=bool(E.key.is_unique), pairs_expected=int(EP.n_pairs.sum()), pairs_by_type=EP.groupby('pair_type').n_pairs.sum().to_dict(),
               subset=dict(n=len(S), keys=S.image_key.tolist(), parity=S.groupby(['pattern', 'phase_id']).code_parity.unique().apply(list).astype(str).to_dict()),
               sigma0_seed_null=bool(M[M.sigma == 0].seed.isna().all()), one_seed_per_family_surface_layout_phase_rep=bool(seeds.groupby(['family', 'surface', 'layout', 'phase_id', 'rep']).seed.nunique().eq(1).all()),
               seed_shared_across_patterns_and_sigma=bool(seeds.groupby(['family', 'surface', 'layout', 'phase_id', 'rep']).seed.nunique().eq(1).all()),
               reps_distinct_seeds=bool(seeds.groupby(['family', 'surface', 'layout', 'pattern', 'phase_id', 'sigma']).seed.nunique().eq(2).all()),
               parity_by_phase=M[M.pattern != 'plain'].groupby('phase_id').code_parity.unique().apply(list).astype(str).to_dict(),
               plain_parity_zero=bool((M[M.pattern == 'plain'].code_parity == 0).all()), refs_complete=bool(M[M.family == 'dots'].ref_image_key.notna().all()),
               refs_are_sigma0=bool(set(M[M.family == 'dots'].ref_image_key) <= set(M[M.sigma == 0].image_key)),
               pair_key_four_patterns=bool(M.groupby('pair_key').pattern.nunique().eq(4).all()), peak_max=float(M.peak_f32.max()), budget_ok=bool(M.budget_ok.all()),
               Q_values=sorted(set(np.round(M.Q.astype(float), 3)))[:6], n_material=mat, n_positions=N.to_dict('records'),
               plan_files=files, written=write, resources=dict(raw_records_96_MiB=round(raw_96, 1), raw_estimate_6336_GiB=round(raw_96 * 66 / 1024, 2), free_disk_GiB=round(disk, 1), workers_default=6))
    chk['ok'] = bool(chk['images'] == 6336 and chk['unique_images'] == 6336 and chk['sigma0'] == 576 and chk['expected'] == dict(decode=17424, diag=38016, eval=24552)
                     and chk['expected_unique'] and chk['pairs_expected'] == 38016 and chk['fingerprints_equal'] and chk['freeze_s3_code04']['mismatches'] == 0 and len(S) == 24
                     and chk['sigma0_seed_null'] and chk['one_seed_per_family_surface_layout_phase_rep'] and chk['reps_distinct_seeds'] and chk['plain_parity_zero'] and chk['refs_complete']
                     and chk['refs_are_sigma0'] and chk['pair_key_four_patterns'] and chk['budget_ok'] and not mat['formal_rasters_in_executed'] and not mat['formal_seeds_in_executed']
                     and not mat['formal_image_keys_with_stored_image'] and not mat['formal_run_dirs']
                     and all(v['equal_to_frozen_file'] for v in files.values()))
    return chk


if __name__ == '__main__':
    r = run(write=(sys.argv[1] == 'write'))
    if len(sys.argv) > 2:
        K.checks_add(sys.argv[2], r)
    print(json.dumps(K.clean({k: v for k, v in r.items() if k not in ('n_positions',)}), indent=1, ensure_ascii=False)[:6000])
    print(pd.DataFrame(r['n_positions']).to_string())
