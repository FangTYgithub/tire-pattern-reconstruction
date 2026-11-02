# Assemble continuous-line imaging and decoding settings from named configurations.

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs' / 'line01r' / 'b'
A_OUT = ROOT / 'outputs' / 'line01r' / 'a'
CASES = OUT / 'cases'
LEDGER_DIR = OUT / 'ledger'
TASK_FILE = Path(r'C:\Users\16944\Documents\reference\2026-09-21\dang\outputs\research_runs\tasks\LINE-01R-B_protocol.md')
AUDIT_FILE = Path(r'C:\Users\16944\Documents\reference\2026-09-21\dang\outputs\research_runs\audits\LINE-01R-A_review.md')
AUDIT_JSON = AUDIT_FILE.parent / 'LINE-01R-A_review.json'
TASK_ID = 'LINE-01R-B'

sys.path.insert(0, str(HERE))
import l01r_common as A

now_local, now_utc, sha_file, sha_bytes, sha_array = A.now_local, A.now_utc, A.sha_file, A.sha_bytes, A.sha_array
clean, save_json, load_json, save_text, save_csv, rel = A.clean, A.save_json, A.load_json, A.save_text, A.save_csv, A.rel
surface, omega, geom, grid_truth, design_old, load_old_case = A.surface, A.omega, A.geom, A.grid_truth, A.design, A.load_old_case


R_MM, DELTA_MM, E_MM = 20.0, 1.0, 0.0
Z0_MM, Z1_MM = -E_MM - DELTA_MM, -E_MM + R_MM + DELTA_MM
FLOAT_TOL_MM = 1e-6
PSF_PX = 1.0
Q_BUDGET = 11695.0
PEAK_FIXED, PEAK_BRIDGE, PEAK_CAP = 1.0, 0.5, 1.0
PHASES = (-5 / 16, 7 / 16)
SIGMAS_B1, SIGMAS_B2, SIGMA_LEVELS = (0.0, 20.0, 40.0), (0.0, 30.0), (0.0, 20.0, 30.0, 40.0)
SEED_BASE = 10200000
SURFACES = ('baseline', 'curv_x_strong')
PHIS = (0.0, 45.0)
WIDTHS_B1, PITCH_B1 = (0.5, 1.5), 8.0
PITCHES_B2, W_B2 = (4.45, 6.3), 1.0
REGIMES = ('fixed_peak', 'equal_integral')
D_LIST, TAU_LIST, D_MAIN, TAU_MAIN = A.D_LIST, A.TAU_LIST, 6.0, 0.15
RENDER_CAP_RESEARCH, RENDER_CAP_TESTS, PLANNED = 192, 64, 176
GATE_PX = 3.0

PATHS = {'P1_C1_hist': dict(detector='rowcentroid'),
         'P2_C1_guard': dict(detector='rowcentroid_ext', guard_min_chain_px=6.0, guard_width_window='auto_geom', guard_width_mult=(0.5, 2.0), vote_valid_only=True),
         'P4_C3_cal': dict(detector='steger', steger_threshold_rule='design_or_noise', guard_min_chain_px=6.0, vote_valid_only=True)}
P3_VARIANTS = {'P3_k2': dict(detector='rowcentroid_ext', binarise_rule='otsu_noise_floor', noise_k=2.0, guard_min_chain_px=6.0, guard_width_window='auto_geom', guard_width_mult=(0.5, 2.0), vote_valid_only=True),
               'P3_k3': dict(detector='rowcentroid_ext', binarise_rule='otsu_noise_floor', noise_k=3.0, guard_min_chain_px=6.0, guard_width_window='auto_geom', guard_width_mult=(0.5, 2.0), vote_valid_only=True)}
PATHS_A_REPLAY = {'P2_C1_guard_A': dict(detector='rowcentroid_ext', guard_min_chain_px=6.0, guard_width_window='auto', guard_width_mult=(0.5, 2.0), vote_valid_only=False)}


def p3_applies(spec: dict) -> bool:

    return spec['block'] == 'B1' and spec['regime'] == 'fixed_peak' and float(spec['sigma']) in (20.0, 40.0)


def paths_override() -> dict | None:

    p = os.environ.get('LINE01RB_PATHS_JSON')
    if not p:
        return None
    d = json.loads(Path(p).read_text(encoding='utf-8'))
    out = {}
    for name, pp in d.items():
        pp = dict(pp)
        if isinstance(pp.get('steger_noise_coef'), dict):
            pp['steger_noise_coef'] = {float(k): float(v) for k, v in pp['steger_noise_coef'].items()}
        if isinstance(pp.get('guard_width_mult'), list):
            pp['guard_width_mult'] = tuple(pp['guard_width_mult'])
        out[name] = pp
    return out


def paths_for(spec: dict) -> dict:
    ov = paths_override()
    if ov is not None:
        return dict(ov)
    return {**PATHS, **(P3_VARIANTS if p3_applies(spec) else {})}


def out_dir() -> Path:

    return Path(os.environ.get('LINE01RB_OUT_DIR', str(OUT)))


def render_cap() -> int:

    return int(os.environ.get('LINE01RB_RENDER_CAP', RENDER_CAP_RESEARCH))


def set_budget_context(cases_dir, cap: int | None = None, out: Path | None = None) -> dict:

    d = set_ledger_dir(cases_dir)
    if cap is not None:
        os.environ['LINE01RB_RENDER_CAP'] = str(int(cap))
    if out is not None:
        os.environ['LINE01RB_OUT_DIR'] = str(out)
    return dict(ledger_dir=str(d), render_cap=render_cap(), test_cap=RENDER_CAP_TESTS, out_dir=str(out_dir()), cases_dir=str(cases_dir))


def window_cfg() -> dict:

    return {'measurement_volume_mm': Z1_MM - Z0_MM, 'search_z_min_mm': Z0_MM, 'search_margin_mm': FLOAT_TOL_MM}


def m4_cfg(peak: float) -> dict:
    return {**A.m4_cfg(), **window_cfg(), 'pattern_peak': float(peak), 'psf_sigma_px': PSF_PX}


def m3_cfg(sigma: float, seed: int) -> dict:
    return A.m3_cfg(sigma, seed)


def check_window(r: dict) -> dict:

    eff = dict(z0=float(r['window_z0_mm']), z1=float(r['window_z1_mm']), margin=float(r['depth_margin_mm']), width=float(r['window_z1_mm']) - float(r['window_z0_mm']))
    ok = abs(eff['z0'] - Z0_MM) < 1e-9 and abs(eff['z1'] - Z1_MM) < 1e-9 and abs(eff['margin'] - FLOAT_TOL_MM) < 1e-12
    if not ok:
        raise RuntimeError(f'effective window {eff} differs from the registered [{Z0_MM}, {Z1_MM}] / tol {FLOAT_TOL_MM}')
    return dict(eff, registered=dict(z0=Z0_MM, z1=Z1_MM, width=Z1_MM - Z0_MM, float_tol_mm=FLOAT_TOL_MM), controlling_keys=dict(search_z_min_mm='z0', measurement_volume_mm='z1 - z0', search_margin_mm='acceptance tolerance (explicit, priority over depth_margin_frac)'), ok=True)


SRC_FILES = ('m1_tire_surface.py', 'm2_pattern.py', 'm3_virtual_camera.py', 'm4_reconstruction.py', 'm4_lines.py', 'surface_eval_r3.py', 'm5_evaluation.py')
_SRC: dict = {}


def src_versions() -> dict:
    if not _SRC:
        _SRC.update({f'src/{n}': sha_file(ROOT / 'src' / n) for n in SRC_FILES})
        _SRC['scripts/line01/l01_truth.py'] = sha_file(ROOT / 'scripts' / 'line01' / 'l01_truth.py')
    return dict(_SRC)


def check_freeze(path, stage: str = '') -> dict:

    fz = load_json(path); sv = src_versions(); bad = []
    bad += [f'src:{k}' for k, v in fz.get('src', {}).items() if sv.get(k) != v]
    bad += [f'script:{k}' for k, v in fz.get('scripts', {}).items() if not (ROOT / k).exists() or sha_file(ROOT / k) != v]
    bad += [f'config:{k}' for k, v in fz.get('config', {}).items() if not (ROOT / k).exists() or sha_file(ROOT / k) != v]
    if bad:
        raise RuntimeError(f'freeze mismatch ({stage or "check"}): {bad}')
    return dict(ok=True, stage=stage, freeze_id=fz.get('freeze_id'), n_checked=len(fz.get('src', {})) + len(fz.get('scripts', {})) + len(fz.get('config', {})))


def geometry_record() -> dict:
    g = geom(); c = g['cfg']
    return dict(focal_length_mm=float(c['focal_length_mm']), pixel_pitch_mm=float(c['pixel_pitch_mm']), baseline_mm=float(c['baseline_mm']), working_dist_mm=float(c['working_dist_mm']),
                image_width_px=int(c['image_width_px']), image_height_px=int(c['image_height_px']), f_px=float(g['f_px']), mm_per_px_at_wd=float(g['mm_per_px']), theta_deg=float(g['theta_deg']))


def surface_record(name: str) -> dict:
    mesh, cfg, ref_fn, desc = surface(name)
    om = omega(name)
    return dict(name=name, reference=clean(desc), omega_hash=om['hash'], n_omega_nodes=int(om['n_nodes']), dent_disc_r_mm=float(A.C0.DENT_R_MM))


def forward_key(spec: dict, raster_sha: str, peak: float) -> str:

    sv = src_versions()
    fk = dict(pattern=dict(type='olines', s_mm=float(spec['s_mm']), w_mm=float(spec['w_mm']), phi_deg=float(spec['phi_deg']), phase_frac=float(spec['phase_frac']), raster_sha256=raster_sha, generator_sha=sv['src/m2_pattern.py']),
              brightness=dict(peak=round(float(peak), 12), regime=spec['regime']), surface=surface_record(spec['surface']), geometry=geometry_record(), psf_sigma_px=PSF_PX,
              noise=dict(model='M3: PSF -> clip[0,1] -> + N(0, sigma/255) seeded (mottle_seed + 104729) -> clip -> uint8', sigma=float(spec['sigma']), seed=int(spec['noise_seed']), renderer_sha=sv['src/m3_virtual_camera.py']),
              m3_other=dict(albedo=1.0, albedo_exposure=1.0, specular_k=0.0, mottle_amp=0.0, outlier_frac=0.0, batch_size=512, deposit='sum', splat='bilinear', rays_per_pixel=1))
    return hashlib.sha256(json.dumps(clean(fk), sort_keys=True).encode()).hexdigest()


def decode_key(path_params: dict, peak: float) -> str:
    sv = src_versions()
    return hashlib.sha256((sv['src/m4_lines.py'] + sv['src/m4_reconstruction.py'] + json.dumps(clean(dict(path_params, window=window_cfg(), pattern_peak=round(float(peak), 12), psf_sigma_px=PSF_PX)), sort_keys=True)).encode()).hexdigest()[:16]


def eval_key() -> str:

    from surface_eval_r3 import DEFORMATION_VERSION, DEFORMATION_QUANTILES, EVAL_VERSION
    sv = src_versions()
    body = dict(d=list(D_LIST), tau=list(TAU_LIST), support_rule='exact', eval_version=EVAL_VERSION, deformation=DEFORMATION_VERSION, q=list(DEFORMATION_QUANTILES),
                surfaces={s: dict(omega_hash=omega(s)['hash'], nodes=surface_record(s)['reference']) for s in SURFACES}, dent_disc_r_mm=float(A.C0.DENT_R_MM), gate_px=GATE_PX)
    return hashlib.sha256((sv['src/surface_eval_r3.py'] + sv['src/m5_evaluation.py'] + sv['scripts/line01/l01_truth.py'] + json.dumps(clean(body), sort_keys=True)).encode()).hexdigest()[:16]


def noise_realisation(surface_name: str, phase_index: int, sigma: float) -> tuple[str, int]:

    idx = (SURFACES.index(surface_name) * len(PHASES) + int(phase_index)) * len(SIGMA_LEVELS) + SIGMA_LEVELS.index(float(sigma))
    return f'N{idx:02d}_{surface_name}_ph{phase_index}_s{int(sigma)}', SEED_BASE + idx


TRIAL = [('B1', 'baseline', 0.0, 0.5, 8.0, 'fixed_peak', 20.0, 0), ('B1', 'curv_x_strong', 45.0, 1.5, 8.0, 'equal_integral', 40.0, 1), ('B1', 'baseline', 45.0, 0.5, 8.0, 'equal_integral', 0.0, 1),
         ('B1', 'curv_x_strong', 0.0, 1.5, 8.0, 'fixed_peak', 0.0, 0), ('B2', 'baseline', 0.0, 1.0, 4.45, 'fixed_peak', 30.0, 0), ('B2', 'curv_x_strong', 45.0, 1.0, 6.3, 'equal_integral', 0.0, 1),
         ('BR', 'baseline', 0.0, 0.5, 8.0, 'fixed_peak_0.5', 40.0, 0), ('BR', 'curv_x_strong', 45.0, 1.5, 8.0, 'fixed_peak_0.5', 40.0, 1)]


def _rows() -> list[dict]:
    rows = []
    for surf, phi, w, regime, sig, pi in itertools.product(SURFACES, PHIS, WIDTHS_B1, REGIMES, SIGMAS_B1, range(len(PHASES))):
        rows.append(dict(block='B1', surface=surf, phi_deg=phi, w_mm=w, s_mm=PITCH_B1, regime=regime, sigma=sig, phase_index=pi, purpose='width x brightness regime x orientation at the common normal pitch 8 mm'))
    for surf, phi, s, regime, sig, pi in itertools.product(SURFACES, PHIS, PITCHES_B2, REGIMES, SIGMAS_B2, range(len(PHASES))):
        rows.append(dict(block='B2', surface=surf, phi_deg=phi, w_mm=W_B2, s_mm=s, regime=regime, sigma=sig, phase_index=pi, purpose='boundary / hard pitches: candidate sets and numbering'))
    for surf, phi, w, pi in itertools.product(SURFACES, PHIS, WIDTHS_B1, range(len(PHASES))):
        rows.append(dict(block='BR', surface=surf, phi_deg=phi, w_mm=w, s_mm=PITCH_B1, regime='fixed_peak_0.5', sigma=40.0, phase_index=pi, purpose='brightness bridge: the B1 fixed-peak sigma-40 configuration at peak 0.5 (same noise field)'))
    return rows


def pattern_for(spec: dict):
    import m2_pattern as M2
    pat, meta = M2.generate_pattern('olines', spacing_mm=float(spec['s_mm']), feature_size_mm=float(spec['w_mm']), line_angle_deg=float(spec['phi_deg']), line_phase_frac=float(spec['phase_frac']))
    return np.asarray(pat, np.float32), meta


def budget_of(spec: dict, integral: float) -> dict:

    if spec['regime'] == 'fixed_peak':
        peak, req = PEAK_FIXED, PEAK_FIXED
    elif spec['regime'] == 'fixed_peak_0.5':
        peak, req = PEAK_BRIDGE, PEAK_BRIDGE
    elif spec['regime'] == 'equal_integral':
        req = Q_BUDGET / integral if integral > 0 else float('inf'); peak = req
    else:
        raise ValueError(spec['regime'])
    feasible = bool(req <= PEAK_CAP + 1e-12)
    return dict(integral_base=float(integral), required_peak=float(req), peak_cap=PEAK_CAP, peak=(float(peak) if feasible else None), integral_scaled=(float(peak * integral) if feasible else None),
                feasible=feasible, infeasible_reason=(None if feasible else f'required peak {req:.3f} > cap {PEAK_CAP} (not rendered, not clipped; kept in the design denominator)'))


_MANIFEST: list | None = None


def manifest() -> list[dict]:

    global _MANIFEST
    if _MANIFEST is not None:
        return _MANIFEST
    rows = _rows()
    for i, r in enumerate(rows):
        r['phase_frac'] = float(PHASES[r['phase_index']])
        r['noise_realisation'], r['noise_seed'] = noise_realisation(r['surface'], r['phase_index'], r['sigma'])
        pat, meta = pattern_for(r)
        integral = float(pat.sum())
        r.update(budget_of(r, integral)); r['raster_sha256'] = sha_array(pat); r['lit_px'] = int((pat > 0).sum()); r['n_lines'] = int(meta['olines']['n_lines']); r['width_px'] = float(meta['olines']['width_px']); r['spacing_px'] = float(meta['olines']['spacing_px'])
        r['regime_pair_id'] = hashlib.sha256(json.dumps([r['surface'], r['phi_deg'], r['w_mm'], r['s_mm'], r['sigma'], r['phase_index']]).encode()).hexdigest()[:12]
        r['brightness_pair_id'] = (hashlib.sha256(json.dumps(['bridge', r['surface'], r['phi_deg'], r['w_mm'], r['phase_index']]).encode()).hexdigest()[:12] if (r['block'] == 'BR' or (r['block'] == 'B1' and r['regime'] == 'fixed_peak' and r['sigma'] == 40.0)) else None)
        r['phase_block'] = f"ph{r['phase_index']}"
        r['trial'] = (r['block'], r['surface'], r['phi_deg'], r['w_mm'], r['s_mm'], r['regime'], r['sigma'], r['phase_index']) in TRIAL
        r['forward_key'] = forward_key(r, r['raster_sha256'], r['peak'] if r['peak'] is not None else -1.0)
        r['case_id'] = r['forward_key']
        r['planned_index'] = i
    keys = [r['forward_key'] for r in rows]
    assert len(set(keys)) == len(keys), 'duplicate forward keys in the manifest'
    _MANIFEST = rows
    return rows


def manifest_df():
    import pandas as pd
    return pd.DataFrame(manifest())


def spec_of(case_id: str) -> dict:
    for r in manifest():
        if r['case_id'] == case_id:
            return r
    raise KeyError(case_id)


def progress_file(which: str) -> Path:
    assert which.replace('_', '').isalnum(), which
    return out_dir() / f'progress_{which}.json'


def manifest_from_csv(path) -> list[dict]:

    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    rows = []
    for r in df.to_dict(orient='records'):
        s = dict(r)
        for k in ('phi_deg', 'w_mm', 's_mm', 'sigma', 'phase_frac', 'peak', 'integral_base', 'required_peak'):
            if k in s and s[k] is not None and s[k] == s[k]:
                s[k] = float(s[k])
        for k in ('phase_index', 'noise_seed', 'planned_index'):
            if k in s and s[k] == s[k]:
                s[k] = int(s[k])
        s['feasible'] = bool(s.get('feasible')) and str(s.get('feasible')).lower() not in ('false', '0', 'nan')
        s['trial'] = str(s.get('trial', False)).lower() == 'true'
        if s['feasible']:
            pat, meta = pattern_for(s)
            if sha_array(pat) != s['raster_sha256']:
                raise ValueError(f"manifest row {s.get('case_id', '?')[:12]}: raster hash does not reproduce")
            if forward_key(s, s['raster_sha256'], s['peak']) != s['forward_key'] or s['case_id'] != s['forward_key']:
                raise ValueError(f"manifest row {s.get('case_id', '?')[:12]}: forward key does not reproduce")
        rows.append(s)
    assert len({r['forward_key'] for r in rows}) == len(rows), 'duplicate forward keys in the manifest'
    return rows


def progress(which: str, **kw) -> dict:
    p = progress_file(which)
    if p.exists():
        pr = load_json(p)
    else:
        pr = dict(task_id=TASK_ID, part=which, state='setup', started_local=now_local(), started_utc=now_utc(), stages={}, counts={}, blockers=[], pending_decisions=[],
                  executor='execution (local session)', selfcheck_status='not run', independent_review_status='not started (reference)')
    pr.update(kw); pr['updated_local'] = now_local(); pr['updated_utc'] = now_utc()
    save_json(p, pr)
    return pr


def stage(which: str, name: str, **kw) -> dict:
    pr = progress(which); st = dict(pr.get('stages', {}))
    st[name] = {**st.get(name, {}), **clean(kw), 'updated_local': now_local()}
    return progress(which, stages=st)


def log(msg: str, which: str = 'main') -> None:
    line = f'[{now_local()}] {msg}'
    print(line, flush=True)
    d = out_dir(); d.mkdir(parents=True, exist_ok=True)
    with open(d / f'run_{which}.log', 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


CAPS = {'research': RENDER_CAP_RESEARCH, 'tests': RENDER_CAP_TESTS}


def ledger_dir() -> Path:

    return Path(os.environ.get('LINE01RB_LEDGER_DIR', str(LEDGER_DIR)))


def set_ledger_dir(cases_dir) -> Path:
    d = LEDGER_DIR if Path(cases_dir).resolve() == CASES.resolve() else Path(cases_dir).resolve().parent / 'ledger'
    os.environ['LINE01RB_LEDGER_DIR'] = str(d)
    return d


def ledger_add(group: str, rec: dict) -> None:
    d = ledger_dir(); d.mkdir(parents=True, exist_ok=True)
    with open(d / f'{group}_{os.getpid()}.jsonl', 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(clean(rec), ensure_ascii=False) + '\n')


def ledger_records(group: str | None = None) -> list[dict]:
    recs = []
    d = ledger_dir()
    if d.exists():
        for p in sorted(d.glob('*.jsonl')):
            g = p.name.split('_')[0]
            if group is not None and g != group:
                continue
            for line in p.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    r = json.loads(line); r['_group'] = g; recs.append(r)
    return sorted(recs, key=lambda r: r.get('t_local', ''))


def attempt_table(group: str = 'research') -> list[dict]:

    recs = ledger_records(group)
    rows = {}
    for i, r in enumerate(recs):
        ev = r.get('event')
        if ev is None:
            rows[f'legacy:{i}'] = dict(r, outcome=('ok' if r.get('ok') else 'failed'), started=True, ended=True)
        elif ev == 'start':
            k = r.get('token') or f"{r.get('case_id')}:{r.get('t_local')}:{r.get('pid')}"
            rows.setdefault(k, dict(outcome='interrupted', started=True, ended=False)).update({kk: v for kk, v in r.items() if kk != 'event'})
        elif ev == 'end':
            k = r.get('token') or f"{r.get('case_id')}:{r.get('t_start')}:{r.get('pid')}"
            row = rows.setdefault(k, dict(started=True, ended=False))
            row.update({kk: v for kk, v in r.items() if kk != 'event'}); row['ended'] = True
            row['outcome'] = r.get('outcome') or ('ok' if r.get('ok') else 'failed')
    return list(rows.values())


def ledger_counts() -> dict:
    out = {}
    for g in ('research', 'tests'):
        cap = render_cap() if g == 'research' else RENDER_CAP_TESTS
        rows = attempt_table(g)
        by = {}
        for r in rows:
            b = by.setdefault(r.get('purpose', '?'), dict(attempts=0, ok=0, failed=0, interrupted=0)); b['attempts'] += 1; b[r['outcome'] if r['outcome'] in b else 'failed'] += 1
        out[g] = dict(attempts=len(rows), ok=sum(1 for r in rows if r['outcome'] == 'ok'), failed=sum(1 for r in rows if r['outcome'] == 'failed'), interrupted=sum(1 for r in rows if r['outcome'] == 'interrupted'),
                      unique_images=len({r.get('image_sha256') for r in rows if r['outcome'] == 'ok' and r.get('image_sha256')}), cap=cap, by_purpose=by)
    return out


def reconcile_ledger(group: str = 'research') -> list[dict]:

    closed = []
    for r in attempt_table(group):
        if r.get('started') and not r.get('ended'):
            rec = dict(event='end', token=r.get('token'), case_id=r.get('case_id'), purpose=r.get('purpose'), t_local=now_local(), t_start=r.get('t_local'), pid_start=r.get('pid'), ok=False, outcome='interrupted', error='attempt started but never ended (process interrupted); reconciled at restart')
            ledger_add(group, rec); closed.append(rec)
    return closed


def budget_check(group: str, n: int = 1) -> None:
    c = ledger_counts()[group]
    if c['attempts'] + n > c['cap']:
        raise RuntimeError(f'render cap of group {group} exhausted: {c["attempts"]} + {n} > {c["cap"]}')


def render_research(mesh: dict, pattern: np.ndarray, cfg: dict, case_id: str, purpose: str = 'B_research', attempt: int = 1, token: str | None = None):

    from m3_virtual_camera import synthesise_camera_image
    budget_check('research')
    tok = token or f'{case_id[:12]}:{os.getpid()}:{int(time.time() * 1000)}'
    t_start = now_local()
    ledger_add('research', dict(event='start', token=tok, t_local=t_start, purpose=purpose, case_id=case_id, attempt=int(attempt), pid=os.getpid(), sigma=cfg.get('noise_sigma'), seed=cfg.get('mottle_seed'), render_cap=render_cap()))
    rec = dict(event='end', token=tok, t_local=None, t_start=t_start, purpose=purpose, case_id=case_id, attempt=int(attempt), ok=False, outcome='failed', seconds=None, image_sha256=None, error=None, pid=os.getpid())
    t0 = time.monotonic()
    try:
        img, hit, pts3d, info = synthesise_camera_image(mesh_dict=mesh, pattern=pattern, cfg=cfg)
    except Exception as e:
        rec.update(t_local=now_local(), seconds=round(time.monotonic() - t0, 3), error=repr(e)[:300]); ledger_add('research', rec); raise
    rec.update(t_local=now_local(), ok=True, outcome='ok', seconds=round(time.monotonic() - t0, 3), image_sha256=sha_bytes(img.tobytes()), n_rays=int(info['n_rays_cast']), n_hits=int(info['n_hits']), n_valid_pixels=int(info['n_valid_pixels']))
    ledger_add('research', rec)
    return img, info


def wrapped_render_tests(purpose: str):

    import m3_virtual_camera as M3
    orig = M3.synthesise_camera_image

    def f(*a, **k):
        if ledger_counts()['tests']['attempts'] + 1 > RENDER_CAP_TESTS:
            raise RuntimeError(f'test-render cap ({RENDER_CAP_TESTS}) exhausted')
        rec = dict(t_local=now_local(), purpose=purpose, ok=False)
        t0 = time.monotonic()
        try:
            out = orig(*a, **k)
        except Exception as e:
            rec.update(error=repr(e)[:200], seconds=round(time.monotonic() - t0, 3)); ledger_add('tests', rec); raise
        rec.update(ok=True, seconds=round(time.monotonic() - t0, 3), image_sha256=sha_bytes(out[0].tobytes())); ledger_add('tests', rec)
        return out
    return orig, f


def image_resource_stats(img: np.ndarray, peak: float, sigma: float) -> dict:

    from m4_lines import binarise, estimate_noise_sigma
    thr, binary = binarise(img); fg = binary > 0
    sig_hat, status, frac, zf = estimate_noise_sigma(img)
    return dict(max=int(img.max()), mean_fg=(float(img[fg].mean()) if fg.any() else None), p99=float(np.percentile(img, 99)), fg_frac=float(fg.mean()), frac_255=float((img == 255).mean()), frac_ge_250=float((img >= 250).mean()),
                frac_0=float((img == 0).mean()), fg_frac_255=(float((img[fg] == 255).mean()) if fg.any() else None), otsu_threshold=float(thr), sigma_hat=sig_hat, sigma_hat_status=status, nominal_line_peak_grey=255.0 * float(peak),
                sigma_over_255P=(float(sigma) / (255.0 * float(peak)) if peak and peak > 0 else None), note='clipping fractions are software limits of the model, not sensor saturation')


REQUIRED_ARRAYS = ('image', 'ref_k', 'ref_v', 'ref_u', 'ref_inner')
PATH_ARRAYS_FULL = ('features', 'status', 'line_id', 'chain', 'segment', 'truth_k', 'confirmable', 'points', 'statusO', 'pointsO', 'run_width', 'n_candidates', 'guard_reason', 'multi_idx', 'multi_ids')


def record_state(cases_dir: Path, case_id: str, fkey: str, dkeys: dict, ekey: str, strict: bool = False) -> tuple[str, str]:

    pj = cases_dir / f'{case_id}.json'; pz = cases_dir / f'{case_id}.npz'
    if not pj.exists():
        return 'missing', 'no json'
    try:
        r = load_json(pj)
        if r.get('status') != 'ok':
            return 'json_bad', f"status {r.get('status')}"
    except Exception as e:
        return 'json_bad', str(e)[:80]
    if not pz.exists():
        return 'npz_missing', 'no npz'
    try:
        with np.load(pz) as z:
            names = set(z.files)
            if not all(a in names for a in REQUIRED_ARRAYS):
                return 'npz_bad', f'missing arrays {sorted(set(REQUIRED_ARRAYS) - names)}'
            img = np.asarray(z['image'])
            for n in list(r.get('paths', {})):
                if f'{n}_points' not in names:
                    return 'npz_bad', f'missing {n}_points'
                pts = z[f'{n}_points']
                if strict:
                    for a in PATH_ARRAYS_FULL:
                        if f'{n}_{a}' not in names:
                            return 'npz_bad', f'missing {n}_{a}'
                        arr = z[f'{n}_{a}']
                        if a in ('features', 'status', 'line_id', 'chain', 'segment', 'truth_k', 'confirmable', 'statusO', 'pointsO', 'n_candidates', 'guard_reason') and len(arr) != len(pts):
                            return 'npz_bad', f'{n}_{a} length {len(arr)} != {len(pts)}'
                    if int(r['paths'][n]['denominators']['n_detected']) != len(pts):
                        return 'npz_bad', f'{n}: json n_detected {r["paths"][n]["denominators"]["n_detected"]} != npz {len(pts)}'
    except Exception as e:
        return 'npz_bad', str(e)[:80]
    if sha_bytes(img.tobytes()) != r.get('image', {}).get('sha256'):
        return 'image_mismatch', 'npz image hash != json'
    if r.get('forward_key') != fkey:
        return 'stale_forward', 'forward key differs'
    if set(r.get('paths', {})) != set(dkeys) or any(r['paths'][n].get('decode_key') != dkeys[n] for n in dkeys):
        return 'stale_decode', 'decode key / path set differs'
    if r.get('eval_key') != ekey:
        return 'stale_eval', 'evaluation key differs'
    return 'valid', ''


def classify(rec: dict) -> str:

    if rec.get('status') != 'ok':
        return 'program_error'
    if rec.get('render', {}).get('n_valid_pixels', 1) == 0 or all(p['counts']['n_detected'] == 0 for p in rec['paths'].values()):
        return 'no_output'
    for n in ('P2_C1_guard', 'P4_C3_cal'):
        p = rec['paths'].get(n)
        if p and ((p['P'].get('pts_ref_gross_rate') or 0.0) > 0.01 or p['P'].get('n_accepted', 0) < 100):
            return 'science_fail'
    return 'ok'


def env_record() -> dict:
    return A.env_record()
