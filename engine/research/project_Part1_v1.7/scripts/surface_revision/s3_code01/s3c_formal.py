# Construct encoding cases and the corresponding control arms.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

import s3c_common as K

FORMAL = K.FORMAL
SEED_BASE = 11300000
FAMILY_INDEX = {'dots': 0, 'lines': 1}
SURF_INDEX = {s: i for i, s in enumerate(K.MAIN)}
PATTERNS = ('plain', 'weak', 'mid', 'strong')
EXIT = dict(K.EXIT, interrupted=9)


def seed_of(fam: str, surf: str, phase_idx: int, rep: int, sigma: float):
    return None if float(sigma) == 0 else SEED_BASE + 10000 * FAMILY_INDEX[fam] + 100 * (SURF_INDEX[surf] * 8 + int(phase_idx)) + int(rep)


def m3_digest() -> str:
    import rdy_inputs as IN
    return IN.m3_digest()


def raster_key(binary) -> str:
    import rdy_inputs as IN
    return IN.raster_key(binary)


def build_pattern(fam: str, layout: str, pattern: str, phase: dict, levels: dict) -> dict:
    import s3c_precheck as PC
    if fam == 'dots':
        radii = [PC.R0] if pattern == 'plain' else list(levels['dots']['radii'][pattern])
        R = PC.dot_raster(layout, phase['u'], phase['v'], radii)
        return dict(binary=(np.asarray(R['pat']) > 0).astype(np.float32), centres=R['c'], classes=R['cls'], radii=radii, meta_full=None)
    wn, ww = (None, None) if pattern == 'plain' else levels['lines']['widths'][pattern]
    R = PC.line_raster(float(layout), phase['phase_frac'], wn, ww)
    return dict(binary=(np.asarray(R['pat']) > 0).astype(np.float32), centres=None, classes=None, widths=(None if wn is None else [wn, ww]), meta_full=R['meta'])


def image_identity(row: dict) -> str:
    return 'S3CI-' + K.digest(dict(raster=row['raster_key'], peak_f32=row['peak_f32'], surface=K.RK.SURF_MAP[row['surface']], psf=K.PSF_PX, sigma=float(row['sigma']),
                                   seed=(None if pd.isna(row['seed']) else int(row['seed'])), m3=row['m3_digest'], platform=K.PLATFORM_VERSION), 24)


def seen_seeds() -> set:
    s = set()
    for p, col in ((K.S345_PLAN, 'noise_seed'), (K.S12 / 'manifest.csv', 'noise_seed')):
        d = pd.read_csv(p, low_memory=False, usecols=[col]); s |= set(pd.to_numeric(d[col], errors='coerce').dropna().astype(np.int64))
    return s


def seen_rasters() -> set:
    keys = set(pd.read_csv(K.S12 / 'inputs.csv', low_memory=False, usecols=['raster_key']).raster_key)
    for f in (K.S345_RUN / 'images').glob('*.json'):
        try:
            keys.add(K.load_json(f)['identity_fields']['raster'])
        except Exception:
            pass
    return keys


# Assemble experiment rows with their configuration and input identities.
def build_manifest() -> dict:
    import s3c_precheck as PC
    lev = K.load_json(K.RESULTS / 'precheck_levels.json')['levels']; lay = K.load_json(K.RESULTS / 'precheck_layouts.json')['selected']
    ph = PC.new_phases(); m3 = m3_digest(); rows = []; rasters = {}
    for fam in ('dots', 'lines'):
        layouts = [lay[fam]['layout1'], lay[fam]['layout2']]
        phases = [dict(phase_id=p[0], u=p[1], v=p[2]) for p in ph['dots']] if fam == 'dots' else [dict(phase_id=p[0], phase_frac=p[1]) for p in ph['lines']]
        for layout in layouts:
            for pi, phd in enumerate(phases):
                for pattern in PATTERNS:
                    B = build_pattern(fam, layout, pattern, phd, lev); rk = raster_key(B['binary']); res = PC.resource(B['binary'])
                    rasters[rk] = B
                    for surf in K.MAIN:
                        for sigma in K.SIGMAS:
                            for rep in ((0,) if sigma == 0 else (0, 1)):
                                r = dict(family=fam, surface=surf, platform_surface=K.RK.SURF_MAP[surf], layout=layout, pattern=pattern,
                                         level_value=(None if pattern == 'plain' else lev[fam][pattern]), sizes_mm=';'.join(f'{x:.6f}' for x in (B['radii'] if fam == 'dots' else (B['widths'] or [1.0]))),
                                         phase_id=phd['phase_id'], phase_index=pi, u=phd.get('u'), v=phd.get('v'), phase_frac=phd.get('phase_frac'), sigma=float(sigma), rep=rep,
                                         seed=seed_of(fam, surf, pi, rep, sigma), raster_key=rk, Q0=res['Q0'], v_scale=res['v'], peak_f32=res['peak_f32'], Q=res['Q'], usage=res['usage'],
                                         limit=res['limit'], budget_ok=res['budget_ok'], m3_digest=m3,
                                         arms=('use;ignore' if (fam == 'dots' and pattern != 'plain') else ('single;single_common' if fam == 'dots' else ('LC_ignore;LC_rule' if pattern != 'plain' else 'P4;LC_ignore_frontend'))))
                                r['image_key'] = image_identity(r); rows.append(r)
    M = pd.DataFrame(rows)
    FORMAL.mkdir(parents=True, exist_ok=True); K.save_csv(M, FORMAL / 'manifest.csv')

    seen_s, seen_r = seen_seeds(), seen_rasters()
    new_seeds = set(M.seed.dropna().astype(np.int64))
    chk = dict(n_rows=len(M), n_unique_images=int(M.image_key.nunique()), expected=4224, counts_ok=bool(len(M) == 4224 and M.image_key.nunique() == 4224),
               by_family=M.groupby('family').size().to_dict(), by_pattern=M.groupby(['family', 'pattern']).size().to_dict(),
               decode_outputs=int(sum(len(a.split(';')) for a in M.arms)), budget_ok=bool(M.budget_ok.all()), limits=M.groupby(['family', 'layout', 'limit']).size().to_dict(),
               seed_overlap_with_seen=sorted(new_seeds & seen_s)[:10], seeds_ok=not (new_seeds & seen_s), raster_overlap_with_seen=sorted(set(M.raster_key) & seen_r)[:10],
               rasters_ok=not (set(M.raster_key) & seen_r), n_rasters=int(M.raster_key.nunique()),
               pairing_ok=bool(M.groupby(['family', 'surface', 'phase_id', 'sigma', 'rep']).seed.nunique(dropna=False).max() == 1),
               sigma0_once=bool((M[M.sigma == 0].groupby(['family', 'surface', 'layout', 'pattern', 'phase_id']).size() == 1).all()), t_local=K.now_local())
    K.save_json(FORMAL / 'manifest_checks.json', chk)
    R = M.drop_duplicates(['family', 'layout', 'pattern', 'phase_id'])[['family', 'layout', 'pattern', 'level_value', 'sizes_mm', 'phase_id', 'raster_key', 'Q0', 'v_scale', 'peak_f32', 'Q', 'usage', 'limit']]
    K.save_csv(R, FORMAL / 'resources.csv')
    return chk


def code_hashes() -> dict:
    files = [K.HERE / 's3c_dev.py', K.HERE / 's3c_formal.py', K.HERE / 's3c_precheck.py', K.HERE / 's3c_common.py']
    files += [K.RDY / f for f in ('rdy_decode.py', 'rdy_eval.py', 'rdy_inputs.py', 'rdy_common.py', 'rdy_platform.py')]
    files += [K.ROOT / 'scripts' / 'line01r' / f for f in ('lc_pipeline.py', 'lc_readout.py', 'lc_r4_runtime.py', 'lc_r3_reliability.py', 'lc_pattern.py', 'lc_r4_common.py', 'lc_r4_rules.py')]
    files += [K.ROOT / 'scripts' / 'encoding_prep' / f for f in ('ep_detect.py', 'ep_patterns.py')] + [K.ROOT / 'scripts' / 'spotnoise_round5' / 'r5_detect.py']
    files += [K.ROOT / 'scripts' / 'surface_revision' / 'sections_adapter' / 's1r_eval.py'] + [K.ROOT / 'src' / f for f in ('m2_pattern.py', 'm3_virtual_camera.py', 'm4_reconstruction.py', 'm4_lines.py')]
    return {K.rel(f): K.sha_file(f) for f in files}


def write_freeze(rules: dict, root=FORMAL) -> dict:
    allowed = dict(dots={'B0', 'DF', 'DS'}, lines={'use', 'R', 'LB', 'LG'})
    for fam, r in rules.items():
        if r not in allowed[fam]:
            raise K.Stop('check', f'rule {r!r} not registered for {fam}')
    fz = dict(rules=rules, code=code_hashes(), manifest_sha256=K.sha_file(root / 'manifest.csv'), platform=K.PLATFORM_VERSION, t_local=K.now_local())
    K.save_json(root / 'freeze.json', fz); return fz


def verify_freeze(root=FORMAL) -> dict:
    p = root / 'freeze.json'
    if not p.exists():
        raise K.Stop('freeze', 'no freeze record: the formal run needs the rules frozen by the next task book')
    fz = K.load_json(p); now = code_hashes()
    bad = [k for k, h in fz['code'].items() if now.get(k) != h]
    if bad or K.sha_file(root / 'manifest.csv') != fz['manifest_sha256']:
        raise K.Stop('freeze', f'freeze mismatch: {bad[:5]} manifest_equal={K.sha_file(root / "manifest.csv") == fz["manifest_sha256"]}')
    return fz


def write_permit(task_file: str, root=FORMAL, drill=False) -> dict:
    p = os.path.abspath(task_file)
    if not drill and (not os.path.exists(p) or 'S3-CODE-01' in os.path.basename(p)):
        raise K.Stop('permit', 'a NEW task book is required for the formal run (S3-CODE-01 does not authorise imaging)')
    rec = dict(task_file=p, task_sha256=(K.sha_file(p) if os.path.exists(p) else None), drill=bool(drill), t_local=K.now_local())
    K.save_json(root / 'permit.json', rec); return rec


class Lock:
    def __init__(self, p):
        self.p = p

    def __enter__(self):
        self.p.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(self.p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise K.Stop('lock', f'another run holds {self.p}')
        os.write(self.fd, str(os.getpid()).encode()); return self

    def __exit__(self, *exc):
        os.close(self.fd); os.remove(self.p)


class Ledger:
    def __init__(self, root, cap_render: int):
        self.p = root / 'run' / 'ledger.jsonl'; self.cap = cap_render

    def records(self):
        return [json.loads(x) for x in self.p.read_text(encoding='utf-8').splitlines()] if self.p.exists() else []

    def n_render(self):
        return sum(1 for r in self.records() if r['kind'] == 'render' and r['event'] == 'start')

    def add(self, **kw):
        K.guard(self.p); self.p.parent.mkdir(parents=True, exist_ok=True)
        with open(self.p, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(K.clean(dict(t=K.now_local(), **kw))) + '\n')


def _complete(p, ident) -> bool:
    if not p.exists():
        return False
    try:
        r = K.load_json(p)
    except Exception:
        return False
    z = p.with_suffix('.npz')
    return bool(r.get('complete') and r.get('identity') == ident and z.exists() and K.sha_file(z) == r.get('npz_sha256'))


def render_binding(row: dict, B: dict):
    import rdy_platform as P
    bd = P.bound()
    pattern = (B['binary'] * np.float32(row['peak_f32'])).astype(np.float32)
    res = bd.call('render', bd.sid(row['surface']), pattern, float(row['peak_f32']), float(row['sigma']), int(row['seed'] or 0), K.PSF_PX)
    return np.asarray(res['image'])


def render_stub(row: dict, B: dict):

    C = pd.read_csv(K.S3_CASES, low_memory=False); fam = 'point' if row['family'] == 'dots' else 'line'
    ik = sorted(C[C.family == fam].image_key.unique())[int(row['phase_index']) % 8]
    return K.load_npz(K.S345_RUN / 'images' / f'{ik}.npz')['image']


# Render one encoding image and evaluate the requested decoding rules.
def run_unit(row: dict, root, rules: dict, led: Ledger, renderer) -> dict:
    import s3c_dev as DV
    import s3c_precheck as PC
    lev = K.load_json(K.RESULTS / 'precheck_levels.json')['levels']
    phd = dict(phase_id=row['phase_id'], u=row['u'], v=row['v'], phase_frac=row['phase_frac'])
    B = build_pattern(row['family'], row['layout'], row['pattern'], phd, lev)
    if raster_key(B['binary']) != row['raster_key']:
        raise K.Stop('technical', f"{row['image_key']}: raster does not reproduce the manifest key")
    ident = row['image_key']; ip = root / 'images' / f'{ident}.json'
    if _complete(ip, ident):
        img = K.load_npz(ip.with_suffix('.npz'))['image']
    else:
        if led.n_render() >= led.cap:
            raise K.Stop('cap', 'render cap reached')
        led.add(kind='render', event='start', key=ident)
        img = renderer(row, B)
        if img.dtype != np.uint8 or img.shape != (512, 512):
            raise K.Stop('technical', f'render returned {img.dtype} {img.shape}')
        h = K.save_npz(ip.with_suffix('.npz'), image=img)
        K.save_json(ip, dict(identity=ident, row={k: row[k] for k in ('family', 'surface', 'layout', 'pattern', 'phase_id', 'sigma', 'rep', 'seed', 'raster_key', 'peak_f32')}, npz_sha256=h,
                             renderer=renderer.__name__, complete=True, t_local=K.now_local()))
        led.add(kind='render', event='end', key=ident, status='ok')

    import rdy_platform as P
    sid = K.RK.SURF_MAP[row['surface']]; mesh = P.mesh(row['surface'])
    pattern = (B['binary'] * np.float32(row['peak_f32'])).astype(np.float32)
    ev = {}
    if row['family'] == 'dots':
        radii = [float(x) for x in str(row['sizes_mm']).split(';')]; k = 1 if row['pattern'] == 'plain' else 2
        ctx = dict(binary=B['binary'], centres=B['centres'], pattern=pattern, peak=float(row['peak_f32']), size_mm=max(radii), meta_full=None, row=dict(radius_mm=max(radii)))
        D = DV.dot_decode(img, B['binary'], k, radii, rules['dots'], 'enc')
        for arm in D['arms']:
            m, arr = DV.evaluate(sid, 'point', DV.dot_dec(D, arm), ctx, mesh); ev[arm] = (m, arr)
        if k == 1:
            rs_min = min(lev['dots']['radii'][x][0] for x in ('weak', 'mid', 'strong'))
            D2 = DV.dot_decode(img, B['binary'], 1, [rs_min], rules['dots'], 'enc')
            m, arr = DV.evaluate(sid, 'point', DV.dot_dec(D2, 'single'), ctx, mesh); ev['single_common'] = (m, arr)
    else:
        inp = dict(binary=B['binary'], peak=float(row['peak_f32']), pattern=pattern, meta_full=B['meta_full'])
        ctx = dict(binary=B['binary'], centres=None, pattern=pattern, peak=float(row['peak_f32']), size_mm=None, meta_full=B['meta_full'], row=dict(radius_mm=None))
        if row['pattern'] == 'plain':
            import rdy_decode as DC
            out = DC.dec_binding(img, dict(inp, centres=None), 'P4', 'P4', None)
            m, arr = DV.evaluate(sid, 'line', out['arms']['P4'], ctx, mesh); ev['P4'] = (m, arr)
            import lc_pipeline as LC
            mid = lev['lines']['widths']['mid']; Pm = DC.lc_params()
            Pm = dict(Pm, LC=dict(Pm['LC'], class_widths_px=dict(narrow=mid[0] / K.NOMINAL_MM_PER_PX, wide=mid[1] / K.NOMINAL_MM_PER_PX)))
            front = LC.shared_front_end(img, B['meta_full']['olines'], P.s2().line_cfg(float(row['peak_f32'])), pattern, Pm['LC'])
            rec = LC.finish(front, front['cand'], dict(class_mode='ignore'))
            dec = dict(pts=np.asarray(rec['points'], np.float64).reshape(-1, 3), accepted=np.asarray(rec['accepted'], bool), aux=dict(r=rec))
            m, arr = DV.evaluate(sid, 'line', dec, ctx, mesh); ev['LC_ignore_frontend'] = (m, arr)
        else:
            Lr = DV.lines_decode(img, inp); rule = {'use': 'LC_use', 'R': 'LC_R'}.get(rules['lines'], rules['lines'])
            for arm, rec in (('LC_ignore', Lr['recs']['LC_ignore']), ('LC_rule', Lr['recs'][rule])):
                dec = dict(pts=np.asarray(rec['points'], np.float64).reshape(-1, 3), accepted=np.asarray(rec['accepted'], bool), aux=dict(r=rec))
                m, arr = DV.evaluate(sid, 'line', dec, ctx, mesh); ev[arm] = (m, arr)
    for arm, (m, arr) in ev.items():
        ep = root / 'eval' / f'{ident}__{arm}.json'
        h = K.save_npz(ep.with_suffix('.npz'), **{kk: vv for kk, vv in arr.items() if isinstance(vv, np.ndarray)})
        K.save_json(ep, dict(identity=ident, arm=arm, rules=rules, metrics=m, state=m['state'], npz_sha256=h, complete=True, t_local=K.now_local()))
    return dict(key=ident, arms=sorted(ev))


def cmd_run(root=FORMAL, workers=1, max_units=None, renderer=render_binding, cap_extra=16, only=None) -> int:
    with Lock(root / 'run' / 'run.lock'):
        p = root / 'permit.json'
        if not p.exists():
            raise K.Stop('permit', 'no permit (python -B s3c_formal.py permit --task-file <new task book>)')
        fz = verify_freeze(root)
        M = pd.read_csv(root / 'manifest.csv', low_memory=False)
        if only is not None:
            M = M[M.image_key.isin(only)]
        done = [r for r in M.to_dict('records') if all(_complete(root / 'eval' / f"{r['image_key']}__{a}.json", r['image_key']) for a in str(r['arms']).split(';'))]
        todo = [r for r in M.to_dict('records') if r['image_key'] not in {d['image_key'] for d in done}]
        led = Ledger(root, cap_render=len(M) + cap_extra); n = 0
        for r in todo:
            if max_units is not None and n >= max_units:
                led.add(kind='run', event='interrupted', done=n); return EXIT['interrupted']
            try:
                run_unit(r, root, fz['rules'], led, renderer); n += 1
            except K.Stop:
                raise
            except Exception as e:
                led.add(kind='unit', event='failed', key=r['image_key'], error=repr(e)[:300]); raise K.Stop('technical', f"{r['image_key']}: {traceback.format_exception_only(type(e), e)[-1][:200]}")
        led.add(kind='run', event='finished', done=n)
    return 0


def drill() -> dict:
    import shutil
    root = FORMAL / 'drill'
    if root.exists():
        shutil.rmtree(K.guard(root))
    root.mkdir(parents=True)
    M = pd.read_csv(FORMAL / 'manifest.csv', low_memory=False)
    sub = pd.concat([M[(M.family == f) & (M.sigma == s) & (M.surface == 'M24_curv') & (M.phase_id == 'N0') & (M.rep == 0) & (M.layout == M[M.family == f].layout.iloc[0])]
                     for f in ('dots', 'lines') for s in (0.0, 40.0)]).reset_index(drop=True)
    K.save_csv(sub, root / 'manifest.csv'); out = {}
    write_permit('drill', root, drill=True); write_freeze(dict(dots='DF', lines='R'), root)
    rc = run_guarded(lambda: cmd_run(root, max_units=3, renderer=render_stub)); n1 = Ledger(root, 0).n_render()
    rc2 = run_guarded(lambda: cmd_run(root, renderer=render_stub)); n2 = Ledger(root, 0).n_render()
    out['resume'] = dict(first_exit=rc, renders_after_first=n1, second_exit=rc2, renders_total=n2, n_images=len(sub), ok=bool(rc == EXIT['interrupted'] and rc2 == 0 and n2 == len(sub)))
    bad = sub.iloc[:1].copy(); bad['raster_key'] = 'R-broken'; K.save_csv(pd.concat([bad]), root / 'manifest_bad.csv')
    os.replace(root / 'manifest.csv', root / 'manifest_good.csv'); os.replace(root / 'manifest_bad.csv', root / 'manifest.csv'); write_freeze(dict(dots='DF', lines='R'), root)
    for f in (root / 'images').glob(f"{bad.image_key.iloc[0]}*"):
        f.unlink()
    for f in (root / 'eval').glob(f"{bad.image_key.iloc[0]}*"):
        f.unlink()
    rc3 = run_guarded(lambda: cmd_run(root, renderer=render_stub)); out['error_exit'] = dict(exit=rc3, ok=bool(rc3 != 0))
    os.replace(root / 'manifest_good.csv', root / 'manifest.csv'); write_freeze(dict(dots='DF', lines='R'), root)
    fz = K.load_json(root / 'freeze.json'); fz['code'][next(iter(fz['code']))] = '0' * 64; K.save_json(root / 'freeze.json', fz)
    rc4 = run_guarded(lambda: cmd_run(root, renderer=render_stub)); out['freeze_mismatch'] = dict(exit=rc4, ok=bool(rc4 == EXIT['freeze']))
    write_freeze(dict(dots='DF', lines='R'), root); (root / 'run' / 'run.lock').write_text('999')
    rc5 = run_guarded(lambda: cmd_run(root, renderer=render_stub)); (root / 'run' / 'run.lock').unlink(); out['lock'] = dict(exit=rc5, ok=bool(rc5 == EXIT['lock']))
    try:
        write_permit(str(K.TASK_FILE), FORMAL / 'drill_permit_probe'); out['permit_refuses_this_task_book'] = dict(ok=False)
    except K.Stop as e:
        out['permit_refuses_this_task_book'] = dict(ok=bool(e.code == 'permit'))
    out['no_m3'] = 'stub renderer copies stored S3-reduced images; render_binding never called'
    K.save_json(K.CHECKS / 'formal_drills.json', dict(out, ok=all(v.get('ok', True) for v in out.values() if isinstance(v, dict)), t_local=K.now_local()))
    return out


def run_guarded(fn) -> int:
    try:
        return fn()
    except K.Stop as e:
        return EXIT.get(e.code, 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--rules'); ap.add_argument('--task-file'); ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--max-units', type=int)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'manifest':
            print(json.dumps(K.clean(build_manifest()), indent=1)); return 0
        if a.cmd == 'show':
            M = pd.read_csv(FORMAL / 'manifest.csv', low_memory=False); print(M.groupby(['family', 'layout', 'pattern']).size().to_string()); return 0
        if a.cmd == 'freeze':
            print(json.dumps(K.clean(write_freeze(json.loads(a.rules))), indent=1)[:800]); return 0
        if a.cmd == 'permit':
            print(write_permit(a.task_file)); return 0
        if a.cmd == 'run':
            return cmd_run(FORMAL, a.workers, a.max_units)
        if a.cmd == 'status':
            M = pd.read_csv(FORMAL / 'manifest.csv', low_memory=False)
            n = sum(all(_complete(FORMAL / 'eval' / f"{r['image_key']}__{x}.json", r['image_key']) for x in str(r['arms']).split(';')) for r in M.to_dict('records'))
            print(dict(images=len(M), complete=n)); return 0
        if a.cmd == 'drill':
            r = drill(); print(json.dumps(K.clean(r), indent=1)); return 0 if all(v.get('ok', True) for v in r.values() if isinstance(v, dict)) else EXIT['check']
        raise K.Stop('technical', f'unknown command {a.cmd}')
    except K.Stop as e:
        print(e, file=sys.stderr); return EXIT.get(e.code, 2)


if __name__ == '__main__':
    sys.exit(main())
