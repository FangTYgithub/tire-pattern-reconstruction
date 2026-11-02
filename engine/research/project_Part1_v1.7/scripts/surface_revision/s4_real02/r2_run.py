# Render layout cases, decode their patterns and evaluate reconstructed points.

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd

import r2_common as K

_REG = None


def registry() -> dict:
    global _REG
    if _REG is None:
        _REG = K.load_json(K.OUT / 'layouts.json')
    return _REG


def manifest() -> pd.DataFrame:
    return pd.read_csv(K.OUT / 'manifest.csv', keep_default_na=False, low_memory=False, float_precision='round_trip')


def _rowlike(r: dict) -> dict:
    import sp2_common as S2
    return dict(section='S4', family=r['family'], arm=r['arm'], radius_mm=(float(S2.R_MM) if r['family'] == 'point' else ''), strategy=r['strategy'])


def inputs_of(r: dict) -> dict:
    import r2_layouts as L
    lay = registry()['baselines'][r['baseline']]['layouts'][r['layout']]
    inp = L.build_input(r['baseline'], lay, r['phase'], r['layout'] == 'regular')
    if inp['raster_key'] != r['raster_key'] or float(np.float32(inp['peak'])) != float(r['peak_f32']):
        raise K.Stop('check', f"{r['image_key']}: adapter inputs differ from the manifest")
    return inp


def stage_image(r: dict, inp: dict, allowed: bool, timing: dict):
    import pb_m3 as PM
    import rdy_platform as P
    ip = K.IMG / f"{r['image_key']}.json"
    if K.complete(ip, r['image_key']):
        return K.load_npz(ip.with_suffix('.npz'))['image'], 'stored'
    if not allowed:
        raise K.Stop('budget', f"{r['image_key']}: render needed but no budget unit reserved")
    t0 = time.monotonic(); bd = P.bound()
    res = PM.render(bd.sid(r['surface']), inp['pattern'], float(inp['peak']), float(r['sigma']), int(r['render_seed']), K.RK.PSF_PX, K.BOUNDARY)
    img = np.asarray(res['image'])
    if img.dtype != np.uint8 or img.shape != (512, 512):
        raise K.Stop('technical', f'render returned {img.dtype} {img.shape}')
    timing['render_s'] = round(time.monotonic() - t0, 3)
    h = K.save_npz(ip.with_suffix('.npz'), image=img)
    K.save_json(ip, dict(identity=r['image_key'], baseline=r['baseline'], family=r['family'], surface=r['surface'], layout=r['layout'], phase=r['phase'], sigma=float(r['sigma']),
                         noise_seed=r['noise_seed'], render_seed=int(r['render_seed']), raster_key=inp['raster_key'], peak=float(inp['peak']), lit_px=inp['lit_px'], integral_f32=inp['integral_f32'],
                         boundary=K.BOUNDARY, image_code=r['image_code'], render_meta=K.clean(res['meta']), seconds=timing['render_s'], npz_sha256=h, complete=True, t_local=K.now_local()))
    return img, 'rendered'


def stage_decode(r: dict, img, inp: dict, timing: dict) -> dict:
    import rdy_decode as DC
    rp = K.RAW / f"{r['decode_key']}.json"
    if K.complete(rp, r['decode_key']):
        return DC.unpack(K.load_npz(rp.with_suffix('.npz')), K.load_json(rp)['meta'])
    t0 = time.monotonic(); out = DC.decode(r['method'], img, inp, _rowlike(r))
    if sorted(out['arms']) != [r['arm']]:
        raise K.Stop('technical', f"{r['method']}: arms {sorted(out['arms'])} != {r['arm']}")
    timing['decode_s'] = round(time.monotonic() - t0, 3)
    A, meta = DC.pack(out); h = K.save_npz(rp.with_suffix('.npz'), **A)
    K.save_json(rp, dict(identity=r['decode_key'], image_key=r['image_key'], method=r['method'], arm=r['arm'], decode_code=r['decode_code'], meta=meta, seconds=timing['decode_s'], npz_sha256=h,
                         complete=True, t_local=K.now_local()))
    return DC.unpack(A, meta)


def stage_equiv(r: dict, img, inp: dict, out: dict, timing: dict) -> dict:

    import rdy_decode as DC
    import r2_eval as E
    ep = K.EQV / f"{r['image_key']}.json"; ident = K.digest(dict(img=r['image_key'], dec=r['decode_code'], ev=r['eval_code'], kind='S0 table/ordinary'), 24)
    if ep.exists():
        x = K.load_json(ep)
        if x.get('identity') == ident and x.get('complete'):
            return x
    o = inp['ordinary']
    if o is None or not o['raster_equal']:
        x = dict(identity=ident, image_key=r['image_key'], equal=False, reason='ordinary raster differs', complete=True); K.save_json(ep, x); return x
    oinp = dict(inp, binary=o['binary'], centres=o['centres'], meta_full=o['meta_full'], pattern=(np.asarray(o['binary']) * np.float32(inp['peak'])).astype(np.float32), table=None)
    fam = r['family']; t0 = time.monotonic()
    import sp2_common as S2
    od = DC.dec_binding(img, oinp, 'D0' if fam == 'point' else 'P4', 'ordinary', float(S2.R_MM) if fam == 'point' else None)
    timing['equiv_decode_s'] = round(time.monotonic() - t0, 3)
    a = out['arms'][r['arm']]; b = od['arms']['ordinary']; cmp = {}
    if fam == 'point':
        cmp['feats'] = bool(np.array_equal(np.asarray(a['aux']['feats']), np.asarray(b['aux']['feats'])))
        for k in ('matched', 'ambiguous', 'duplicate', 'best_idx', 'point_idx'):
            cmp[f'tr_{k}'] = bool(np.array_equal(np.asarray(a['aux']['tr'][k]), np.asarray(b['aux']['tr'][k])))
    else:
        ra, rb = a['aux']['r'], b['aux']['r']
        for k in ('features', 'status', 'line_id', 'segment_id', 'chain', 'n_candidates', 'guard_reason'):
            cmp[k] = bool(np.array_equal(np.asarray(ra[k]), np.asarray(rb[k])))
    cmp['accepted'] = bool(np.array_equal(a['accepted'], b['accepted']))
    pa, pb = np.asarray(a['pts'], np.float64), np.asarray(b['pts'], np.float64)
    if pa.shape != pb.shape:
        dmax = float('inf'); same_nan = False
    else:
        same_nan = bool(np.array_equal(np.isnan(pa), np.isnan(pb))); both = np.isfinite(pa) & np.isfinite(pb); dmax = float(np.abs(pa - pb)[both].max()) if both.any() else 0.0
    cmp['points_le_1e-6mm'] = bool(same_nan and dmax <= 1e-6)
    ma, _, _ = E.evaluate(fam, r['surface'], a, inp, K.BOUNDARY, E.domain('omega1'))
    mb, _, _ = E.evaluate(fam, r['surface'], b, dict(oinp, table=inp['table']) if fam == 'line' else oinp, K.BOUNDARY, E.domain('omega1'))
    diffs = {}
    for k in sorted(set(ma) & set(mb)):
        va, vb = ma[k], mb[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            tol = 0 if (float(va).is_integer() and float(vb).is_integer() and k.startswith(('n_', 'id_', 'img_n', 'loc_n', 'glob_n'))) else 1e-9
            if abs(float(va) - float(vb)) > tol:
                diffs[k] = [va, vb]
        elif va != vb and not (va is None and vb is None):
            diffs[k] = [va, vb]
    x = dict(identity=ident, image_key=r['image_key'], family=fam, sigma=float(r['sigma']), compare=cmp, max_abs_point_diff_mm=dmax, evaluation_differences=diffs,
             equal=bool(all(cmp.values()) and not diffs), complete=True, t_local=K.now_local())
    K.save_json(ep, x)
    return x


def stage_eval(r: dict, img, inp: dict, out: dict, timing: dict) -> str:
    import r2_eval as E
    import rdy_eval as RE
    import rdy_platform as P
    ep = K.EVL / f"{r['eval_key']}.json"
    if K.complete(ep, r['eval_key']):
        return 'stored'
    dec = out['arms'][r['arm']]; t0 = time.monotonic()
    m1, arr, info = E.evaluate(r['family'], r['surface'], dec, inp, K.BOUNDARY, E.domain('omega1'))
    ms, _a, _i = E.evaluate(r['family'], r['surface'], dec, inp, K.BOUNDARY, E.domain('omega_star'))
    extras = dict(displacement=RE.displacement(inp), nodes_omega1=E.node_counts(arr['pts'], arr['valid'], r['surface'], E.domain('omega1')))
    if r['family'] == 'point':
        if float(r['sigma']) == 0.0:
            ref = img; role = 'self'
        else:
            rp = K.IMG / f"{r['sigma0_ref_key']}.json"
            if not K.complete(rp, r['sigma0_ref_key']):
                raise K.Stop('dependency', f"sigma-0 reference {r['sigma0_ref_key']} of {r['image_key']} missing (must be complete first)")
            ref = K.load_npz(rp.with_suffix('.npz'))['image']; role = 'sigma0_reference'
        si, code = RE.source_identity(img, ref, inp, dec, P.mesh(r['surface']))
        extras['source_identity'] = dict(si, source=role, ref_key=r['sigma0_ref_key']); arr['src_ident_code'] = code
    timing['eval_s'] = round(time.monotonic() - t0, 3)
    h = K.save_npz(ep.with_suffix('.npz'), **arr)
    K.save_json(ep, dict(identity=r['eval_key'], decode_key=r['decode_key'], image_key=r['image_key'], family=r['family'], surface=r['surface'], sigma=float(r['sigma']), arm=r['arm'],
                         boundary=K.BOUNDARY, domain_main=dict(name='omega1', hash=E.domain('omega1')['hash']), domain_bridge=dict(name='omega_star', hash=E.domain('omega_star')['hash']),
                         eval_code=r['eval_code'], metrics=m1, metrics_omega_star={k: v for k, v in ms.items() if k.startswith('glob_') or k in ('state', 'n_valid')}, state=m1['state'],
                         extras=extras, info=info, seconds=timing['eval_s'], npz_sha256=h, complete=True, t_local=K.now_local()))
    return 'evaluated'


def unit(args) -> dict:
    r, allowed = args
    timing = {}; t0 = time.monotonic()
    inp = inputs_of(r)
    img, src = stage_image(r, inp, allowed, timing)
    out = stage_decode(r, img, inp, timing)
    eq = stage_equiv(r, img, inp, out, timing) if r['layout'] == 'regular' else None
    if eq is not None and not eq['equal']:
        raise K.Stop('check', f"S0 table / ordinary equivalence failed for {r['image_key']}: {eq.get('compare')} {list(eq.get('evaluation_differences', {}))[:6]}")
    ev = stage_eval(r, img, inp, out, timing)
    timing['total_s'] = round(time.monotonic() - t0, 3)
    return dict(image_key=r['image_key'], image=src, eval=ev, equiv=(None if eq is None else eq['equal']), timing=timing, pid=os.getpid())


def _worker(args):
    try:
        return unit(args)
    except Exception as e:
        return dict(image_key=args[0]['image_key'], error=f'{type(e).__name__}: {e}', tb=traceback.format_exc()[-3000:])


def done(r: dict) -> bool:
    ok = K.complete(K.IMG / f"{r['image_key']}.json", r['image_key']) and K.complete(K.RAW / f"{r['decode_key']}.json", r['decode_key']) and K.complete(K.EVL / f"{r['eval_key']}.json", r['eval_key'])
    if ok and r['layout'] == 'regular':
        p = K.EQV / f"{r['image_key']}.json"; ok = p.exists() and K.load_json(p).get('equal') is True
    return ok


def schedule(keys: list | None, workers: int, label: str) -> dict:

    M = manifest(); U = M.drop_duplicates('image_key').set_index('image_key', drop=False)
    keys = list(U.index) if keys is None else list(keys)
    rows = [U.loc[k].to_dict() for k in keys]
    waves = [sorted([r for r in rows if float(r['sigma']) == 0], key=lambda r: r['image_key']), sorted([r for r in rows if float(r['sigma']) > 0], key=lambda r: r['image_key'])]
    B = K.Budget(); log = K.RUNDIR / f'ledger_{label}.jsonl'; t0 = time.time(); res = []; fails = []
    def rec(x):
        K.guard(log); log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(K.clean(dict(x, t=K.now_local())), ensure_ascii=False) + '\n')
    for wave in waves:
        todo = [r for r in wave if not done(r)]
        if not todo:
            continue
        with ProcessPoolExecutor(max_workers=workers) as ex:
            it = iter(todo); running = {}
            def submit():
                r = next(it, None)
                if r is None:
                    return False
                need = not K.complete(K.IMG / f"{r['image_key']}.json", r['image_key'])
                n = B.reserve(r['image_key'], label) if need else None
                fut = ex.submit(_worker, (r, need)); running[fut] = (r['image_key'], n); return True
            for _ in range(2 * workers):
                if fails or not submit():
                    break
            while running:
                fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for f in fin:
                    k, n = running.pop(f); x = f.result(); x['budget_n'] = n; rec(x); res.append(x)
                    if 'error' in x:
                        fails.append(x); B.log(dict(n=n, key=k, purpose=label, event='failed', error=x['error'][:300]))
                    elif n is not None:
                        B.log(dict(n=n, key=k, purpose=label, event='ok'))
                    if not fails:
                        submit()
        if fails:
            break
    st = dict(label=label, n_requested=len(rows), n_processed=len(res), n_failed=len(fails), seconds=round(time.time() - t0, 1), workers=workers, budget=B.state(),
              renders=sum(1 for x in res if x.get('image') == 'rendered'), t_local=K.now_local())
    K.save_json(K.RUNDIR / f'status_{label}.json', st)
    if fails:
        raise K.Stop('technical', f"{len(fails)} unit failure(s); first {fails[0]['image_key']}: {fails[0]['error'][:300]}")
    return st


def legacy_refs() -> dict:
    import rdy_platform as P
    M = manifest(); S0 = M[(M.layout == 'regular') & (M.phase == 'H0') & M.sigma.isin([0.0, 100.0])].drop_duplicates('image_key')
    B = K.Budget(); out = []
    for r in S0.to_dict('records'):
        lk = 'R2L-' + K.digest(dict(formal=r['image_key'], boundary='legacy', entry='frozen binding render (adp_surf02 -> shared m3_virtual_camera)'), 24)
        lp = K.OUT / 'legacy_ref' / f'{lk}.json'
        if not K.complete(lp, lk):
            inp = inputs_of(r); n = B.reserve(lk, 'legacy_reference'); bd = P.bound(); t0 = time.monotonic()
            try:
                res = bd.call('render', bd.sid(r['surface']), inp['pattern'], float(inp['peak']), float(r['sigma']), int(r['render_seed']), K.RK.PSF_PX)
            except Exception as e:
                B.log(dict(n=n, key=lk, purpose='legacy_reference', event='failed', error=repr(e)[:300])); raise
            img = np.asarray(res['image']); h = K.save_npz(lp.with_suffix('.npz'), image=img)
            K.save_json(lp, dict(identity=lk, formal_image_key=r['image_key'], family=r['family'], surface=r['surface'], sigma=float(r['sigma']), render_seed=int(r['render_seed']), boundary='legacy',
                                 entry='frozen binding render', render_meta=K.clean(res.get('meta', {})), seconds=round(time.monotonic() - t0, 3), npz_sha256=h, complete=True, t_local=K.now_local()))
            B.log(dict(n=n, key=lk, purpose='legacy_reference', event='ok'))
        out.append(dict(legacy_key=lk, formal_image_key=r['image_key'], family=r['family'], surface=r['surface'], sigma=float(r['sigma'])))
    return dict(n=len(out), refs=out)
