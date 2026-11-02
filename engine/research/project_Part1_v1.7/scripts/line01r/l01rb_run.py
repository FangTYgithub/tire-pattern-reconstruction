# Run line cases and compare measured, numbered and truth-assigned reconstructions.

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l01rb_common as B

CASES = Path(os.environ.get('LINE01RB_CASES_DIR', str(B.CASES)))
WHICH = os.environ.get('LINE01RB_PROGRESS', 'main')
INJECT_ERROR = os.environ.get('LINE01RB_INJECT_ERROR', '')
INJECT_BLANK = os.environ.get('LINE01RB_INJECT_BLANK', '')
OLD_IMAGES = os.environ.get('LINE01RB_OLD_IMAGES', '') == '1'
FREEZE_JSON = os.environ.get('LINE01RB_FREEZE_JSON', '')
_FREEZE_OK = {}
_S: dict = {}


def surf(name):
    if name not in _S:
        mesh, cfg, ref_fn, desc = B.surface(name); om = B.omega(name); gt, zmin = B.grid_truth(mesh)
        Z = np.asarray(mesh['Z'], np.float64); Zb = np.asarray(mesh['Z_base'], np.float64); idx = om['idx']
        _S[name] = dict(mesh=mesh, cfg=cfg, ref_fn=ref_fn, grid=gt, has_dent=bool(cfg.get('has_indentation')), base_rel=(Zb.ravel()[idx] - Z.min()), dip_true=(Zb - Z).ravel()[idx],
                        domain=dict(x=om['x'], y=om['y'], z_ref=om['z_ref'], area=om['area'], hash=om['hash']), masks=dict(dent=om['in_dent'], fov_legacy=om['in_fov_legacy']))
    return _S[name]


def compact_r3(ev):
    out = dict(version=ev['version'], triangulation=ev['triangulation'], n_triangles=ev['n_triangles'], n_used=ev['n_used'], sub={})
    for dom, sub in ev['subdomains'].items():
        out['sub'][dom] = dict(C_interpolable=sub['C_interpolable'], nn_p95=sub['nn_dist']['p95'], nn_max=sub['nn_dist']['max'],
                               by_d={d: dict(C_valid=b['C_valid'], rmse=b['rmse'], p95=b['p95'], bias=b['bias'], C_tau=b['C_tau']) for d, b in sub['by_d'].items()})
    return out


def evaluate(points, acc, S):

    from surface_eval_r3 import surface_eval_r3_dev, point_metrics, deformation_quantiles
    from m5_evaluation import compute_indentation_recovery
    pts = np.asarray(points, np.float64); acc = np.asarray(acc, bool)
    ev = surface_eval_r3_dev(pts, acc, S['domain'], B.D_LIST, B.TAU_LIST, masks=S['masks'], support_rule='exact', return_fields=True)
    fields = ev.pop('fields')
    fin = acc & np.isfinite(pts).all(axis=1)
    zr = np.full(len(pts), np.nan)
    if fin.any():
        zr[fin] = S['ref_fn'](pts[fin, 0], pts[fin, 1])
    out = dict(r3=compact_r3(ev), **point_metrics(pts, acc, zr, prefix='pts_ref'), n_accepted=int(acc.sum()))
    out['deformation'] = deformation_quantiles(fields, S['domain'], S['masks']['dent'], S['dip_true'], S['base_rel'], B.D_LIST)
    if S['has_dent']:
        ind = compute_indentation_recovery(pts.astype(np.float32), S['mesh'], S['cfg'], ambiguous_mask=~acc)
        e = ind.get('ind_depth_err_mm'); d_gt = ind.get('ind_depth_gt_mm')
        out['amplitude'] = dict(rel_err_pct=(100.0 * e / d_gt) if (e is not None and np.isfinite(e) and d_gt) else None, err_mm=(e if (e is not None and np.isfinite(e)) else None), depth_rec_mm=ind.get('ind_depth_rec_mm'),
                                depth_rec_pts_mm=ind.get('ind_depth_rec_pts_mm'), coverage=ind.get('ind_coverage_frac'), center_err_mm=ind.get('ind_center_err_mm'), n_pts_in_contour=ind.get('ind_n_pts_in_contour'), definition='M5 compute_indentation_recovery (interpolated peak, FOV window)')
    return B.clean(out)


def row_flags(ref, feats, ids, acc, gate=B.GATE_PX):
    Rr = ref['rows']; n = len(Rr['v'])
    by_row: dict = {}
    for i in np.flatnonzero(acc):
        by_row.setdefault(int(round(feats[i, 1])), []).append(i)
    rec = np.zeros(n, bool); ident = np.zeros(n, bool)
    for j in range(n):
        cand = by_row.get(int(Rr['v'][j]), [])
        if not cand:
            continue
        du = np.array([feats[i, 0] - Rr['u'][j] for i in cand]); near = np.abs(du) <= gate
        if near.any():
            rec[j] = True
            if (ids[np.array(cand)[near]] == Rr['k'][j]).any():
                ident[j] = True
    return rec, ident


def gaps_per_line(ref, rec):
    Rr = ref['rows']; n_gaps = 0; gap_rows = 0
    for k in np.unique(Rr['k']):
        sel = np.flatnonzero(Rr['k'] == k); o = sel[np.argsort(Rr['v'][sel])]; r = rec[o]
        if r.sum() < 2:
            continue
        first, last = np.flatnonzero(r)[0], np.flatnonzero(r)[-1]
        miss = ~r[first:last + 1]
        d = np.diff(np.concatenate([[0], miss.astype(int), [0]]))
        n_gaps += int((d == 1).sum()); gap_rows += int(miss.sum())
    return n_gaps, gap_rows


def numbering_layer(r, kO, conf):

    nc = np.asarray(r['n_candidates']); adm = np.asarray(r['admissible'], bool); ids = np.asarray(r['candidate_ids']); st = np.asarray(r['status']); seg = np.asarray(r['segment_id']); lid = np.asarray(r['line_id'])
    hist = {str(k): int((nc == k).sum()) for k in (0, 1, 2)}; hist['3+'] = int((nc >= 3).sum())
    multi = np.flatnonzero(nc >= 2); n_adj = 0; n_par = 0; n_k2 = 0; lists = []
    for i in multi:
        c = np.sort(ids[adm[i]]); lists.append(c)
        n_adj += int(c.max() - c.min() == len(c) - 1)
        if len(c) == 2:
            n_k2 += 1; n_par += int((c[1] - c[0]) % 2 == 1)
    acc = st == 0; wrong = acc & conf & (lid != kO)
    wseg = np.unique(seg[wrong]) if wrong.any() else np.zeros(0, np.int64)
    runs = [int((wrong & (seg == s)).sum()) for s in wseg]
    kmax = int(nc.max()) if len(nc) else 0
    multi_ids = np.full((len(multi), max(kmax, 1)), B.A.C0 and -1_000_000, np.int64)
    for j, c in enumerate(lists):
        multi_ids[j, :len(c)] = c
    truth_adm = None
    if conf.any():
        col = {int(k): j for j, k in enumerate(ids)}
        ok = [adm[i, col[int(kO[i])]] if int(kO[i]) in col else False for i in np.flatnonzero(conf)]
        truth_adm = float(np.mean(ok)) if ok else None
    segs = r['segments']
    return dict(K_hist=hist, K_max=kmax, n_multi=int(len(multi)), frac_multi=(float(len(multi) / len(nc)) if len(nc) else None), multi_adjacent_frac=(n_adj / len(multi) if len(multi) else None),
                K2_parity_different_frac=(n_par / n_k2 if n_k2 else None), n_K2=n_k2, n_unknown_features=int((st == 2).sum()), n_inconsistent=int((st == 3).sum()), n_no_candidate=int((st == 1).sum()),
                segments=dict(n_total=len(segs), numbered=sum(1 for s in segs if s['decision'] == 'numbered'), unknown=sum(1 for s in segs if s['decision'] == 'unknown'), no_candidate=sum(1 for s in segs if s['decision'] == 'no_candidate')),
                wrong=dict(n_features=int(wrong.sum()), n_segments=int(len(wseg)), max_run=(max(runs) if runs else 0)), truth_id_admissible_frac=truth_adm), dict(multi_idx=multi.astype(np.int64), multi_ids=multi_ids)


def oracle_layer(r, kO, conf, fam, cfg, centrelines):

    from m4_lines import line_candidates, reconstruct_numbered
    feat = r['features']; st = np.asarray(r['status'])
    cand = line_candidates(feat, fam, cfg, centrelines=centrelines); col = {int(k): j for j, k in enumerate(cand['ids'])}
    stO = np.full(len(feat), 1, np.int64)
    for i in np.flatnonzero(conf):
        j = col.get(int(kO[i]))
        if j is not None and cand['admissible'][i, j]:
            stO[i] = 0
        elif j is not None:
            stO[i] = 3
    guard = st >= 8
    stO = np.where(guard, st, stO)
    ptsO, projO, stO, chO = reconstruct_numbered(feat, kO, stO, cand, fam, cfg)
    guardO = stO >= 8
    cons = dict(features_same=True, n_features=int(len(feat)), guard_mask_same=bool(np.array_equal(guard, guardO)), guard_reasons_same=bool(np.array_equal(st[guard], stO[guardO])) if guard.any() else True,
                n_guard_P=int(guard.sum()), n_guard_O=int(guardO.sum()), admissible_same=bool(np.array_equal(cand['admissible'], np.asarray(r['admissible'], bool))),
                window_same=bool(abs(cand['window_z0_mm'] - r['window_z0_mm']) < 1e-12 and abs(cand['window_z1_mm'] - r['window_z1_mm']) < 1e-12), O_accepts_only_admissible_truth=True)
    cons['all_ok'] = all(v is True for k, v in cons.items() if k.endswith('same') or k.startswith('O_'))
    return stO, ptsO, cons


def load_image_for(spec, token=None):

    if OLD_IMAGES:
        rec0, z0 = B.load_old_case(spec['old_case_id']); img = np.asarray(z0['image'])
        return img, dict(source=B.rel(B.A.OLD_CASES / f"{spec['old_case_id']}.npz"), n_valid_pixels=int((img > 0).sum()), seconds=0.0, attempt=0, rendered=False)
    if token is None:
        raise RuntimeError('render without a reservation token: every research render must be reserved by the parent scheduler')
    S = surf(spec['surface']); pat_bin, meta = B.pattern_for(spec); pat = (pat_bin * np.float32(spec['peak'])).astype(np.float32)
    prev = [x for x in B.ledger_records('research') if x.get('case_id') == spec['case_id']]
    t0 = time.monotonic()
    img, info = B.render_research(S['mesh'], pat, B.m3_cfg(float(spec['sigma']), int(spec['noise_seed'])), spec['case_id'], purpose=token.get('purpose', 'B_research'), attempt=len(prev) + 1, token=token.get('id'))
    return img, dict(source='render', n_valid_pixels=int(info['n_valid_pixels']), n_hits=int(info['n_hits']), seconds=round(time.monotonic() - t0, 3), attempt=len(prev) + 1, rendered=True, reservation=token.get('id'),
                     m3=dict(noise_sigma=float(spec['sigma']), mottle_seed=int(spec['noise_seed']), psf_sigma_px=B.PSF_PX, brightness=float(info.get('brightness', 255)), surface_z_range_mm=list(info.get('surface_z_range_mm', (None, None)))))


def _peak_rss_mb():
    try:
        sys.path.insert(0, str(B.ROOT / 'scripts' / 'spotnoise')); import sn_lib
        return float(sn_lib.peak_rss_mb())
    except Exception:
        return None


def worker_config(paths):

    p4 = paths.get('P4_C3_cal') or {}
    coef = p4.get('steger_noise_coef')
    return dict(pid=os.getpid(), render_cap=B.render_cap(), ledger_dir=str(B.ledger_dir()), out_dir=str(B.out_dir()), cases_dir=str(CASES), paths=sorted(paths), paths_json=os.environ.get('LINE01RB_PATHS_JSON'),
                steger_noise_coef_id=(p4.get('steger_noise_coef_id') or ('registered_fp1e-4' if coef is None else 'custom')), steger_noise_coef={str(k): float(v) for k, v in coef.items()} if isinstance(coef, dict) else None,
                window=B.window_cfg(), strict_npz=os.environ.get('LINE01RB_STRICT_NPZ') == '1', old_images=OLD_IMAGES, src_versions=B.src_versions())


def reference_and_G(spec, S, meta, pat):

    import l01_truth as T
    d = B.out_dir() / 'g_cache'; d.mkdir(parents=True, exist_ok=True)
    sv = B.src_versions(); sref = B.surface_record(spec['surface'])
    key = B.sha_bytes(json.dumps(dict(surface=sref['reference'], omega=sref['omega_hash'], raster=spec['raster_sha256'], inner_margin='l01_truth.inner_margin_px default', truth_sha=sv['scripts/line01/l01_truth.py'], eval_key=B.eval_key()), sort_keys=True).encode())[:16]
    pj, pz = d / f'{key}.json', d / f'{key}.npz'
    if pj.exists() and pz.exists():
        try:
            g = B.load_json(pj)
            with np.load(pz) as z:
                rows = dict(k=np.asarray(z['k']), v=np.asarray(z['v']), u=np.asarray(z['u']), xyz=np.asarray(z['xyz']), t=np.asarray(z['t']), w_arc=np.asarray(z['w_arc']), inner=np.asarray(z['inner']).astype(bool)); g_xyz = np.asarray(z['g_xyz'])
            ref = dict(rows=rows, g_xyz=g_xyz, n_lines=g['n_lines'], centre_source=g.get('centre_source'), inner_margin_px=g.get('inner_margin_px'))
            return ref, dict(g['G'], cache_key=key, cache_hit=True), key
        except Exception:
            pass
    ref = T.line_reference(S['mesh'], meta['olines'], 1.0, pattern=pat); Rr = ref['rows']
    G = evaluate(ref['g_xyz'], np.ones(len(ref['g_xyz']), bool), S)
    G['n_points'] = int(len(ref['g_xyz'])); G['definition'] = 'ideal samples of the true surface at the rasterised line centres inside the camera image (l01_truth.line_reference g_xyz)'


    tz = pz.with_name(f'{key}.{os.getpid()}.npz.tmp'); tj = pj.with_name(f'{key}.{os.getpid()}.json.tmp')
    try:
        with open(tz, 'wb') as fh:
            np.savez_compressed(fh, k=Rr['k'].astype(np.int32), v=Rr['v'].astype(np.int32), u=Rr['u'].astype(np.float32), xyz=np.asarray(Rr['xyz'], np.float32), t=np.asarray(Rr['t'], np.float32), w_arc=np.asarray(Rr['w_arc'], np.float32), inner=Rr['inner'], g_xyz=np.asarray(ref['g_xyz'], np.float32))
        tj.write_text(json.dumps(B.clean(dict(key=key, surface=spec['surface'], raster_sha256=spec['raster_sha256'], w_mm=spec['w_mm'], s_mm=spec['s_mm'], phi_deg=spec['phi_deg'], phase_frac=spec['phase_frac'], n_lines=ref['n_lines'], centre_source=ref.get('centre_source'), inner_margin_px=ref.get('inner_margin_px'), G=G, created_local=B.now_local())), ensure_ascii=False, indent=1), encoding='utf-8')
        for tmp, dst in ((tz, pz), (tj, pj)):
            if dst.exists():
                tmp.unlink(missing_ok=True)
            else:
                try:
                    os.replace(tmp, dst)
                except (PermissionError, FileExistsError, OSError):
                    tmp.unlink(missing_ok=True)
    except Exception:
        for tmp in (tz, tj):
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
    return ref, dict(G, cache_key=key, cache_hit=False), key


def decode_paths(img, spec, meta, pat, S, ref, paths):
    import l01_truth as T
    from m4_lines import run_olines, line_family, raster_centrelines, STATUS
    cfg = B.m4_cfg(spec['peak']); fam = line_family(meta['olines']); cl = raster_centrelines(pat, fam); Rr = ref['rows']
    out = {}; arrays = {}
    for name, pp in paths.items():
        t1 = time.monotonic()
        r = run_olines(img, meta['olines'], cfg, pattern=pat, **pp)
        win = B.check_window(r)
        feat = r['features']; acc = r['accepted']; st = r['status']
        ident = T.truth_identity(S['mesh'], feat, meta['olines']); kO = ident['k']; conf = ident['confirmable']
        stO, ptsO, cons = oracle_layer(r, kO, conf, fam, cfg, cl); accO = stO == 0
        t_dec = time.monotonic() - t1; t2 = time.monotonic()
        lmP = T.line_metrics(ref, feat, r['line_id'], acc, chain=r['chain'], truth_ids=kO); lmO = T.line_metrics(ref, feat, kO, accO, chain=r['chain'], truth_ids=kO)
        recP, idP = row_flags(ref, feat, r['line_id'], acc); gapsP, gaprowsP = gaps_per_line(ref, recP)
        EP = evaluate(r['points'], acc, S); EO = evaluate(ptsO, accO, S)
        both = acc & accO; dz = r['points'][both, 2] - ptsO[both, 2] if both.any() else np.zeros(0)
        num, num_arr = numbering_layer(r, kO, conf)
        denom = dict(n_detected=int(len(feat)), n_accepted=int(acc.sum()), accepted_confirmable_correct=int((acc & conf & (r['line_id'] == kO)).sum()), accepted_confirmable_wrong=int((acc & conf & (r['line_id'] != kO)).sum()),
                     accepted_unconfirmable=int((acc & ~conf).sum()), rejected_by_reason={STATUS[k]: int((st == k).sum()) for k in STATUS if k != 0}, n_confirmable=int(conf.sum()), n_unconfirmable=int((~conf).sum()),
                     ref_rows=int(len(Rr['v'])), ref_rows_recovered=int(recP.sum()), ref_rows_missed=int((~recP).sum()), ref_rows_identity=int(idP.sum()), ref_gaps=gapsP, ref_gap_rows=gaprowsP,
                     chain_fragments_accepted=lmP.get('n_segments_accepted'), n_chains_total=r['counts']['n_chains'], n_segments_total=int(len(np.unique(r['segment_id'][r['segment_id'] >= 0]))),
                     O_accepted=int(accO.sum()), common_accepted=int(both.sum()), P_only=int((acc & ~accO).sum()), O_only=int((~acc & accO).sum()),
                     dz_P_minus_O_max_abs_mm=(float(np.abs(dz).max()) if len(dz) else None), dz_P_minus_O_rms_mm=(float(np.sqrt(np.mean(dz ** 2))) if len(dz) else None))
        out[name] = dict(params=pp, decode_key=B.decode_key(pp, spec['peak']), seconds_decode=round(t_dec, 2), seconds_eval=round(time.monotonic() - t2, 2), window=win,
                         detect_info={k: v for k, v in r['detect_info'].items() if not isinstance(v, np.ndarray)}, counts=r['counts'], checks=r['checks'], numbering=num, consistency=cons, denominators=denom,
                         P=dict(line_metrics=lmP, **EP), O=dict(line_metrics=lmO, **EO))
        arrays.update({f'{name}_features': feat, f'{name}_status': st.astype(np.int8), f'{name}_line_id': r['line_id'].astype(np.int32), f'{name}_chain': r['chain'].astype(np.int32), f'{name}_segment': np.asarray(r['segment_id']).astype(np.int32),
                       f'{name}_truth_k': kO.astype(np.int32), f'{name}_confirmable': conf, f'{name}_points': r['points'].astype(np.float32), f'{name}_statusO': stO.astype(np.int8), f'{name}_pointsO': ptsO.astype(np.float32),
                       f'{name}_run_width': (np.asarray(r['run_width_px'], np.float32) if r.get('run_width_px') is not None else np.zeros(0, np.float32)), f'{name}_n_candidates': r['n_candidates'].astype(np.int8),
                       f'{name}_guard_reason': np.asarray(r['guard_reason']).astype(np.int8), f'{name}_multi_idx': num_arr['multi_idx'], f'{name}_multi_ids': num_arr['multi_ids'].astype(np.int32)})
    return out, arrays


def reeval_paths(rec, z, S):

    out = {}
    for name, p in rec['paths'].items():
        pts = np.asarray(z[f'{name}_points'], np.float64); st = np.asarray(z[f'{name}_status']); ptsO = np.asarray(z[f'{name}_pointsO'], np.float64); stO = np.asarray(z[f'{name}_statusO'])
        EP = evaluate(pts, st == 0, S); EO = evaluate(ptsO, stO == 0, S)
        p = dict(p); p['P'] = dict(line_metrics=p['P'].get('line_metrics'), **EP); p['O'] = dict(line_metrics=p['O'].get('line_metrics'), **EO); p['reevaluated_local'] = B.now_local()
        out[name] = p
    return out


# Render and decode one line case, including reference-identity diagnostics.
def work(args):
    case_id, mode = args[0], args[1]; token = args[2] if len(args) > 2 else None
    t0 = time.monotonic()
    try:
        if FREEZE_JSON and os.getpid() not in _FREEZE_OK:
            _FREEZE_OK[os.getpid()] = B.check_freeze(FREEZE_JSON, stage=f'worker_init pid {os.getpid()}')
        if INJECT_ERROR and case_id.startswith(INJECT_ERROR):
            raise RuntimeError('injected program error (rehearsal)')
        spec = SPECS[case_id]; S = surf(spec['surface'])
        import l01_truth as T
        pat_bin, meta = B.pattern_for(spec); pat = (pat_bin * np.float32(spec['peak'])).astype(np.float32)
        paths = B.paths_for(spec)
        if mode == 'reeval':
            rec = B.load_json(CASES / f'{case_id}.json')
            with np.load(CASES / f'{case_id}.npz') as z:
                rec['paths'] = reeval_paths(rec, z, S)
            rec['eval_key'] = B.eval_key(); rec['mode_last'] = 'reeval'; rec['finished_local'] = B.now_local(); rec['seconds'] = round(time.monotonic() - t0, 2)
            rec['class'] = B.classify(rec)
            B.save_json(CASES / f'{case_id}.json', rec)
            return dict(case_id=case_id, status='ok', mode=mode, seconds=rec['seconds'], cls=rec['class'])
        if mode == 'redecode':
            rec_old = B.load_json(CASES / f'{case_id}.json')
            with np.load(CASES / f'{case_id}.npz') as z:
                img = np.asarray(z['image'])
            image = dict(rec_old['image'], source='npz_reuse', rendered=False)
        else:
            img, image = load_image_for(spec, token)
            image['sha256'] = B.sha_bytes(img.tobytes())
        if INJECT_BLANK and case_id.startswith(INJECT_BLANK):
            img = np.zeros_like(img); image = dict(image, injected_blank=True, n_valid_pixels=0)
        img_sha = B.sha_bytes(img.tobytes()); image['sha256'] = img_sha; image['image_key'] = B.sha_bytes((spec['forward_key'] + img_sha).encode())[:16]
        t_ref = time.monotonic()
        ref, G, gkey = reference_and_G(spec, S, meta, pat); Rr = ref['rows']
        resource = B.image_resource_stats(img, spec['peak'], float(spec['sigma']))
        rec = dict(case_id=case_id, spec=spec, forward_key=spec['forward_key'], image=image, resource=resource, eval_key=B.eval_key(), window_registered=B.window_cfg(),
                   reference=dict(n_rows=int(len(Rr['v'])), n_rows_inner=int(Rr['inner'].sum()), n_lines=ref['n_lines'], arc_total=float(Rr['w_arc'].sum()), n_g=int(len(ref['g_xyz'])), seconds=round(time.monotonic() - t_ref, 2), g_cache_key=gkey, g_cache_hit=G.get('cache_hit')),
                   G=G, src_versions=B.src_versions(), paths={}, started_local=B.now_local(), mode_last=mode, worker_config=worker_config(paths))
        rec['paths'], arrays = decode_paths(img, spec, meta, pat, S, ref, paths)
        rec['seconds'] = round(time.monotonic() - t0, 2); rec['finished_local'] = B.now_local(); rec['status'] = 'ok'; rec['class'] = B.classify(rec); rec['worker_peak_rss_mb'] = _peak_rss_mb()
        CASES.mkdir(parents=True, exist_ok=True)
        tmp = CASES / f'{case_id}.npz.tmp'
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, image=img, ref_k=Rr['k'].astype(np.int32), ref_v=Rr['v'].astype(np.int32), ref_u=Rr['u'].astype(np.float32), ref_inner=Rr['inner'], **arrays)
        tmp.replace(CASES / f'{case_id}.npz')
        B.save_json(CASES / f'{case_id}.json', rec)
        return dict(case_id=case_id, status='ok', mode=mode, seconds=rec['seconds'], cls=rec['class'], pid=os.getpid(), peak_rss_mb=rec['worker_peak_rss_mb'], render_s=image.get('seconds'), ref_s=rec['reference']['seconds'],
                    decode_s=sum(p['seconds_decode'] for p in rec['paths'].values()), eval_s=sum(p['seconds_eval'] for p in rec['paths'].values()), g_hit=G.get('cache_hit'))
    except Exception:
        err = traceback.format_exc()
        CASES.mkdir(parents=True, exist_ok=True)
        B.save_json(CASES / f'{case_id}.error.json', dict(case_id=case_id, status='error', mode=mode, error=err, finished_local=B.now_local()))
        return dict(case_id=case_id, status='error', mode=mode, seconds=round(time.monotonic() - t0, 2), error=err.splitlines()[-1][:200])


SPECS: dict = {}


def set_specs(specs: list[dict]):
    SPECS.clear(); SPECS.update({s['case_id']: s for s in specs})


def plan(specs, strict=False):
    ek = B.eval_key(); todo = []; states = {}
    for s in specs:
        dk = {n: B.decode_key(pp, s['peak']) for n, pp in B.paths_for(s).items()}
        st, detail = B.record_state(CASES, s['case_id'], s['forward_key'], dk, ek, strict=strict)
        states[st] = states.get(st, 0) + 1
        if st == 'valid':
            continue
        todo.append((s['case_id'], 'reeval' if st == 'stale_eval' else ('redecode' if st == 'stale_decode' else 'full')))
    return todo, states


# Schedule pending line cases within the selected execution limits.
def run(specs, workers, label, strict=False, purpose='B_research', cap=None, mem_budget_gb=None, per_worker_gb=1.5, stop_on_error=False, deadline_monotonic=None, chunk_per_worker=4):
    set_specs(specs)
    cap = B.RENDER_CAP_RESEARCH if cap is None else int(cap)
    ctx = B.set_budget_context(CASES, cap=cap)
    os.environ['LINE01RB_STRICT_NPZ'] = '1' if strict else '0'
    reconciled = B.reconcile_ledger('research')
    if reconciled:
        B.log(f'{label}: reconciled {len(reconciled)} interrupted attempt(s) from a previous run', WHICH)

    mem = None
    try:
        sys.path.insert(0, str(B.ROOT / 'scripts' / 'spotnoise')); import sn_lib
        tot, free = sn_lib.ram_gb(); mem = dict(total_gb=tot, available_gb=free)
    except Exception:
        pass
    budget_gb = float(mem_budget_gb) if mem_budget_gb else (0.8 * mem['available_gb'] if mem else None)
    workers_req = workers
    if budget_gb is not None and per_worker_gb and workers * per_worker_gb > budget_gb:
        workers = max(1, int(budget_gb // per_worker_gb)); B.log(f'{label}: memory budget {budget_gb:.1f} GB < {workers_req} x {per_worker_gb} GB -> workers reduced to {workers}', WHICH)
    todo0, states = plan(specs, strict=strict)
    n_render = sum(1 for _, m in todo0 if m == 'full') if not OLD_IMAGES else 0
    lc = B.ledger_counts()['research']
    if lc['attempts'] + n_render > cap:
        B.progress(WHICH, state='stopped_budget', blockers=[f'planned renders {n_render} + attempts {lc["attempts"]} > cap {cap}'])
        B.log(f'STOP: render cap would be exceeded ({lc["attempts"]} + {n_render} > {cap})', WHICH); return None

    tokens = iter(range(lc['attempts'] + 1, lc['attempts'] + 1 + n_render)); todo = []; reserved = []
    for cid, m in todo0:
        tok = None
        if m == 'full' and not OLD_IMAGES:
            k = next(tokens); tok = dict(id=f'{label}:{k}', reserved_index=k, purpose=purpose); reserved.append(dict(case_id=cid, token=tok['id']))
        todo.append((cid, m, tok))
    B.ledger_dir().mkdir(parents=True, exist_ok=True)
    B.save_json(B.ledger_dir() / f'reservation_{label}_{os.getpid()}.json', dict(label=label, t_local=B.now_local(), attempts_before=lc['attempts'], cap=cap, n_reserved=n_render, reserved=reserved, ledger_dir=str(B.ledger_dir())))
    disk0 = sum(f.stat().st_size for f in CASES.glob('*') if f.is_file()) / 1e6 if CASES.exists() else 0.0
    B.progress(WHICH, state=f'{label}_running')
    paths_now = B.paths_for(specs[0]) if specs else {}
    B.stage(WHICH, label, planned=len(specs), record_states=states, to_run=len(todo), to_render=n_render, to_redecode=sum(1 for _, m, _t in todo if m == 'redecode'), to_reeval=sum(1 for _, m, _t in todo if m == 'reeval'),
            started_local=B.now_local(), workers=workers, workers_requested=workers_req, memory=dict(mem or {}, budget_gb=budget_gb, per_worker_gb=per_worker_gb), strict_npz=bool(strict), render_cap=cap, budget_context=ctx,
            reservation=dict(attempts_before=lc['attempts'], n_reserved=n_render, interrupted_reconciled=len(reconciled)), parent_config=worker_config(paths_now),
            eval_key=B.eval_key(), decode_keys={n: B.decode_key(pp, 1.0) for n, pp in paths_now.items()}, cases_dir=str(CASES))
    B.log(f'{label}: {len(specs)} cases, states {states}, {len(todo)} to run ({n_render} renders reserved), workers {workers}', WHICH)
    t0 = time.monotonic(); res = []; stop = None
    dispatched = 0; time_stop = False

    def handle(r):
        res.append(r)
        if r['status'] != 'ok':
            B.log(f"  ERROR {r['case_id'][:12]} [{r['mode']}]: {r.get('error')}", WHICH)

            return 'effective window' in str(r.get('error', '')) or bool(stop_on_error)
        return False

    def past_deadline():
        return deadline_monotonic is not None and time.monotonic() >= float(deadline_monotonic)
    if workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        chunk = max(1, int(chunk_per_worker) * workers)
        with mp.Pool(workers, initializer=set_specs, initargs=(specs,)) as pool:
            k = 0
            while k < len(todo) and stop is None:
                if past_deadline():
                    time_stop = True; break
                part = todo[k:k + chunk]; dispatched += len(part); k += len(part)
                for r in pool.imap_unordered(work, part, chunksize=1):
                    if handle(r):
                        stop = ('effective window differs from the registered definition' if 'effective window' in str(r.get('error', '')) else f"program error in case {r['case_id'][:12]}: {r.get('error')}"); pool.terminate(); break
                i = len(res)
                el = time.monotonic() - t0; B.log(f'  {i}/{len(todo)} done, ok {sum(1 for x in res if x["status"] == "ok")} err {sum(1 for x in res if x["status"] != "ok")}, {el / 60:.1f} min, {el / max(i, 1):.1f} s/case wall', WHICH)
                B.progress(WHICH, state=f'{label}_running', done_in_stage=i, planned_in_stage=len(todo), remaining=len(todo) - i, eta_min=round((len(todo) - i) * el / max(i, 1) / 60, 1))
    else:
        for cid, m, tok in todo:
            if past_deadline():
                time_stop = True; break
            dispatched += 1
            r = work((cid, m, tok))
            if handle(r):
                stop = ('effective window differs from the registered definition' if 'effective window' in str(r.get('error', '')) else f"program error in case {cid[:12]}: {r.get('error')}"); break
            B.log(f"  {cid[:12]} [{m}] {r['status']} {r['seconds']}s {r.get('cls', '')}", WHICH)
    if time_stop:
        B.log(f'{label}: deadline reached, no new case dispatched ({dispatched}/{len(todo)} dispatched, {len(res)} finished)', WHICH)
    cls = {}
    for r in res:
        c = r.get('cls', 'program_error') if r['status'] == 'ok' else 'program_error'; cls[c] = cls.get(c, 0) + 1
    modes = {}
    for _, m, _t in todo:
        modes[m] = modes.get(m, 0) + 1
    el = time.monotonic() - t0

    res_info = dict(disk_mb_cases_before=round(disk0, 1), disk_mb_cases_after=round(sum(f.stat().st_size for f in CASES.glob('*') if f.is_file()) / 1e6, 1) if CASES.exists() else None)
    try:
        sys.path.insert(0, str(B.ROOT / 'scripts' / 'spotnoise')); import sn_lib
        res_info.update(peak_rss_mb_parent=sn_lib.peak_rss_mb(), ram_total_free_gb=sn_lib.ram_gb())
    except Exception:
        pass
    lc_after = B.ledger_counts()['research']

    per_pid = {}
    for r in res:
        if r.get('pid') is not None and r.get('peak_rss_mb') is not None:
            per_pid[r['pid']] = max(per_pid.get(r['pid'], 0.0), float(r['peak_rss_mb']))
    res_info.update(worker_peak_rss_mb_by_pid=per_pid, worker_peak_rss_mb_max=(max(per_pid.values()) if per_pid else None), process_tree_peak_rss_mb_estimate=((res_info.get('peak_rss_mb_parent') or 0.0) + sum(per_pid.values())) if per_pid else res_info.get('peak_rss_mb_parent'),
                    timing_split_s=dict(render_sum=round(sum(r.get('render_s') or 0.0 for r in res), 1), reference_sum=round(sum(r.get('ref_s') or 0.0 for r in res), 1), decode_sum=round(sum(r.get('decode_s') or 0.0 for r in res), 1), eval_sum=round(sum(r.get('eval_s') or 0.0 for r in res), 1), wall=round(el, 1)),
                    g_cache_hits=sum(1 for r in res if r.get('g_hit')), g_cache_misses=sum(1 for r in res if r.get('g_hit') is False))
    added = lc_after['attempts'] - lc['attempts']
    acct = dict(reserved=n_render, started=added, ok=lc_after['ok'] - lc['ok'], failed=lc_after['failed'] - lc['failed'], interrupted=lc_after['interrupted'] - lc['interrupted'], not_started=n_render - added, reused_valid=states.get('valid', 0),
                unique_valid_cases_after=sum(1 for s in specs if B.record_state(CASES, s['case_id'], s['forward_key'], {n: B.decode_key(pp, s['peak']) for n, pp in B.paths_for(s).items()}, B.eval_key(), strict=strict)[0] == 'valid'))
    acct['closed'] = (acct['started'] == acct['ok'] + acct['failed'] + acct['interrupted']) and (acct['started'] <= n_render) and (lc_after['attempts'] == lc_after['ok'] + lc_after['failed'] + lc_after['interrupted'])
    reconciled_after = B.reconcile_ledger('research')
    acct['in_flight_interrupted_after_stop'] = len(reconciled_after); acct['dispatched'] = dispatched; acct['not_dispatched'] = len(todo) - dispatched
    B.stage(WHICH, label, done=(stop is None and not time_stop and all(r['status'] == 'ok' for r in res)), n_run=len(res), classes=cls, modes_executed=modes, elapsed_s=round(el, 1), s_per_case_wall=(round(el / len(res), 2) if res else None),
            finished_local=B.now_local(), ledger=B.ledger_counts(), stop_reason=stop, time_limit_stop=bool(time_stop), resources=res_info, render_accounting=acct)
    B.progress(WHICH, state=('stopped: ' + stop) if stop else (f'{label}_partial_time_limit' if time_stop else f'{label}_done'))
    B.log(f'{label} finished: {cls} in {el / 60:.1f} min' + (f'; STOP {stop}' if stop else ''), WHICH)
    return res


def main():
    global CASES, WHICH
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--trial', action='store_true'); ap.add_argument('--full', action='store_true')
    ap.add_argument('--cases', type=str, default=''); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--manifest', type=str, default='', help='manifest CSV (default: the B manifest); forward keys are re-verified')
    ap.add_argument('--progress', type=str, default='', help='progress file name (main | rehearsal | c2 | ...)'); ap.add_argument('--cases-dir', type=str, default='')
    ap.add_argument('--check-only', action='store_true', help='report the record states and the render reservation, run nothing'); ap.add_argument('--strict-npz', action='store_true')
    ap.add_argument('--purpose', type=str, default='B_research'); ap.add_argument('--cap', type=int, default=0, help='render cap of this manifest (default: the B cap)')
    ap.add_argument('--out-dir', type=str, default='', help='directory of progress files / logs / check files (default: the B directory)'); ap.add_argument('--paths-json', type=str, default='', help='JSON {name: params} replacing the B path set')
    ap.add_argument('--mem-budget-gb', type=float, default=0.0); ap.add_argument('--per-worker-gb', type=float, default=1.5)
    a = ap.parse_args()
    if a.cases_dir:
        os.environ['LINE01RB_CASES_DIR'] = a.cases_dir; CASES = Path(a.cases_dir)
    if a.progress:
        os.environ['LINE01RB_PROGRESS'] = a.progress; WHICH = a.progress
    if a.out_dir:
        os.environ['LINE01RB_OUT_DIR'] = a.out_dir
    if a.paths_json:
        os.environ['LINE01RB_PATHS_JSON'] = str(Path(a.paths_json).resolve())
    if a.cap:
        os.environ['LINE01RB_RENDER_CAP'] = str(a.cap)
    man = B.manifest() if not a.manifest else B.manifest_from_csv(a.manifest)
    specs = [s for s in man if s['feasible']]
    infeasible = [s for s in man if not s['feasible']]
    if a.cases:
        want = a.cases.split(','); specs = [s for s in specs if any(s['case_id'].startswith(w) for w in want)]
    elif a.trial:
        specs = [s for s in specs if s.get('trial')]
    elif not a.full and not a.check_only:
        ap.error('choose --trial, --full, --cases or --check-only')
    if a.limit:
        specs = specs[:a.limit]
    if a.check_only:
        ctx = B.set_budget_context(CASES, cap=(a.cap or None)); set_specs(specs); todo, states = plan(specs, strict=a.strict_npz); lc = B.ledger_counts()['research']
        out = dict(t_local=B.now_local(), manifest=a.manifest or 'B', cases_dir=str(CASES), budget_context=ctx, n_specs=len(specs), record_states=states, to_run={m: sum(1 for _, mm in todo if mm == m) for m in ('full', 'redecode', 'reeval')},
                   renders_needed=sum(1 for _, m in todo if m == 'full'), attempts_so_far=lc['attempts'], interrupted_so_far=lc['interrupted'], cap=B.render_cap(), within_cap=(lc['attempts'] + sum(1 for _, m in todo if m == 'full') <= B.render_cap()),
                   paths=sorted(B.paths_for(specs[0])) if specs else [], parent_config=worker_config(B.paths_for(specs[0])) if specs else None)
        B.save_json(B.out_dir() / f'check_{WHICH}.json', out); print(json.dumps(out, ensure_ascii=False)); return
    B.stage(WHICH, 'manifest', planned_total=len(man), feasible=len(man) - len(infeasible), infeasible=len(infeasible), infeasible_cases=[dict(case_id=s['case_id'][:12], reason=s.get('infeasible_reason')) for s in infeasible], manifest_file=(a.manifest or 'B'))
    run(specs, a.workers, 'trial' if a.trial else ('subset' if a.cases else 'full'), strict=a.strict_npz, purpose=a.purpose, cap=(a.cap or None), mem_budget_gb=(a.mem_budget_gb or None), per_worker_gb=a.per_worker_gb)


if __name__ == '__main__':
    main()
