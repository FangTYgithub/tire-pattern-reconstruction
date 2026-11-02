# Expose the fixed surface platform through the common adapter interface.

from __future__ import annotations

import sys

import numpy as np

import adp_common as C

S2_DIR = C.ROOT / 'scripts' / 'surface_revision' / 'surf02'
S2_CONFIG = C.ROOT / 'outputs' / 'surface_revision' / 'surf02' / 'config' / 'surf02_config.json'
RELEASE_RECORD = C.ROOT / 'outputs' / 'surface_revision' / 'surf02' / 'release' / 'release_surf02.json'
ID_MAP = {'M10_curv': 'A_curv', 'M24_curv': 'C_curv', 'B_curv': 'B_curv', 'M10_curv_gauss': 'A_curv_gauss'}
METHODS = {'D0': 'D0', 'P4_C3_cal': 'P4_C3_cal'}
_S2 = None


def s2():
    global _S2
    if _S2 is None:
        if str(S2_DIR) not in sys.path:
            sys.path.append(str(S2_DIR))
        import s2_common as S
        _S2 = S
    return _S2


def version():
    return s2().VERSION


def dependency_drift(manifest_path) -> list[str]:

    m = C.load_json(manifest_path); bad = []
    for rp, h in m['files'].items():
        q = C.absp(rp)
        if not q.exists():
            bad.append(f'{rp}: missing')
        elif C.sha_file(q) != h:
            bad.append(f'{rp}: sha256 differs')
    return bad


def release():
    if not RELEASE_RECORD.exists():
        return dict(released=False, reason=f'no SURF-02 release record at {C.rel(RELEASE_RECORD)}', config_sha256=None)
    r = C.load_json(RELEASE_RECORD)
    man = C.absp(r['dependency_manifest'])
    if not man.exists() or C.sha_file(man) != r.get('dependency_manifest_sha256'):
        raise C.AdapterStop('platform_mismatch', f'dependency manifest {r.get("dependency_manifest")} missing or changed since the release')
    if r.get('version') != version():
        raise C.AdapterStop('platform_mismatch', f'released version {r.get("version")} differs from the platform {version()}')
    drift = dependency_drift(man)
    if drift:
        raise C.AdapterStop('platform_mismatch', f'numerical dependencies changed since the release ({len(drift)}): {drift[:6]}')
    return dict(released=bool(r.get('released')), config_sha256=C.sha_file(man), record=C.rel(RELEASE_RECORD), version=r.get('version'))


def surface_record(sid):
    S = s2(); mesh = S.build_mesh(sid); dom = S.roi_domain(mesh)
    return dict(surface=sid, mesh_sha16=S.mesh_sha(mesh), c_ref_mm=S.C_REF, grid=list(S.GRID_SCI), frame=mesh['meta']['frame'], z_alignment=mesh['z_alignment'],
                domain_hash=dom['hash'], n_domain_nodes=dom['n_nodes'], roi_r_mm=S.ROI_R_MM, window_cov_roi=dom['window_cov_roi'])


def sensor_record():
    return s2().geometry_record()


def search_window():
    return (float(s2().Z0_MM), float(s2().Z1_MM))


def render(sid, pattern, peak, sigma, seed, psf_px):
    from m3_virtual_camera import synthesise_camera_image
    S = s2()
    cfg = S.m3_cfg(float(sigma), int(seed))
    if abs(float(cfg.get('psf_sigma_px', psf_px)) - float(psf_px)) > 1e-12:
        raise C.AdapterStop('platform_mismatch', f"platform PSF {cfg.get('psf_sigma_px')} != candidate PSF {psf_px}")
    img, _, _, m3 = synthesise_camera_image(S.build_mesh(sid), np.asarray(pattern, np.float32), cfg)
    if m3.get('z_alignment') != 'absolute':
        raise C.AdapterStop('platform_mismatch', f"M3 alignment {m3.get('z_alignment')} (expected 'absolute', no Z - min Z)")
    return dict(image=np.asarray(img, np.uint8), meta=dict(z_alignment=m3.get('z_alignment'), n_hits=int(m3['n_hits']), noise_sigma=float(m3['noise_sigma']), mottle_seed=int(m3['mottle_seed'])))


def decode(method, image, ctx):
    S = s2()
    if method == 'D0':
        RD = S.RD
        binpat = np.asarray(ctx['binary'], np.float32)
        cd = RD.candidates(image); feats, det_info, code, fidx, _ = RD.select(cd, 'D0')
        cfg = S.dot_cfg()
        pts, info, tr = RD.reconstruct_from(feats, det_info, binpat, cfg=cfg)
        z0, z1 = S.M4.search_window(cfg, float(cfg['measurement_volume_mm']))
        if abs(z0 - S.Z0_MM) > 1e-9 or abs(z1 - S.Z1_MM) > 1e-9:
            raise C.AdapterStop('platform_mismatch', f'effective dot window {(z0, z1)}')
        pidx = tr['point_idx']; has_pt = pidx >= 0
        acc = np.zeros(len(pts), bool); acc[pidx[has_pt]] = True
        amb = np.zeros(len(pts), bool); amb[pidx[has_pt & tr['ambiguous']]] = True
        acc &= ~amb & np.isfinite(pts).all(axis=1)
        return dict(pts=np.asarray(pts, np.float64), accepted=acc, aux=dict(feats=feats, tr=tr, window=(z0, z1), cd=cd, sel_code=code, sel_fidx=fidx, det_info=det_info, info=info, cfg=cfg))
    if method == 'P4_C3_cal':
        from m4_lines import run_olines
        cfg = S.line_cfg(float(ctx['peak']))
        r = run_olines(image, ctx['meta_full']['olines'], cfg, pattern=np.asarray(ctx['pattern'], np.float32), **S.P4)
        if abs(float(r['window_z0_mm']) - S.Z0_MM) > 1e-9 or abs(float(r['window_z1_mm']) - S.Z1_MM) > 1e-9:
            raise C.AdapterStop('platform_mismatch', f"effective line window {r['window_z0_mm']}, {r['window_z1_mm']}")
        return dict(pts=np.asarray(r['points'], np.float64), accepted=np.asarray(r['accepted'], bool), aux=dict(r=r, cfg=cfg))
    raise C.AdapterStop('capability_missing', f'SURF-02 v1 shim has no decode method {method!r}')


def evaluate(sid, family, decoded, ctx):

    S = s2(); mesh = S.build_mesh(sid); dom = S.roi_domain(mesh); gauss = S.SURFACES[sid]['gauss']
    pts = decoded['pts']; acc = decoded['accepted']
    if family == 'point':
        g = S.C0.geom(); feats = decoded['aux']['feats']; tr = decoded['aux']['tr']; c = np.asarray(ctx['centres'], np.float64); binpat = np.asarray(ctx['binary'], np.float32)
        r_px = float(ctx['row']['radius_mm']) / S.SL.mmpp()
        n = len(feats); truth = np.full(n, -1, np.int64); conf = np.zeros(n, bool)
        if n:
            hit, P = S.C0.cast(feats, g['cam'], mesh); uvp, inside = S.C0.project(P, g['proj'])
            d = np.hypot(uvp[:, None, 0] - c[None, :, 0], uvp[:, None, 1] - c[None, :, 1]); j = d.argmin(axis=1); dmin = d[np.arange(n), j]
            conf = hit & inside & (dmin <= r_px + 1.0); truth = np.where(conf, j, -1)
        proj_dot, _ = S.RD.projector_dot_index(binpat, c)
        matched = np.where(tr['best_idx'] >= 0, proj_dot[np.clip(tr['best_idx'], 0, None)], -1)
        has_pt = tr['point_idx'] >= 0; na = ~tr['ambiguous']
        exp = S.SL.expected_dots(mesh, c); uv_e = np.asarray(exp['uv'])
        in_img = exp['hit'] & np.isfinite(uv_e).all(axis=1) & (uv_e[:, 0] >= 0) & (uv_e[:, 0] <= S.IMG - 1) & (uv_e[:, 1] >= 0) & (uv_e[:, 1] <= S.IMG - 1)
        idn = dict(correct=int((has_pt & conf & (matched == truth) & na).sum()), wrong=int((has_pt & conf & (matched != truth) & na).sum()), unconfirmable=int((has_pt & ~conf & na).sum()),
                   n_expected_visible=int(in_img.sum()))
        idn['ref_rows'] = None
    else:
        r = decoded['aux']['r']; feat = r['features']; lid = np.asarray(r['line_id'])
        ident = S.T.truth_identity(mesh, feat, ctx['meta_full']['olines']); kO = ident['k']; conf = ident['confirmable']
        ref = S.T.line_reference(mesh, ctx['meta_full']['olines'], 1.0, pattern=np.asarray(ctx['pattern'], np.float32))
        rec_rows, _ = S._row_flags(ref, feat, lid, acc)
        idn = dict(correct=int((acc & conf & (lid == kO)).sum()), wrong=int((acc & conf & (lid != kO)).sum()), unconfirmable=int((acc & ~conf).sum()),
                   ref_rows=int(len(ref['rows']['v'])), ref_rows_recovered=int(rec_rows.sum()))
    ev = S.evaluate(pts, acc, mesh, dom, gauss)
    return eval_record_from_s2(ev, idn, family)


def eval_record_from_s2(ev: dict, idn: dict, family: str) -> dict:
    S_g = 1.0
    try:
        S_g = float(s2().GROSS_MM)
    except Exception:
        pass
    nodes = ev['nodes_roi']; am = ev.get('amplitude') or {}
    d6 = (((ev.get('r3_dev') or {}).get('subdomains') or {}).get('global') or {}).get('by_d', {}).get('6.0')
    rec = dict(state='no_output' if ev['no_output'] else 'ok', n_points_out=ev['n_output'], n_accepted=ev['n_valid'], n_in_domain=ev['n_valid_roi'],
               n_out_of_domain=ev['n_valid'] - ev['n_valid_roi'], point_stats=ev['pts_roi'], surface_stats=dict(n=nodes.get('n', 0), rmse=nodes.get('rmse'), mae=nodes.get('mae'),
                                                                                                                  p95_abs=nodes.get('p95_abs'), bias=nodes.get('bias')),
               n_domain_nodes=nodes['n_nodes'], gross_threshold_mm=S_g, gross_rate=ev['gross_rate_roi'],
               n_id_correct=idn['correct'], n_id_wrong=idn['wrong'], n_id_unconfirmable=idn['unconfirmable'], n_id_denominator=idn['correct'] + idn['wrong'] + idn['unconfirmable'],
               n_out_of_window=ev['n_out_of_window'], n_out_of_grid=ev['n_out_of_grid'], n_nonfinite=ev['n_nonfinite'])
    if family in ('point', 'dots'):
        rec.update(n_expected=idn.get('n_expected_visible') or 0, n_recovered=idn['correct'])
    else:
        rec.update(n_expected=idn.get('ref_rows') or 0, n_recovered=idn.get('ref_rows_recovered') or 0)
    if am.get('applicable'):
        rec.update(amp_true_mm=am['true_mm'], amp_est_mm=am.get('rec_peak_mm'), amp_definition=am.get('definition'))
    if d6:
        rec.update(Ctau=d6['C_tau']['0.15'], Cvalid=d6['C_valid'])
    elif ev['no_output']:
        rec.update(Ctau=0.0, Cvalid=0.0)
    return rec


def stored_case_record(case_json: dict) -> dict:

    fam = case_json['case']['family']; idn = dict(case_json['identity'])
    if fam == 'lines':
        idn.setdefault('ref_rows_recovered', idn.get('ref_rows_recovered'))
    return eval_record_from_s2(case_json['evaluation'], idn, 'point' if fam == 'dots' else 'line')


def replay(case_ids=None, out_json=None) -> dict:

    S = s2(); cases_dir = C.ROOT / 'outputs' / 'surface_revision' / 'surf02' / 'cases'
    ids = case_ids or sorted(p.stem for p in cases_dir.glob('*.json') if '_grid' not in p.stem)
    inv = {v: k for k, v in ID_MAP.items()}; out = []
    for cid in ids:
        rec = C.load_json(cases_dir / f'{cid}.json')
        with np.load(cases_dir / f'{cid}.npz') as z:
            img = np.asarray(z['img']); pts0 = np.asarray(z['pts'], np.float64); acc0 = np.asarray(z['acc'], bool)
        cs = rec['case']; sid = cs['surface']; fam = 'point' if cs['family'] == 'dots' else 'line'
        if fam == 'point':
            c, binpat, pat, _ = S.dot_pattern(cs['phase_index'])
            ctx = dict(binary=binpat, centres=c, pattern=pat, peak=S.DOT['peak'], row=dict(radius_mm=S.DOT['r_mm']), meta_full=None)
            method = 'D0'
        else:
            spec, pat_bin, pat, meta = S.line_pattern(cs['phase_index'])
            ctx = dict(binary=pat_bin, centres=None, pattern=pat, peak=S.LINE['peak'], row={}, meta_full=meta)
            method = 'P4_C3_cal'
        dec = decode(method, img, ctx)
        ev = evaluate(sid, fam, dec, ctx)
        same_pts = bool(dec['pts'].shape == pts0.shape and np.allclose(dec['pts'], pts0, atol=1e-5, equal_nan=True) and np.array_equal(dec['accepted'], acc0))
        stored = stored_case_record(rec)
        same_metrics = all((ev.get(k) == stored.get(k)) for k in ('n_accepted', 'n_in_domain', 'n_id_correct', 'n_id_wrong', 'n_id_unconfirmable', 'n_expected', 'n_recovered'))
        out.append(dict(case_id=cid, surface=inv[sid], family=fam, same_points=same_pts, same_counts=same_metrics))
    res = dict(t_local=C.now_local(), n=len(out), all_equal=all(r['same_points'] and r['same_counts'] for r in out), cases=out, note='decodes of stored SURF-02 images only; no imaging')
    if out_json:
        C.save_json(C.assert_inside_out(out_json), res)
    return res


def binding_candidate() -> dict:

    import adp_binding as BD
    b = BD.binding_template()
    b['status'] = 'CANDIDATE for SURF-02 v1 via adp_surf02 shim: release_record / release_record_sha256 missing (main project issues them after user confirmation)'
    b['platform'] = dict(module='adp_surf02', module_dir=C.rel(C.HERE), expected_version=version(), release_record=C.rel(RELEASE_RECORD), release_record_sha256=None)
    b['capabilities'] = {k: k for k in BD.CAPABILITIES}
    b['surface_id_map'] = dict(ID_MAP)
    b['methods'] = dict(METHODS)
    b['unmapped_decode_arms'] = ['ENC_coded', 'ENC_ignore', 'LC_ignore', 'LC_use', 'LC_R', 'LC_A', 'P4_ref', 'SPAT_D0', 'SPAT_P4table', 'P2_C1_guard']
    return b
