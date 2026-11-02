# Prepare and execute extended environment cases with fixed DF/P4 decoding.

from __future__ import annotations

import json
import time
import traceback

import numpy as np

import c_common as K

K5 = K.K5
_ST: dict = {}


def platform():
    if not _ST:
        import pandas as pd
        import r_inputs as RI
        import mp5_inputs as IN
        import m6_platform as PL
        PL.platform(); K.set_sandbox_env()
        M5 = pd.read_csv(K.SRC5 / 'formal_manifest.csv', float_precision='round_trip', low_memory=False)
        _ST.update(RI=RI, IN=IN, PL=PL, T=RI.template_rows(M5), q=IN.adapter(), mesh={})
        K.set_sandbox_env()
    return _ST


def _num(x):
    return None if K.undef(x) else float(x)


def prep(row: dict) -> dict:
    p = platform(); RI, IN, T = p['RI'], p['IN'], p['T']
    fam = row['family']; cfg = row['config']
    if cfg not in T:
        raise K.Stop('identity', f'{row["case_key"]}: no template for {cfg}')
    tpl = T[cfg]['row']
    ph = dict(u=float(row['phase_u']), v=float(row['phase_v'])) if fam == 'point' else dict(phi=float(row['phase_frac']))
    inp = RI.inputs(tpl, fam, ph)
    pat = np.asarray(inp['pattern'], np.float32)
    chk = dict(raster=inp['raster_key'] == row['raster_key'], pattern=K5.content_sha(pat) == row['pattern_sha'], peak=float(np.float32(inp['peak'])) == float(row['peak_f32']),
               integral=abs(float(pat.sum(dtype=np.float64)) - float(row['integral'])) <= 1e-6, template=tpl['cand_id'] == row['template_cand_id'])
    if not all(chk.values()):
        raise K.Stop('identity', f'{row["case_key"]}: inputs differ from the manifest {chk}')
    r = dict(platform_surface=row['platform_surface'], sigma=float(row['sigma']), seed=row.get('seed'), psf_true_px=float(row['psf_true_px']), field_json=row['field_json'])
    spec, rec, route = RI.forward_record(r, inp)
    fk = K5.digest(rec)
    if fk != row['forward_key'] or route != row['forward_route']:
        raise K.Stop('identity', f'{row["case_key"]}: recomputed forward key / route differs from the manifest')
    ik = IN.input_key(tpl['cand_id'], fam, ph, inp); p['q'].register(ik, inp); K.set_sandbox_env()
    radius = float(tpl['radius_mm']) if fam == 'point' else float('nan')
    return dict(row=row, case_id=row['case_key'], family=fam, sid=row['platform_surface'], surface=row['surface'], pattern=pat, spec=spec, forward_key=fk, route=route,
                decoder_row=dict(input_key=ik, radius_mm=radius, family=fam), ctx=dict(meta_full=inp['meta_full'], binary=inp['binary']), method=row['method'],
                input_key=ik, inputs=dict(template=tpl['cand_id'], raster_key=inp['raster_key'], peak=float(inp['peak']), lit_px=float(inp['lit_px'])))


def render(pr: dict):
    p = platform()
    import mp3_forward as F
    import r_forward as RFW
    if pr['sid'] not in p['mesh']:
        import rdy_platform as RP
        p['mesh'][pr['sid']] = RP.s2().build_mesh(pr['sid'])
    fn = RFW.render if pr['route'] == 'r_forward' else F.render
    out = fn(p['mesh'][pr['sid']], pr['pattern'], pr['spec'])
    img = out['image']
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.shape != (512, 512):
        raise K.Stop('integrity', f"{pr['case_id']}: render returned {type(img).__name__}")
    if int(out['diag'].get('rays_per_pixel', 0)) != 1:
        raise K.Stop('integrity', f"{pr['case_id']}: rays per pixel {out['diag'].get('rays_per_pixel')}")
    return img, out['diag']


def decode_eval(pr: dict, img) -> dict:
    p = platform(); q = p['q']; PL = p['PL']; m = pr['method']
    res = q.decode(dict(pr['decoder_row']), m, dict(image=np.asarray(img, np.uint8)))
    if not isinstance(res, tuple) or len(res) != 2:
        raise K.Stop('integrity', f"{pr['case_id']}: decoder returned {type(res).__name__}")
    raw, dmeta = res; raw = {k: np.asarray(v) for k, v in raw.items()}
    ev, rec, fi = PL.evaluate(pr, m, raw, dmeta)
    return dict(raw=raw, dmeta=dmeta, ev=ev, rec=rec)


DIAG_KEEP = ('rays_per_pixel', 'n_rays_cast', 'n_hits', 'n_valid_pixels', 'field_id', 'psf_true_px', 'noise_sha', 'noise_std_preclip_grey', 'frac_pre_clip_above_1', 'image_sha',
             'received_light_ratio', 'gain_hit_weighted_mean', 'signal_peak_noise_free_grey', 'image_mean_grey', 'image_frac_255', 'gain_min', 'gain_max')


def result_row(pr: dict, img, diag: dict, d: dict, seconds: float) -> dict:
    row = pr['row']; m = d['rec']['metrics']; fam = pr['family']
    fi = ((d['rec'].get('full_identity') or {}).get('counts') or {}).get('all') if fam == 'line' else None
    c, w, u = ((fi['c'], fi['w'], fi['u']) if fi else (m.get('id_correct'), m.get('id_wrong'), m.get('id_unconf')))
    sup = d['rec'].get('support') or {}
    info = (d['dmeta'] or {})
    out = dict(case_key=pr['case_id'], family=fam, config=row['config'], surface=row['surface'], phase_block=row['phase_block'], sigma=float(row['sigma']), rep=int(row['rep']),
               env=row['env'], psf_true_px=float(row['psf_true_px']), field_kind=row['field_kind'], method=pr['method'], forward_route=pr['route'], forward_key=pr['forward_key'],
               image_sha=K5.content_sha(img), state=d['rec']['state'],
               glob_Ctau=m.get('glob_Ctau'), loc_Ctau=m.get('loc_Ctau'), glob_Cvalid=m.get('glob_Cvalid'), loc_Cvalid=m.get('loc_Cvalid'), glob_pt_rmse=m.get('glob_pt_rmse'),
               glob_pt_p95=m.get('glob_pt_p95'), loc_pt_rmse=m.get('loc_pt_rmse'), glob_surf_rmse=m.get('glob_surf_rmse'), n_valid=m.get('n_valid'), glob_n_valid=m.get('glob_n_valid'),
               n_accepted=m.get('n_accepted'), c=c, w=w, u=u, w_share=((w / (c + w)) if (c is not None and c + w > 0) else None),
               u_share=((u / (c + w + u)) if (c is not None and c + w + u > 0) else None), sup_surf_rmse=sup.get('sup_surf_rmse'),
               img_n_expected=m.get('img_n_expected'), img_n_recovered=m.get('img_n_recovered'), n_detected=m.get('n_detected'),
               diag_received_light_ratio=diag.get('received_light_ratio'), diag_signal_peak_noise_free_grey=diag.get('signal_peak_noise_free_grey'),
               diag_frac_pre_clip_above_1=diag.get('frac_pre_clip_above_1'), diag_noise_sha=diag.get('noise_sha'), seconds=round(seconds, 2))
    if fam == 'line':
        out.update(dec_steger_sigma=(info.get('pack_meta') or {}).get('a0_r', {}).get('detect_info', {}).get('steger_sigma_px'),
                   dec_steger_thr=(info.get('pack_meta') or {}).get('a0_r', {}).get('detect_info', {}).get('steger_threshold'),
                   dec_sigma_hat=(info.get('pack_meta') or {}).get('a0_r', {}).get('detect_info', {}).get('noise_sigma_hat'))
    return out


def save_pack(pr: dict, img, diag: dict, d: dict, row_out: dict, group: str) -> str:
    p = K.DATA / group / f"{pr['case_id'].replace('|', '__')}.npz"
    m = pr['method']
    arrays = dict(image=img); arrays.update({f'{m}__raw__{k}': v for k, v in d['raw'].items()}); arrays.update({f'{m}__ev__{k}': v for k, v in d['ev'].items()})
    meta = dict(case_key=pr['case_id'], forward_key=pr['forward_key'], route=pr['route'], input_key=pr['input_key'], inputs=pr['inputs'],
                diag={k: diag.get(k) for k in DIAG_KEEP}, method=m, decode_meta=K.clean(d['dmeta']), metrics=d['rec']['metrics'], support=d['rec'].get('support'),
                full_identity=d['rec'].get('full_identity'), state=d['rec']['state'], raw_digest=K5.arrays_digest(d['raw']), eval_digest=K5.arrays_digest(d['ev']), result=row_out)

    def w(t):
        with open(t, 'wb') as fh:
            np.savez_compressed(fh, _meta=np.array(json.dumps(K.clean(meta), sort_keys=True, ensure_ascii=False)), **arrays)
    q = K.atomic(p, w)
    return K.sha_file(q)


def work(args) -> dict:
    row, group = args
    t0 = time.time(); step = 'prep'
    try:
        pr = prep(row); step = 'render'
        img, diag = render(pr); step = 'decode/evaluate'
        d = decode_eval(pr, img); step = 'pack'
        out = result_row(pr, img, diag, d, time.time() - t0)
        sha = save_pack(pr, img, diag, d, out, group)
        return dict(ok=True, case_key=row['case_key'], row=out, pack_sha=sha)
    except BaseException as e:
        return dict(ok=False, case_key=row['case_key'], step=step, error=repr(e)[:400], trace=traceback.format_exc()[-1500:])
