# Match size-spacing cases to existing image and decoding records.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_common as K

SOURCES = K.CHECKS / 'reuse_sources.json'
CHECK = K.CHECKS / 'reuse_check.json'


def _seed(x):
    s = str(x).strip()
    return None if s in ('', 'nan', 'None') else int(round(float(s)))


def source_paths(r) -> dict:
    if r.source == 's1':
        return dict(image=K.S1OUT / 'images' / f'{r.s1_image_key}.json', decode=K.S1OUT / 'decodes' / f'{r.s1_decode_key}.json', eval=K.S1OUT / 'decodes' / f'{r.s1_decode_key}.json')
    if r.source == 's2':
        return dict(image=K.S2OUT / 'run' / 'images' / f'{r.s2_image_key}.json', decode=K.S2OUT / 'run' / 'raw' / f'{r.s2_decode_key}.json', eval=K.S2OUT / 'run' / 'eval' / f'{r.s2_decode_key}.json')
    return {}


def eval_record_path(r) -> Path:

    if r.source == 's1':
        return K.S1OUT / 'decodes' / f'{r.s1_decode_key}.json'
    if r.source == 's2':
        return K.S2OUT / 'run' / 'eval' / f'{r.s2_decode_key}.json'
    return K.EVAL_DIR / f'{r.decode_key}.json'


def image_record_path(r) -> Path:
    if r.source == 's1':
        return K.S1OUT / 'images' / f'{r.s1_image_key}.json'
    if r.source == 's2':
        return K.S2OUT / 'run' / 'images' / f'{r.s2_image_key}.json'
    return K.IMG_DIR / f'{r.input_key}.json'


# Check source identities and numerical settings before reusing stored inputs.
def verify(U: pd.DataFrame | None = None, hashes: bool = True) -> dict:
    import adp_surf02 as SA
    import s1r_common as S1K
    import s2p_case as SC
    import s2p_formal as SF
    S = K.s2(); bd = K.bound(); P = K.platform_ids()
    U = pd.read_csv(K.INPUTS, keep_default_na=False, float_precision='round_trip') if U is None else U
    R = U[U.source != 'new']
    S1K.check_s1_dependencies()
    s1deps_sha = S1K.sha_file(S1K.S1_DEPS)
    fz2 = SF.verify_freeze()
    parts2 = fz2['part_digests']
    I1 = pd.read_csv(K.S1OUT / 'manifest' / 's1_images.csv', keep_default_na=False, float_precision='round_trip').set_index('image_key')
    C2 = pd.read_csv(K.S2OUT / 's2_candidates.csv', keep_default_na=False, float_precision='round_trip').set_index('cand_id')
    imaging = {k: v for k, v in S.m3_cfg(0.0, 0).items() if k not in ('noise_sigma', 'mottle_seed')}
    sensor_d = S1K.digest(S.geometry_record()); surf_d = {}
    rows = []; src_hash = {}
    for r in R.itertuples():
        out = dict(input_key=r.input_key, source=r.source, image=False, decode=False, eval=False, reason='')
        ps = source_paths(r)
        if r.source == 's1':
            s1 = I1.loc[r.s1_image_key]
            ok_rec = S1K.complete(ps['image']) and S1K.complete(ps['decode'])
            sid = s1.platform_surface
            if sid not in surf_d:
                surf_d[sid] = S1K.digest(SA.surface_record(sid))
            seed = _seed(s1.noise_seed) if float(s1.sigma) > 0 else None
            ik = 'S1I-' + S1K.digest(dict(raster=s1.raster_key, peak_f32=float(s1.peak_f32), surface=sid, sigma=float(s1.sigma), seed=seed, psf_px=S1K.PSF_PX, imaging=imaging, sensor=sensor_d,
                                          surface_record=surf_d[sid], platform=S.VERSION), 24)
            phys = (s1.raster_key == r.raster_key and float(np.float32(s1.peak_f32)) == float(np.float32(r.peak_f32)) and sid == r.platform_surface and float(s1.sigma) == float(r.sigma)
                    and seed == (_seed(r.noise_seed) if float(r.sigma) > 0 else None))
            out['image'] = bool(ok_rec and ik == r.s1_image_key and phys)
            dcfg = dict(arm=S1K.ARM[s1.family], cfg=(S.dot_cfg() if s1.family == 'point' else dict(P4=S.P4, cfg=S.line_cfg(float(s1.peak)))))
            dk = 'S1D-' + S1K.digest(dict(image=ik, decode=dcfg, platform=S.VERSION, window=list(bd.window())), 24)
            out['decode'] = bool(out['image'] and dk == r.s1_decode_key)
            drec = K.load_json(ps['decode']) if ok_rec else {}
            out['eval'] = bool(out['decode'] and drec.get('s1_dependencies_sha256') == s1deps_sha and drec.get('platform_version') == P['version'])
            if not all((out['image'], out['decode'], out['eval'])):
                out['reason'] = f"rec {ok_rec} key {ik == r.s1_image_key} phys {phys} dec {dk == r.s1_decode_key} deps {drec.get('s1_dependencies_sha256') == s1deps_sha}"
        else:
            c = C2.loc[r.s2_cand_id]; t = c.to_dict(); t['cand_id'] = r.s2_cand_id
            peak = 1.0 if c.regime == 'fixed_peak' else 11695.0 / float(c.integral_I0)
            ids = SC.identities(t, bd, parts2, peak)
            ri, rr, re = K.load_json(ps['image']) if ps['image'].exists() else {}, K.load_json(ps['decode']) if ps['decode'].exists() else {}, K.load_json(ps['eval']) if ps['eval'].exists() else {}
            out['image'] = bool(SC.complete(ps['image'], ids['image']) and c.raster_key == r.raster_key and float(np.float32(peak)) == float(np.float32(r.peak_f32)))
            out['decode'] = bool(out['image'] and SC.complete(ps['decode'], ids['decode']))
            out['eval'] = bool(out['decode'] and SC.complete(ps['eval'], ids['eval']) and re.get('platform_version') == P['version'])
            if not all((out['image'], out['decode'], out['eval'])):
                out['reason'] = f"image id {ri.get('identity') == ids['image']} raw id {rr.get('identity') == ids['decode']} eval id {re.get('identity') == ids['eval']}"
        rows.append(out)
        if hashes:
            for p_ in {ps['image'], ps['decode'], ps['eval']}:
                for q in (p_, p_.with_suffix('.npz')):
                    src_hash[K.rel(q)] = K.sha_file(q) if q.exists() else None
    T = pd.DataFrame(rows)
    bad = T[~(T['image'] & T['decode'] & T['eval'])]
    res = dict(t_local=K.now_local(), n_reused=int(len(T)), by_source={s: int(n) for s, n in T.source.value_counts().items()}, all_three_layers=bool(bad.empty), n_failed=int(len(bad)),
               failed=bad.head(20).to_dict('records'), s1_dependencies_sha256=s1deps_sha, s2_formal_freeze=fz2, platform=dict(version=P['version'], m3=P['m3']),
               layers='image: record + physical tuple + key reproduced from the current platform; decode: decode key / identity reproduced; eval: S1 dependency manifest (no drift) / S2 eval identity + S2 freeze')
    if hashes:
        K.save_json(SOURCES, dict(t_local=K.now_local(), n_files=len(src_hash), files=src_hash))
        res['source_files_hashed'] = len(src_hash)
    K.save_json(CHECK, res)
    return res


def sources_unchanged() -> dict:

    rec = K.load_json(SOURCES)
    changed = [f for f, h in rec['files'].items() if h is None or not (K.ROOT / f).exists() or K.sha_file(K.ROOT / f) != h]
    return dict(t_local=K.now_local(), n_files=rec['n_files'], changed=changed[:20], n_changed=len(changed), ok=not changed, start=rec['t_local'])


if __name__ == '__main__':
    import json
    r = verify()
    print(json.dumps(K.clean({k: v for k, v in r.items() if k != 'failed'} | dict(failed=r['failed'][:3])), ensure_ascii=False, indent=1))
    sys.exit(0 if r['all_three_layers'] else 16)
