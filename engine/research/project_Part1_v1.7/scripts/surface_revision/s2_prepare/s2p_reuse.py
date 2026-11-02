# Identify reusable spacing-layout images and decoding records.

from __future__ import annotations

import numpy as np
import pandas as pd

import s2p_common as K

K.setup()
TABLE = K.OUT / 's2_s1_reuse.csv'


def _f32(x) -> float:
    return float(np.float32(float(x)))


def _seed(sig, s):
    return None if float(sig) == 0.0 or str(s) in ('', 'nan', 'None') else int(float(s))


# Identify spacing cases whose rendering and decoding inputs match existing results.
def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import s1r_common as S1K
    S1 = pd.read_csv(K.S1OUT / 'manifest' / 's1_images.csv', keep_default_na=False, float_precision='round_trip')
    cfg1 = K.load_json(K.S1OUT / 'config' / 's1_config.json')
    m3_1 = K.digest({k: v for k, v in cfg1['platform']['imaging'].items() if k not in ('noise_sigma', 'mottle_seed')})
    import s2p_manifest as MF
    P = MF.platform_ids()
    same_platform = dict(version=(cfg1['platform']['version'] == P['version']), m3=(m3_1 == P['m3']), psf=(float(cfg1['design']['psf_px']) == K.PSF_PX))
    by_raster = {k: g for k, g in S1.groupby('raster_key')}
    rows = []
    for ik, g in df.groupby('image_key'):
        r = g.iloc[0]; cand = by_raster.get(r.raster_key)
        rec = dict(image_key=ik, tier=';'.join(sorted(set(g.tier))), surface=r.surface, family=r.family, layout=r.layout, sigma=r.sigma, phase_label=r.phase_label, rep=r.rep, regime=r.regime,
                   raster_key=r.raster_key, peak_f32=r.peak_f32, s1_image_key='', s1_decode_key='', image_status='incompatible', decode_status='n/a', eval_status='n/a', reason='')
        if cand is None:
            rec['reason'] = 'no S1 image with the same actual raster'
            rows.append(rec); continue
        m = cand[(cand.platform_surface == r.platform_surface)]
        if m.empty:
            rec['reason'] = 'same raster, other surface only'; rows.append(rec); continue
        m = m[m.sigma.astype(float) == float(r.sigma)]
        if m.empty:
            rec['reason'] = 'same raster / surface, other sigma only'; rows.append(rec); continue
        m = m[[_f32(p) == _f32(r.peak_f32) for p in m.peak_f32]]
        if m.empty:
            rec['reason'] = 'same raster / surface / sigma, different float32 peak (brightness regime / Q differ)'; rows.append(rec); continue
        m = m[[_seed(s1, s) == _seed(r.sigma, r.noise_seed) for s1, s in zip(m.sigma, m.noise_seed)]]
        if m.empty:
            rec['reason'] = 'same raster / surface / sigma / peak, different effective noise seed'; rows.append(rec); continue
        if not all(same_platform.values()):
            rec['reason'] = f'platform / imaging configuration differs {same_platform}'; rows.append(rec); continue
        if len(m) != 1:
            rec['reason'] = f'{len(m)} identical S1 images (S1 duplicate?)'; rows.append(rec); continue
        s1 = m.iloc[0]; rec.update(s1_image_key=s1.image_key, s1_decode_key=s1.decode_key)
        ip = K.S1OUT / 'images' / f'{s1.image_key}.json'
        if not S1K.complete(ip):
            rec.update(image_status='potential_match', reason='identical inputs; S1 image not complete yet (pending)')
            rows.append(rec); continue
        irec = K.load_json(ip)
        if _f32(irec['peak']) != _f32(r.peak_f32) or irec.get('platform_version') != P['version']:
            rec.update(image_status='incompatible', reason='S1 image record peak / version differs from its manifest'); rows.append(rec); continue
        rec['image_status'] = 'ready'
        dp = K.S1OUT / 'decodes' / f'{s1.decode_key}.json'
        if not S1K.complete(dp):
            rec.update(decode_status='pending', reason='image ready; S1 decode not complete yet'); rows.append(rec); continue
        drec = K.load_json(dp); eff = drec.get('effective', {})
        ok_dec = drec.get('arm') == K.ARM[r.family] and list(eff.get('window', [])) == list(K.WINDOW) and (r.family == 'point' or _f32(eff.get('pattern_peak_passed')) == _f32(r.peak_f32))
        rec['decode_status'] = 'ready' if ok_dec else 'incompatible'
        rec['eval_status'] = ('ready' if (ok_dec and drec.get('platform_version') == P['version']) else 're-evaluate only')
        rec['reason'] = 'identical image, decode and evaluation inputs' if rec['eval_status'] == 'ready' else 'decode / evaluation record differs'
        rows.append(rec)
    T = pd.DataFrame(rows)
    summ = dict(images=int(len(T)), by_status={f'{t}|{s}': int(n) for (t, s), n in T.groupby(['tier', 'image_status']).size().items()},
                by_reason={k: int(v) for k, v in T.reason.value_counts().items()}, decode_ready=int((T.decode_status == 'ready').sum()), eval_ready=int((T.eval_status == 'ready').sum()),
                same_platform=same_platform, s1_images_seen=int(len(S1)), note='S1 is still running: potential_match / pending grow into ready over time; nothing is copied')
    return T, summ


def verify_layers(row: dict) -> dict:

    import s1r_common as S1K
    S = S1K.s2(); out = dict(image_ok=False, decode_ok=False, eval_ok=False, reasons=[])
    ip = K.S1OUT / 'images' / f"{row['s1_image_key']}.json"; dp = K.S1OUT / 'decodes' / f"{row['s1_decode_key']}.json"
    if row.get('image_status') != 'ready' or not S1K.complete(ip):
        out['reasons'].append('image layer: identity / completeness'); return out
    irec = K.load_json(ip)
    rel = K.load_json(K.REL / 'release_surf02.json')
    out['image_ok'] = bool(_f32(irec['peak']) == _f32(row['peak_f32']) and irec.get('platform_version') == rel['version'])
    if not out['image_ok']:
        out['reasons'].append('image record peak / version'); return out
    if not S1K.complete(dp):
        out['reasons'].append('decode layer: S1 decode not complete'); return out
    drec = K.load_json(dp); deps = K.load_json(K.S1_DEPS)
    dep_sha = K.sha_file(K.S1_DEPS)
    drift = [f for f, h in deps['files'].items() if not (K.ROOT / f).exists() or K.sha_file(K.ROOT / f) != h]
    eff = drec.get('effective', {}); fam = row['family']; peak = float(row['peak_f32'])
    want = S.line_cfg(peak) if fam == 'line' else S.dot_cfg()
    want_c = K.clean({k: v for k, v in want.items() if isinstance(v, (int, float, str, bool, list, tuple)) or v is None})
    got_c = eff.get('decode_cfg') or {}

    def _eq(a, b):
        if isinstance(b, float) and isinstance(a, (int, float)):
            return _f32(a) == _f32(b)
        return K.clean(a) == K.clean(b)
    cfg_ok = set(got_c) == set(want_c) and all(_eq(got_c[k], v) for k, v in want_c.items())
    p4_ok = (fam != 'line') or (K.clean(eff.get('path')) == K.clean(S.P4))
    dec_ok = (drec.get('s1_dependencies_sha256') == dep_sha and not drift and drec.get('arm') == K.ARM[fam] and list(eff.get('window', [])) == list(K.WINDOW) and cfg_ok and p4_ok
              and (fam == 'point' or _f32(eff.get('pattern_peak_passed')) == _f32(peak)))
    out['decode_ok'] = bool(dec_ok)
    if not dec_ok:
        out['reasons'].append(f"decode layer: deps_sha {drec.get('s1_dependencies_sha256') == dep_sha}, drift {drift[:5]}, cfg {cfg_ok}, p4 {p4_ok}"); return out
    cfg1 = K.load_json(K.S1OUT / 'config' / 's1_config.json')
    dom_ok = cfg1['evaluation']['global_']['hash'] == K.load_json(K.S1OUT / 'domain' / 'global_domain_step0p50.json')['hash']
    const_ok = (float(cfg1['evaluation']['d_mm']), float(cfg1['evaluation']['tau_mm']), float(cfg1['evaluation']['gross_mm'])) == (K.D_SUPPORT, K.TAU, K.GROSS)
    out['eval_ok'] = bool(dom_ok and const_ok and drec.get('platform_version') == rel['version'])
    if not out['eval_ok']:
        out['reasons'].append(f'eval layer: domain {dom_ok}, constants {const_ok}')
    out.update(s1_dependencies_sha256=dep_sha, n_dependency_files=len(deps['files']), decode_cfg_keys=sorted(got_c), s1_decode_record=K.rel(dp))
    return out


def write(df):
    T, summ = build(df)
    keep = T[T.image_status != 'incompatible'] if (T.image_status != 'incompatible').any() else T.head(0)
    keep.to_csv(K.guard(TABLE), index=False)
    return T, summ
