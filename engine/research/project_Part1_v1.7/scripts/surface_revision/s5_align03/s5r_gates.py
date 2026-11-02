# Validate joint-study matrices and compute identity-risk counts.

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import s5r_common as K

MANIFEST_COLS = ('manifest_key', 'family', 'combo', 'radius_mm', 'line_width_mm', 'p_parallel_mm', 'p_perp_mm', 'line_pitch_mm', 'orientation_deg', 'surface',
                 'platform_surface', 'phase_label', 'sigma', 'rep', 'noise_seed', 'raster_key', 'lit_px', 'Q0', 'peak', 'integral', 'limit_type', 'image_identity_fields',
                 'm3_digest', 'status', 'existing_image_key')
NUMERIC = ('sigma', 'rep', 'lit_px', 'Q0', 'peak', 'integral')


def _verdict(reasons: list, **extra) -> dict:
    cnt = Counter(r['check'] for r in reasons)
    return dict(ok=not reasons, n_reasons=len(reasons), by_check=dict(cnt), reasons=reasons[:40], **extra)


def _blank(x) -> bool:
    return x is None or (isinstance(x, float) and not np.isfinite(x)) or (isinstance(x, str) and x.strip() in ('', 'nan', 'None', 'NaN'))


def seed_of(x):

    if _blank(x):
        return None
    f = K.num(x) if not isinstance(x, str) else K.num(pd.to_numeric(x, errors='coerce'))
    if f is None or abs(f - round(f)) > 0:
        return 'invalid'
    return int(round(f))


def image_identity_fields(row) -> dict:
    s = seed_of(row['noise_seed'])
    return dict(raster=row['raster_key'], peak_f32=float(np.float32(float(row['peak']))), surface=row['platform_surface'], psf=K.PSF_PX, sigma=float(row['sigma']),
                seed=(None if s in (None, 'invalid') else s), m3=row['m3_digest'], platform=K.PLATFORM_VERSION)


def registered() -> dict:

    P = K.read_csv(K.reg(K.S345 / 'plan_s345.csv'))
    P = P[(P.set_formal == 'S5-main') & P.pair_role.isin(K.COMBOS)]
    s40 = {}
    for r in P[P.sigma == 40.0].itertuples():
        k = (K.family(r.pair_role), r.surface, r.phase_label, int(r.rep)); v = seed_of(r.noise_seed)
        if k in s40 and s40[k] != v:
            raise K.Stop('scope', f'registered sigma-40 seeds not unique for {k}')
        s40[k] = v
    img = {}
    for r in P.drop_duplicates('image_key').itertuples():
        j = K.load_json(K.reg(K.S345 / 'run' / 'images' / f'{r.image_key}.json'))
        img[r.image_key] = dict(key=(r.pair_role, r.surface, r.phase_label, float(r.sigma), int(r.rep)), fields={k: v for k, v in j['identity_fields'].items() if k != 'image_part'},
                                raster_key=j['raster_key'], peak=float(j['peak']), lit_px=float(j['lit_px']), integral=float(j['integral_actual']), npz_sha256=j['npz_sha256'])
    cell0 = {(v['key'][0], v['key'][1], v['key'][2]): v for v in img.values() if v['key'][3] == 0.0}
    return dict(seed40=s40, images=img, cell_sigma0=cell0)


# Check registered case coverage and input identities before execution.
def gate_manifest(M: pd.DataFrame, mode: str = 'full', subset_keys=None, reg: dict | None = None) -> dict:
    R = []

    def bad(check, msg, **kw):
        R.append(dict(check=check, msg=msg, **K.clean(kw)))
    miss_cols = [c for c in MANIFEST_COLS if c not in M.columns]
    if miss_cols:
        bad('columns', 'required columns missing', missing=miss_cols)
        return _verdict(R, mode=mode, n_rows=len(M))
    if mode not in ('full', 'subset'):
        bad('mode', f'unknown mode {mode!r}')
        return _verdict(R, mode=mode, n_rows=len(M))
    M = M.copy()
    for c in NUMERIC:
        v = pd.to_numeric(M[c], errors='coerce')
        nb = M.index[v.isna() | ~np.isfinite(v.astype(float))]
        if len(nb):
            bad('types', f'{c}: non-numeric or non-finite values', rows=len(nb), first=int(nb[0]))
        M[c] = v
    for c in ('combo', 'surface', 'phase_label', 'raster_key', 'image_identity_fields', 'manifest_key', 'm3_digest', 'status'):
        nb = M.index[[_blank(x) or not isinstance(x, str) for x in M[c]]]
        if len(nb):
            bad('types', f'{c}: empty or not text', rows=len(nb), first=int(nb[0]))
    if R:
        return _verdict(R, mode=mode, n_rows=len(M))
    if ((M.rep - M.rep.round()).abs() > 0).any():
        bad('types', 'rep not integral')
        return _verdict(R, mode=mode, n_rows=len(M))
    M['rep'] = M.rep.round().astype(int)
    keys = [(r.combo, r.surface, r.phase_label, float(r.sigma), int(r.rep)) for r in M.itertuples()]
    cnt = Counter(keys); dup = [k for k, n in cnt.items() if n > 1]
    full = K.expected_keys()
    if mode == 'full':
        exp = full
    else:
        if not subset_keys:
            bad('subset', 'subset mode needs a pre-bound key set')
            return _verdict(R, mode=mode, n_rows=len(M))
        exp = {(a, b, c, float(d), int(e)) for a, b, c, d, e in subset_keys}
        if exp - full:
            bad('subset', 'pre-bound subset keys outside the fixed design', keys=sorted(exp - full)[:5])
    have = set(keys)
    if dup:
        bad('keys', 'duplicate keys', n=len(dup), first=list(dup[0]))
    if exp - have:
        bad('keys', 'expected keys missing', n=len(exp - have), first=list(sorted(exp - have)[0]))
    if have - exp:
        bad('keys', 'keys outside the expected set', n=len(have - exp), first=list(sorted(have - exp)[0]))
    if M.manifest_key.duplicated().any():
        bad('identity', 'duplicate manifest_key', n=int(M.manifest_key.duplicated().sum()))

    for r in M.itertuples():
        if r.combo not in K.DOT9 and r.combo not in K.LINE6:
            continue
        fam = K.family(r.combo)
        ok = r.family == fam and r.platform_surface == K.PLATFORM.get(r.surface)
        if fam == 'point':
            rad, pp, pq = K.DOT9[r.combo]
            ok = ok and K.num(r.radius_mm) == rad and K.num(r.p_parallel_mm) == pp and K.num(r.p_perp_mm) == pq
        else:
            w, s = K.LINE6[r.combo]
            ok = ok and K.num(r.line_width_mm) == w and K.num(r.line_pitch_mm) == s and K.num(r.orientation_deg) == 0.0
        if not ok:
            bad('design', 'design fields differ from the fixed design', key=[r.combo, r.surface, r.phase_label, r.sigma, r.rep])

    seeds = [seed_of(x) for x in M.noise_seed]
    M['seed_int'] = [s if s not in ('invalid',) else None for s in seeds]
    for i, (s, sg) in enumerate(zip(seeds, M.sigma)):
        if sg == 0.0 and s is not None:
            bad('seed', 'sigma 0 row carries a seed', row=i)
        if sg > 0.0 and (s is None or s == 'invalid'):
            bad('seed', 'sigma > 0 row without a valid integral seed', row=i)
    if (M.sigma == 0).any() and (M[M.sigma == 0].rep != 0).any():
        bad('rep', 'sigma 0 rows must be rep 0')
    if ((M.sigma > 0) & ~M.rep.isin([1, 2])).any():
        bad('rep', 'sigma > 0 rows must be rep 1 or 2')
    n = M[M.sigma > 0].groupby(['family', 'surface', 'phase_label', 'rep'])['seed_int'].nunique(dropna=False)
    if (n > 1).any():
        bad('seed', 'one family / surface / phase / rep must share one seed across combinations and sigma levels', groups=int((n > 1).sum()))
    if reg is not None:
        for r in M[M.sigma > 0].drop_duplicates(['family', 'surface', 'phase_label', 'rep']).itertuples():
            k = (r.family, r.surface, r.phase_label, int(r.rep))
            if k in reg['seed40'] and reg['seed40'][k] != r.seed_int:
                bad('seed', 'seed differs from the registered sigma-40 seed', group=list(k))

    for k, g in M.groupby(['combo', 'surface', 'phase_label']):
        if g.raster_key.nunique() > 1 or g.peak.nunique() > 1 or g.lit_px.nunique() > 1 or g.platform_surface.nunique() > 1 or g.m3_digest.nunique() > 1:
            bad('noise_axis', 'raster / peak / lit pixels / surface / M3 change along the noise axis', cell=list(k))
        if reg is not None and k in reg['cell_sigma0']:
            c0 = reg['cell_sigma0'][k]
            if (g.raster_key != c0['raster_key']).any() or (g.peak != c0['peak']).any() or (g.lit_px != c0['lit_px']).any():
                bad('noise_axis', 'cell differs from its registered sigma-0 image (raster / peak / lit pixels)', cell=list(k))

    for i, r in enumerate(M.to_dict('records')):
        f = image_identity_fields(r)
        if K.digest(f, 24) != r['image_identity_fields']:
            bad('identity', 'image identity does not match its physical fields', row=i, key=[r['combo'], r['surface'], r['phase_label'], r['sigma'], r['rep']])
        elif 'S5X-' + K.digest(dict(combo=r['combo'], **f), 24) != r['manifest_key']:
            bad('identity', 'manifest_key does not match combination + identity fields', row=i)
        ex = r['existing_image_key']
        has = not _blank(ex)
        if has != (r['status'] == 'reuse_existing') or (r['status'] == 'reuse_existing') != (float(r['sigma']) in K.EXISTING_SIGMAS) or r['status'] not in ('reuse_existing', 'to_image'):
            bad('status', 'reuse status / existing image key / sigma level inconsistent', row=i)
        elif has and reg is not None:
            j = reg['images'].get(ex)
            if j is None:
                bad('identity', 'existing image key not registered', row=i)
            elif j['key'] != (r['combo'], r['surface'], r['phase_label'], float(r['sigma']), int(r['rep'])) or any(K.clean(j['fields'].get(a)) != K.clean(f[a]) for a in f):
                bad('identity', 'existing image record differs from the manifest identity', row=i, image_key=ex)

    pk, lit, q0, itg = M.peak, M.lit_px, M.Q0, M.integral
    if ((pk <= 0) | (pk > 1.0)).any():
        bad('resource', 'peak outside (0, 1]', n=int(((pk <= 0) | (pk > 1.0)).sum()))
    if ((lit <= 0) | ((lit - lit.round()).abs() > 0) | (q0 != lit)).any():
        bad('resource', 'Q0 must equal the lit-pixel count of the raster (positive integer)', n=int(((lit <= 0) | (q0 != lit)).sum()))
    calc = lit * pk.astype(np.float32).astype(np.float64)
    if ((itg - calc).abs() > 1e-6).any():
        bad('resource', 'integral differs from lit pixels x float32(peak)', n=int(((itg - calc).abs() > 1e-6).sum()))
    over = itg > K.Q + K.INTEGRAL_TOL
    if over.any():
        bad('resource', f'integral above Q {K.Q} + tolerance {K.INTEGRAL_TOL:.3g}', n=int(over.sum()))
    lim_int = pk < 1.0
    if (lim_int & ((itg - K.Q).abs() > K.INTEGRAL_TOL)).any():
        bad('resource', 'integral-limited row (peak < 1) does not reach Q within tolerance', n=int((lim_int & ((itg - K.Q).abs() > K.INTEGRAL_TOL)).sum()))
    lt = M.limit_type.astype(str)
    if ((lim_int & ~lt.str.startswith('integral')) | (~lim_int & ~lt.str.startswith('peak'))).any():
        bad('resource', 'limit_type inconsistent with the peak')
    return _verdict(R, mode=mode, n_rows=len(M), n_expected=len(exp), n_missing=len(exp - have), n_extra=len(have - exp), n_duplicate_keys=len(dup),
                    integral_range=[float(itg.min()), float(itg.max())] if len(itg) else None, integral_tolerance=K.INTEGRAL_TOL)


GATE_FIELDS = ('state', 'n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid', 'id_correct', 'id_wrong', 'id_unconf', 'glob_Ctau', 'glob_Cvalid', 'glob_cov_interp',
               'loc_Ctau', 'loc_Cvalid', 'loc_cov_interp', 'glob_pt_rmse', 'glob_pt_p95', 'loc_pt_rmse', 'loc_pt_p95', 'record_complete', 'content_sha')


# Check joint-study completeness and consistency of reconstruction and identity metrics.
def gate_results(R: pd.DataFrame, expected: set, mode: str = 'full', view: str = '') -> dict:

    out = []

    def bad(check, msg, **kw):
        out.append(dict(check=check, msg=msg, **K.clean(kw)))
    miss = [c for c in ('combo', 'surface', 'phase_label', 'sigma', 'rep', 'method', 'image_key') + GATE_FIELDS if c not in R.columns]
    if miss:
        bad('columns', 'required result fields missing', missing=miss)
        return _verdict(out, view=view, mode=mode, n_rows=len(R))
    keys = [(r.combo, r.surface, r.phase_label, float(r.sigma), int(r.rep), r.method) for r in R.itertuples()]
    cnt = Counter(keys); dup = [k for k, n in cnt.items() if n > 1]; have = set(keys)
    if dup:
        bad('completeness', 'duplicate image x method records', n=len(dup), first=list(dup[0]))
    if expected - have:
        bad('completeness', 'expected image x method records missing', n=len(expected - have), first=list(sorted(expected - have)[0]))
    if have - expected:
        bad('completeness', 'records outside the expected set', n=len(have - expected), first=list(sorted(have - expected)[0]))
    img = R.groupby(['combo', 'surface', 'phase_label', 'sigma', 'rep']).image_key.nunique()
    if (img > 1).any():
        bad('pairing', 'methods of one observation point to different images', n=int((img > 1).sum()))

    def f(r, c):
        return K.num(r.get(c)) if c in r else None
    for r in R.to_dict('records'):
        k = [r['combo'], r['surface'], r['phase_label'], r['sigma'], r['rep'], r['method']]
        rc = r.get('record_complete')
        if not (isinstance(rc, (bool, np.bool_)) and bool(rc)) or _blank(r.get('content_sha')):
            bad('provenance', 'record not complete / not bound to stored content', key=k); continue
        if r['state'] not in ('ok', 'no_output'):
            bad('state', f"non-scientific state {r['state']!r} (technical failures are never counted as 0)", key=k); continue
        n_acc, n_val, gv, lv = (f(r, c) for c in ('n_accepted', 'n_valid', 'glob_n_valid', 'loc_n_valid'))
        c, w, u = (f(r, x) for x in ('id_correct', 'id_wrong', 'id_unconf'))
        if None in (n_acc, n_val, gv, lv, c, w, u):
            bad('fields', 'count field missing or non-numeric', key=k); continue
        if (r['state'] == 'no_output') != (n_val == 0):
            bad('state', 'state no_output must be exactly n_valid == 0', key=k)
        n_out = f(r, 'n_out')
        if not (0 <= n_val <= n_acc) or (n_out is not None and n_acc > n_out) or gv > n_val or lv > n_val:
            bad('counts', 'accepted / valid / domain-valid bookkeeping violated', key=k)
        if c + w + u != n_acc or (f(r, 'id_denominator') is not None and f(r, 'id_denominator') != n_acc):
            bad('identity', 'c + w + u differs from the accepted denominator', key=k)
        for d, dn in (('glob', gv), ('loc', lv)):
            ids = [f(r, f'{d}_id_{x}') for x in ('correct', 'wrong', 'unconf')]
            if None not in ids and sum(ids) != dn:
                bad('identity', f'{d}: domain identity counts differ from the domain valid points', key=k)
            rm, p9 = f(r, f'{d}_pt_rmse'), f(r, f'{d}_pt_p95')
            if dn > 0 and (rm is None or p9 is None or rm < 0 or p9 < 0):
                bad('point_error', f'{d}: valid points but undefined point error', key=k)
            if dn == 0 and (rm is not None or p9 is not None):
                bad('point_error', f'{d}: no valid point in the domain but a defined point error', key=k)
            t, v, i = f(r, f'{d}_Ctau'), f(r, f'{d}_Cvalid'), f(r, f'{d}_cov_interp')
            if None in (t, v, i) or not (0 <= t <= v + 1e-12 and v <= i + 1e-12 and i <= 1 + 1e-12):
                bad('coverage', f'{d}: 0 <= C_tau <= C_valid <= C_int <= 1 violated or undefined', key=k); continue
            if n_val == 0 and (t > 0 or v > 0 or i > 0):
                bad('coverage', f'{d}: no valid point in the cloud but positive coverage', key=k)
            if f'{d}_surf_rmse' in r:
                sd = f(r, f'{d}_surf_rmse') is not None
                if sd != (i > 0):
                    bad('surface_error', f'{d}: interpolated-surface error defined <=> interpolable coverage > 0 violated', key=k)
            if f'{d}_n_interp' in r and f(r, f'{d}_n_interp') is not None and (f(r, f'{d}_n_interp') > 0) != (i > 0):
                bad('surface_error', f'{d}: interpolable node count and coverage disagree', key=k)
        if n_val < 3 and (f(r, 'glob_cov_interp') or 0) > 0:
            bad('coverage', 'fewer than 3 valid points but positive global coverage (global_surface rule)', key=k)
        if 'sup_surf_rmse' in r and f(r, 'glob_Cvalid') is not None:
            if (f(r, 'sup_surf_rmse') is not None) != (f(r, 'glob_Cvalid') > 0):
                bad('surface_error', 'supported-node error defined <=> C_valid > 0 violated', key=k)
    ok_rows = R[R.state.isin(['ok', 'no_output'])]
    return _verdict(out, view=view, mode=mode, n_rows=len(R), n_expected=len(expected), n_images=int(R.image_key.nunique()),
                    n_no_output=int((ok_rows.state == 'no_output').sum()), n_valid_without_global_interp=int(((pd.to_numeric(ok_rows.n_valid, errors='coerce') > 0) & (pd.to_numeric(ok_rows.glob_cov_interp, errors='coerce') == 0)).sum()),
                    n_local_cov_without_local_points=int(((pd.to_numeric(ok_rows.loc_n_valid, errors='coerce') == 0) & (pd.to_numeric(ok_rows.loc_cov_interp, errors='coerce') > 0)).sum()))


def risks(c, w, u) -> tuple:
    return (K.num(w) / (K.num(c) + K.num(w)) if (K.num(c) or 0) + (K.num(w) or 0) > 0 else None,
            K.num(u) / (K.num(c) + K.num(w) + K.num(u)) if (K.num(c) or 0) + (K.num(w) or 0) + (K.num(u) or 0) > 0 else None)
