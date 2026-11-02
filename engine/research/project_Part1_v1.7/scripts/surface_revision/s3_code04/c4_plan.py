# Enumerate encoding images and the required decoding arms.

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import c4_common as K
import c4_parity as PA
import c4_protocol as PR
import c3_raster as RS

LAYOUTS = dict(dots=('R5x4', 'R6x4', 'R8x4'), lines=('4.45', '5.6', '8'))
CODED = ('weak', 'mid', 'strong')


def phases() -> dict:
    D = pd.read_csv(K.O1 / 'results' / 'precheck_new_phases.csv')
    return dict(dots=[dict(phase_id=r.phase_id, u=float(r.u), v=float(r.v), phase_frac=None) for r in D[D.family == 'dots'].itertuples()],
                lines=[dict(phase_id=r.phase_id, u=None, v=None, phase_frac=float(r.phase_frac)) for r in D[D.family == 'lines'].itertuples()])


def _mm():
    import ep_patterns as EP
    return EP.mmpp()


def _dot_proj_classes_ok(B: dict, lev: dict, pattern: str) -> bool:

    import ep_detect as ED
    import ep_patterns as EP
    import m4_reconstruction as M4
    import r5_detect as RD
    rs, rl = lev['dots']['radii'][pattern]; pat = np.asarray(B['binary'], np.float32)
    areas = M4.pattern_features(pat, 'coded_dots', return_areas=True)[2]; cfg = ED.coded_cfg([rs, rl], 2, EP.mmpp()); pcls = M4.projector_size_classes(areas, cfg)
    pdot, _ = RD.projector_dot_index(pat, np.asarray(B['centres'], np.float64))
    return bool(len(pcls) == len(B['classes']) and np.array_equal(np.asarray(pcls), np.asarray(B['classes'])[pdot]))


def _line_width_classes_ok(B: dict) -> dict:

    import lc_pattern as LP
    ol = B['meta_full']['olines']; ids = np.asarray(ol['ids'], np.int64); pat = np.asarray(B['binary']) > 0; wpx = ol['code']['widths_px']
    u = np.array([ol['segments'][str(int(k))]['start_uv'][0] for k in ids], float); half = 0.5 * ol['spacing_px']
    cls = LP.class_of_id(ids); wdes = np.where(cls == 1, wpx['wide'], wpx['narrow'])
    clipped = (u - wdes / 2 < -0.5) | (u + wdes / 2 > pat.shape[1] - 0.5)
    widths = []
    for x in u:
        lo, hi = int(max(0, np.floor(x - half))), int(min(pat.shape[1], np.ceil(x + half))); s_ = pat[:, lo:hi].sum(axis=1)
        widths.append(float(s_[s_ > 0].mean()) if (s_ > 0).any() else np.nan)
    widths = np.array(widths); ok_idx = np.isfinite(widths) & ~clipped
    wide_ok = bool((widths[ok_idx & (cls == 1)].min() > widths[ok_idx & (cls == 0)].max()) if (ok_idx & (cls == 1)).any() and (ok_idx & (cls == 0)).any() else False)
    return dict(ok=wide_ok, n_negative_ids=int((ids < 0).sum()), negative_ids_classes=sorted(set(int(c) for c in cls[ids < 0])), parity_wide=ol['code']['parity_wide'],
                clipped_wide=int((clipped & (cls == 1)).sum()), clipped_narrow=int((clipped & (cls == 0)).sum()))


def check_one(fam: str, lay: str, pattern: str, ph: dict, lev: dict) -> dict:
    B0, B1, B2 = (PA.build(fam, lay, pattern, ph, lev, p) for p in (0, 1, 0))
    old = RS.build(fam, lay, pattern, ph, lev)
    B2 = RS.build(fam, lay, pattern, PA.effective_phase(fam, PA.effective_phase(fam, ph, 1), 1), lev)
    mm = _mm(); r = dict(family=fam, layout=lay, pattern=pattern, geometry_phase=ph['phase_id'])
    r['parity0_equals_old'] = bool(np.array_equal(B0['binary'], old['binary']) and (np.array_equal(PA.class_table(fam, B0), PA.class_table(fam, old))))
    if fam == 'dots':
        c0, c1 = np.asarray(B0['centres']), np.asarray(B1['centres']); k0, k1 = np.asarray(B0['classes']), np.asarray(B1['classes'])
        o0, o1 = np.lexsort((c0[:, 0], c0[:, 1])), np.lexsort((c1[:, 0], c1[:, 1]))
        same_n = len(c0) == len(c1); dmax = float(np.abs(c0[o0] - c1[o1]).max() * mm) if same_n and len(c0) else np.inf
        r.update(n_features=len(c0), n_features_p1=len(c1), centre_max_diff_mm=dmax, centres_equal=bool(same_n and dmax <= 1e-9),
                 classes_swapped=bool(same_n and np.array_equal(k0[o0], 1 - k1[o1])),
                 double_swap_restores=bool(np.array_equal(B2['binary'], B0['binary']) and np.array_equal(PA.class_table(fam, B2)[:, 2], PA.class_table(fam, B0)[:, 2])
                                           and float(np.abs(PA.class_table(fam, B2)[:, :2] - PA.class_table(fam, B0)[:, :2]).max() * mm) <= 1e-9),
                 large_is_class1_p0=_dot_proj_classes_ok(B0, lev, pattern), large_is_class1_p1=_dot_proj_classes_ok(B1, lev, pattern),
                 n_class0_p0=int((k0 == 0).sum()), n_class1_p0=int((k0 == 1).sum()), n_class0_p1=int((k1 == 0).sum()), n_class1_p1=int((k1 == 1).sum()), id_map='same centre order (lexsort by v, u)')
    else:
        t0, t1 = PA.class_table(fam, B0), PA.class_table(fam, B1)
        o0, o1 = np.argsort(t0[:, 1]), np.argsort(t1[:, 1]); same_n = len(t0) == len(t1)
        dmax = float(np.abs(t0[o0, 1] - t1[o1, 1]).max() * mm) if same_n and len(t0) else np.inf
        off = np.unique(t1[o1, 0] - t0[o0, 0]) if same_n else np.array([np.nan])
        w0, w1 = _line_width_classes_ok(B0), _line_width_classes_ok(B1)
        r.update(n_features=len(t0), n_features_p1=len(t1), centre_max_diff_mm=dmax, centres_equal=bool(same_n and dmax <= 1e-9),
                 classes_swapped=bool(same_n and np.array_equal(t0[o0, 2], 1 - t1[o1, 2])), double_swap_restores=bool(np.array_equal(B2['binary'], B0['binary']) and np.array_equal(PA.class_table(fam, B2)[:, 1:], PA.class_table(fam, B0)[:, 1:])),
                 large_is_class1_p0=w0['ok'], large_is_class1_p1=w1['ok'], negative_ids_p1=w1['n_negative_ids'], negative_ids_classes_p1=';'.join(map(str, w1['negative_ids_classes'])),
                 clipped_wide_p0=w0['clipped_wide'], clipped_narrow_p0=w0['clipped_narrow'], clipped_wide_p1=w1['clipped_wide'], clipped_narrow_p1=w1['clipped_narrow'],
                 n_class0_p0=int((t0[:, 2] == 0).sum()), n_class1_p0=int((t0[:, 2] == 1).sum()), n_class0_p1=int((t1[:, 2] == 0).sum()), n_class1_p1=int((t1[:, 2] == 1).sum()),
                 id_map=('parity1 id = parity0 id ' + (f'{int(off[0]):+d}' if len(off) == 1 else 'NOT a constant offset')), id_offset_constant=bool(len(off) == 1))
    q0, q1 = float(np.asarray(B0['binary'], np.float64).sum()), float(np.asarray(B1['binary'], np.float64).sum())
    r.update(Q0_p0=q0, Q0_p1=q1, Q0_p1_minus_p0=q1 - q0, raster_differs=bool(not np.array_equal(B0['binary'], B1['binary'])), class_map_fp_p0=PA.class_map_fp(fam, B0), class_map_fp_p1=PA.class_map_fp(fam, B1))
    r['ok'] = bool(r['parity0_equals_old'] and r['centres_equal'] and r['classes_swapped'] and r['double_swap_restores'] and r['large_is_class1_p0'] and r['large_is_class1_p1']
                   and r.get('id_offset_constant', True))
    return r


def _job(a):
    return check_one(*a)


def checks(workers: int = 14) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    lev = RS.levels(); PH = phases(); jobs = [(fam, lay, pat, ph, lev) for fam in ('dots', 'lines') for lay in LAYOUTS[fam] for pat in CODED for ph in PH[fam]]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_job, jobs))
    D = pd.DataFrame(rows); K.save_csv(D, K.RESULTS / 'A_parity_checks.csv')
    s = dict(n=len(D), ok=int(D.ok.sum()), failures=D[~D.ok][['family', 'layout', 'pattern', 'geometry_phase']].to_dict('records'), centre_max_diff_mm=float(D.centre_max_diff_mm.max()),
             parity0_equals_old=int(D.parity0_equals_old.sum()), classes_swapped=int(D.classes_swapped.sum()), double_swap=int(D.double_swap_restores.sum()),
             semantic_class1_large_or_wide=int((D.large_is_class1_p0 & D.large_is_class1_p1).sum()), raster_differs=int(D.raster_differs.sum()),
             Q0_change_range={f: [float(g.Q0_p1_minus_p0.min()), float(g.Q0_p1_minus_p0.max())] for f, g in D.groupby('family')},
             line_negative_ids_covered=int(D[D.family == 'lines'].negative_ids_p1.gt(0).sum()) if 'negative_ids_p1' in D else 0,
             line_clipped_edge_lines={f'p{p}': dict(wide=int(D[f'clipped_wide_p{p}'].sum()), narrow=int(D[f'clipped_narrow_p{p}'].sum())) for p in (0, 1)} if 'clipped_wide_p0' in D else {},
             check_note='semantic width check on non-clipped lines; the first run without the clipping exclusion flagged 6 line configurations (all from one cut wide line at the projector border)',
             all_ok=bool(D.ok.all()))
    PR.checks_add('A_parity', s)
    return s


# Expand the encoding matrix across registered phases and noise realizations.
def manifest() -> dict:
    import rdy_inputs as IN
    import s3c_formal as F
    old = pd.read_csv(K.O3 / 'formal' / 'manifest.csv', low_memory=False); old['layout'] = old.layout.astype(str)
    lev = RS.levels(); PH = {f: {p['phase_id']: p for p in v} for f, v in phases().items()}; m3 = IN.m3_digest()
    rast, res_rows, cmap = {}, [], []
    for (fam, lay, pat, pid), g in old.groupby(['family', 'layout', 'pattern', 'phase_id']):
        par = 0 if pat == 'plain' else K.PARITY_OF_PHASE[pid]
        B = PA.build(fam, lay, pat, PH[fam][pid], lev, par); rk = RS.raster_key(B['binary'])
        Q0 = float(np.asarray(B['binary'], np.float64).sum()); v = min(1.0, K.Q_BUDGET / Q0); I = (np.asarray(B['binary'], np.float32) * np.float32(v)).astype(np.float32)
        Q = float(I.sum(dtype=np.float64)); rel = abs(Q - K.Q_BUDGET) / K.Q_BUDGET if v < 1.0 else None
        ct = PA.class_table(fam, B); fp = PA.class_map_fp(fam, B)
        rast[(fam, lay, pat, pid)] = dict(raster_key=rk, Q0=Q0, v_scale=v, peak_f32=float(np.float32(v)), Q=Q, usage=Q / K.Q_BUDGET, limit=('peak_limited' if v >= 1 else 'integral_limited'),
                                          budget_rel_err=rel, budget_ok=bool(float(np.float32(v)) <= 1.0 and (v >= 1.0 or rel <= 1e-6)), code_parity=par, class_map_fp=fp,
                                          n_class0=int((ct[:, 2] == 0).sum()), n_class1=int((ct[:, 2] == 1).sum()), n_features=int(len(ct)))
        for row in ct:
            cmap.append(dict(family=fam, layout=lay, pattern=pat, geometry_phase=pid, code_parity=par, a=row[0], b=row[1], cls=int(row[2])))
    rows = []
    for r in old.to_dict('records'):
        x = rast[(r['family'], r['layout'], r['pattern'], r['phase_id'])]
        n = dict(r); n.update({k: x[k] for k in ('raster_key', 'Q0', 'v_scale', 'peak_f32', 'Q', 'usage', 'limit', 'budget_ok', 'code_parity', 'class_map_fp')})
        n['geometry_phase'] = r['phase_id']; n['m3_digest'] = m3
        n['image_key'] = K.K3.image_key(n['raster_key'], n['peak_f32'], n['surface'], n['sigma'], K.norm_seed(n['seed']), m3)
        n['pair_key'] = K.digest(dict(f=r['family'], s=r['surface'], l=r['layout'], g=r['phase_id'], sg=float(r['sigma']), rep=int(r['rep'])), 16)
        rows.append(n)
    M = pd.DataFrame(rows)
    ref = M[M.sigma == 0].set_index(['family', 'surface', 'layout', 'pattern', 'phase_id']).image_key
    M['ref_image_key'] = [ref.get((r.family, r.surface, r.layout, r.pattern, r.phase_id)) if r.family == 'dots' else None for r in M.itertuples()]

    R = []
    for (fam, lay, pid), g in M.drop_duplicates(['family', 'layout', 'pattern', 'phase_id']).groupby(['family', 'layout', 'phase_id']):
        pl = g[g.pattern == 'plain'].iloc[0]
        for r in g.itertuples():
            x = rast[(fam, lay, r.pattern, pid)]; resid = x['Q0'] / pl.Q0 - 1.0
            R.append(dict(family=fam, layout=lay, geometry_phase=pid, pattern=r.pattern, code_parity=x['code_parity'], raster_key=x['raster_key'], Q0=x['Q0'], Q0_plain=float(pl.Q0), residual=resid,
                          residual_limit=K.RESID_LIMIT[fam], residual_ok=bool(abs(resid) <= K.RESID_LIMIT[fam]), peak_f32=x['peak_f32'], usage=x['usage'], budget_rel_err=x['budget_rel_err'],
                          budget_ok=x['budget_ok'], n_class0=x['n_class0'], n_class1=x['n_class1'], n_features=x['n_features'], class_map_fp=x['class_map_fp']))
    R = pd.DataFrame(R)

    used_r = set(pd.read_csv(K.ROOT / 'outputs/surface_revision/s12_resource01/inputs.csv', low_memory=False, usecols=['raster_key']).raster_key)
    used_r |= set(pd.read_csv(K.O2 / 'results/plan_manifest.csv', low_memory=False).raster_key)
    for f in (K.RK.RUN / 'images').glob('*.json'):
        try:
            used_r.add(K.load_json(f)['identity_fields']['raster'])
        except Exception:
            pass
    used_i = set(pd.read_csv(K.ROOT / 'outputs/surface_revision/s12_resource01/inputs.csv', low_memory=False, usecols=['input_key']).input_key) | set(pd.read_csv(K.O2 / 'results/plan_manifest.csv', low_memory=False).image_key)
    seeds_old = old.set_index(['family', 'surface', 'layout', 'pattern', 'phase_id', 'sigma', 'rep']).seed
    K.FORMAL.mkdir(parents=True, exist_ok=True); K.save_csv(M, K.FORMAL / 'manifest.csv'); K.save_csv(R, K.RESULTS / 'A_resources.csv')
    K.save_csv(pd.DataFrame(cmap), K.FORMAL / 'class_map.csv')
    even = M[M.phase_id.isin(['N0', 'N2', 'N4', 'N6'])]; o_even = old[old.phase_id.isin(['N0', 'N2', 'N4', 'N6'])]
    chk = dict(rows=len(M), unique_images=int(M.image_key.nunique()), rasters=int(M.raster_key.nunique()), sigma0=int((M.sigma == 0).sum()), expected=dict(rows=6336, rasters=192, sigma0=576),
               counts_ok=bool(len(M) == 6336 and M.image_key.nunique() == 6336 and M.raster_key.nunique() == 192 and (M.sigma == 0).sum() == 576),
               parity_map=M.drop_duplicates(['phase_id', 'code_parity'])[['phase_id', 'pattern', 'code_parity']].sort_values('phase_id').query("pattern != 'plain'")[['phase_id', 'code_parity']].drop_duplicates().to_dict('records'),
               plain_parity_all_zero=bool((M[M.pattern == 'plain'].code_parity == 0).all()),
               even_phases_unchanged=bool(set(even.image_key) == set(o_even.image_key)), odd_coded_changed=int((~M[(M.pattern != 'plain') & M.phase_id.isin(['N1', 'N3', 'N5', 'N7'])].image_key.isin(set(old.image_key))).sum()),
               budget_ok=bool(R.budget_ok.all()), budget_rel_err_max=float(R.budget_rel_err.dropna().max()), residual_ok=bool(R.residual_ok.all()),
               residual_max={f'{f}/{l}': float(g.residual.abs().max()) for (f, l), g in R.groupby(['family', 'layout'])}, residual_failures=R[~R.residual_ok][['family', 'layout', 'geometry_phase', 'pattern', 'code_parity', 'residual']].to_dict('records'),
               refs_complete=bool(M[(M.family == 'dots')].ref_image_key.notna().all()), refs_same_parity=bool(all(M.set_index('image_key').loc[r.ref_image_key, 'code_parity'] == r.code_parity for r in M[M.family == 'dots'].itertuples())),
               sigma0_self_reference=bool((M[(M.family == 'dots') & (M.sigma == 0)].ref_image_key == M[(M.family == 'dots') & (M.sigma == 0)].image_key).all()),
               pairing_complete=bool(M.groupby('pair_key').pattern.nunique().eq(4).all()), pairing_same_seed=bool(M.groupby('pair_key').seed.nunique(dropna=False).eq(1).all()),
               seeds_unchanged=bool(all((K.norm_seed(seeds_old.get((r.family, r.surface, r.layout, r.pattern, r.phase_id, r.sigma, r.rep))) == K.norm_seed(r.seed)) for r in M.itertuples())),
               raster_overlap_used=sorted(set(M.raster_key) & used_r)[:10], input_overlap_used=sorted(set(M.image_key) & used_i)[:10],
               limitation='balanced placement sample (4 blocks per placement); no estimate of the class-swap effect at fixed geometry; not an exhaustive code-phase coverage',
               manifest_sha256=K.sha_file(K.FORMAL / 'manifest.csv'))
    chk['ok'] = bool(chk['counts_ok'] and chk['budget_ok'] and chk['residual_ok'] and chk['refs_complete'] and chk['refs_same_parity'] and chk['sigma0_self_reference'] and chk['pairing_complete']
                     and chk['pairing_same_seed'] and chk['seeds_unchanged'] and not chk['raster_overlap_used'] and not chk['input_overlap_used'] and chk['even_phases_unchanged'] and chk['plain_parity_all_zero'])
    PR.checks_add('A_manifest', chk)
    return chk


def freeze() -> dict:

    deps = K.load_json(K.DEPS_FILE); man = K.FORMAL / 'manifest.csv'
    fz = dict(kind='formal package freeze S3-CODE-04 (coding parity, analysis gate; no permit, imaging needs a later release)', closure=K.freeze_closure(extra=[man, K.FORMAL / 'class_map.csv', K.DEPS_FILE], deps=deps),
              deps=deps, fingerprints={l: K.fingerprint(l, deps=deps) for l in ('decode', 'eval', 'diag')}, manifest_sha256=K.sha_file(man), class_map_sha256=K.sha_file(K.FORMAL / 'class_map.csv'),
              layering='figures / analysis / collection / runner: package closure only (never in decode / eval / diag fingerprints); configs + class map: decode fingerprint',
              permit_fields=['task_file', 'task_sha256', 'released_by', 'manifest_sha256', 'closure_sha256'], t_local=K.now_local())
    fz['closure_sha256'] = K.digest(fz['closure'], 64)
    K.save_json(K.FORMAL / 'frozen.json', fz)
    return {k: v for k, v in fz.items() if k not in ('closure', 'deps')}


if __name__ == '__main__':
    w = sys.argv[1]
    if w == 'freeze':
        print(json.dumps(K.clean(freeze()), indent=1)[:2000])
    if w == 'checks':
        print(json.dumps(K.clean(checks()), indent=1)[:3000])
    elif w == 'manifest':
        print(json.dumps(K.clean(manifest()), indent=1)[:4000])
