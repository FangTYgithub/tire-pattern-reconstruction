# Generate bounded position proposals and evaluate spacing and hole constraints.

from __future__ import annotations

import json
import math
import time

import cv2
import numpy as np

import sp_common as S
import sp_dots_columns as DC
import sp_lines_table as LT
import sn_lib as SL

MAN = S.OUT / 'geometry_manifest.jsonl'
SUM = S.OUT / 'geometry_summary.csv'
RASTERS = S.OUT / 'rasters'


def proposal_vector(n: int, a_px: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    d = rng.uniform(-a_px, a_px, n)
    d = d - d.mean()
    m = float(np.abs(d).max()) if n else 0.0
    scale = (a_px / m) if m > a_px else 1.0
    d = d * scale
    return d, dict(scaled=bool(scale < 1.0), scale=float(scale), zero_sum_residual_px=float(abs(d.sum())), max_abs_px=float(np.abs(d).max()) if n else 0.0)


def accept_masks(g_min_mm: float, h_mm: float, p0_mm: float, h0_mm: float) -> dict:
    out = {'S1_free': True}
    for g in S.GMIN_FRACS:
        out[f'S2_gmin_{g:.2f}'] = bool(g_min_mm >= g * p0_mm - 1e-12)
    for h in S.HMAX_FRACS:
        out[f'S3_hmax_{h:.2f}'] = bool(h_mm <= h * h0_mm + 1e-12)
    for g in S.GMIN_FRACS:
        for h in S.HMAX_FRACS:
            out[f'S4_joint_{g:.2f}_{h:.2f}'] = bool(out[f'S2_gmin_{g:.2f}'] and out[f'S3_hmax_{h:.2f}'])
    return out


def _gap_stats(g_mm: np.ndarray) -> dict:
    if len(g_mm) == 0:
        return dict(g_min_mm=None, g_max_mm=None, g_p05_mm=None, g_p50_mm=None, g_p95_mm=None, g_argmax=None)
    return dict(g_min_mm=float(g_mm.min()), g_max_mm=float(g_mm.max()), g_p05_mm=float(np.percentile(g_mm, 5)), g_p50_mm=float(np.median(g_mm)), g_p95_mm=float(np.percentile(g_mm, 95)), g_argmax=int(np.argmax(g_mm)))


def _disp_stats(d_px: np.ndarray) -> dict:
    k = S.mmpp()
    if len(d_px) == 0:
        return dict(delta_mean_mm=0.0, delta_rms_mm=0.0, delta_max_mm=0.0)
    return dict(delta_mean_mm=float(d_px.mean() * k), delta_rms_mm=float(np.sqrt(np.mean(d_px ** 2)) * k), delta_max_mm=float(np.abs(d_px).max() * k))


def _raster_stats(pat: np.ndarray, pat_regular: np.ndarray | None) -> dict:
    b = (np.asarray(pat) > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(int) if n > 1 else np.zeros(0, int)
    lit = float(b.sum())
    out = dict(lit_px=int(lit), integral=lit, peak_equal_integral=(S.Q_BUDGET / lit) if lit > 0 else None, equal_integral_feasible=bool(lit > 0 and S.Q_BUDGET / lit <= S.PEAK_CAP + 1e-12),
               n_components=int(n - 1), component_area_min=int(areas.min()) if len(areas) else None, component_area_max=int(areas.max()) if len(areas) else None, raster_sha256=S.sha_array(b))
    if pat_regular is not None:
        out['xor_px_vs_regular'] = int((b != (np.asarray(pat_regular) > 0)).sum())
    return out


def line_movable_ids(s_mm: float, a_mm: float, ids_present: list[int], w_mm: float = S.W_MM, image_size_px: int = S.IMG) -> dict:
    s_px = S.px(s_mm); w_px = S.px(w_mm); a_px = S.px(a_mm); half = (image_size_px - 1) / 2.0
    mov = [int(k) for k in ids_present if (abs(k) + 0.5) * s_px <= half - w_px / 2.0 - a_px]
    prot = [int(k) for k in ids_present if int(k) not in mov]
    return dict(movable=mov, protected=prot, protected_left=[k for k in prot if k < 0], protected_right=[k for k in prot if k >= 0], a_px=a_px, rule='(|k| + 0.5) s_px <= (W-1)/2 - w_px/2 - a_px for every phase in [-0.5, 0.5]')


def line_context(s_mm: float, a_mm: float) -> dict:
    pat0, meta0 = LT.table_spec(s_mm, S.W_MM, S.REF_PHASE_LINE, None)
    ids = [int(k) for k in meta0['olines']['ids']]
    mv = line_movable_ids(s_mm, a_mm, ids)
    q0 = {int(k): meta0['olines']['segments'][str(k)]['q_px'] for k in ids}
    if not mv['protected_left'] or not mv['protected_right']:
        raise RuntimeError('no protected line on one side: Omega_g boundary undefined for this a')
    lo = max(mv['protected_left'], key=lambda k: q0[k]); hi = min(mv['protected_right'], key=lambda k: q0[k])
    return dict(s_mm=s_mm, p0_mm=s_mm, a_mm=a_mm, pat_regular=pat0, meta_regular=meta0, ids=ids, q0=q0, movable=mv['movable'], protected=mv['protected'], omega_ids=(lo, hi), omega_q_px=(q0[lo], q0[hi]),
                n_lines=len(ids), n_movable=len(mv['movable']), n_protected_left=len(mv['protected_left']), n_protected_right=len(mv['protected_right']), h0_mm=None)


def line_descriptors(meta: dict, pat: np.ndarray, ctx: dict, disp_px: dict) -> dict:
    o = meta['olines']; k = S.mmpp()
    ids = sorted(int(x) for x in o['ids']); q = {int(x): o['segments'][str(x)]['q_px'] for x in ids}
    lo, hi = ctx['omega_ids']
    inside = [x for x in ids if lo <= x <= hi]
    qs = np.array([q[x] for x in inside]); gaps = np.diff(qs) * k
    gs = _gap_stats(gaps)
    h_mm = gs['g_max_mm'] / 2.0 if gs['g_max_mm'] is not None else None
    loc = float(0.5 * (qs[gs['g_argmax']] + qs[gs['g_argmax'] + 1]) * k) if gs['g_argmax'] is not None else None
    widths = [o['segments'][str(x)]['mean_lit_width_px'] for x in ids]
    d = np.array([float(disp_px.get(x, 0.0)) for x in ctx['movable']])

    fam = LT.line_family_table(o); cl = LT.raster_centrelines_table(pat, fam)
    fam0 = LT.line_family_table(ctx['meta_regular']['olines']); cl0 = LT.raster_centrelines_table(ctx['pat_regular'], fam0)
    res = []
    for x in ctx['movable']:
        if x in cl and x in cl0 and cl[x]['q_offset_px'] is not None and cl0[x]['q_offset_px'] is not None:
            realised = (q[x] + cl[x]['q_offset_px']) - (ctx['q0'][x] + cl0[x]['q_offset_px'])
            res.append(realised - float(disp_px.get(x, 0.0)))
    res = np.array(res)
    out = dict(n_elements=len(ids), n_present_omega=len(inside), n_movable=len(ctx['movable']), n_protected_left=ctx['n_protected_left'], n_protected_right=ctx['n_protected_right'],
               ids_present_same_as_regular=bool(ids == sorted(ctx['ids'])), **_disp_stats(d), **gs, h_mm=h_mm, h_location_mm=loc,
               dmin_allpairs_mm=gs['g_min_mm'], edge_clearance_min_mm=(gs['g_min_mm'] - S.W_MM) if gs['g_min_mm'] is not None else None,
               lit_width_px_min=float(min(widths)) if widths else None, lit_width_px_max=float(max(widths)) if widths else None,
               raster_centre_residual_max_px=float(np.abs(res).max()) if len(res) else None, raster_centre_residual_rms_px=float(np.sqrt(np.mean(res ** 2))) if len(res) else None,
               n_clipped_lines=len(o['clipped_lines']), table_hash=o['table']['hash'])
    out.update(_raster_stats(pat, ctx['pat_regular']))
    return out


def dot_context(name: str, a_mm: float) -> dict:
    p_par, p_perp = S.DOT_BASELINES[name]
    lay0 = DC.layout(p_par, p_perp, *S.REF_PHASE_DOT, None, 'u')
    mv = DC.movable_columns(p_par, a_mm)
    present = lay0['ix_present']; movable = [i for i in present if i in mv['ix_movable']]; protected = [i for i in present if i not in movable]
    ixs, pos = DC.column_positions(lay0)
    pos_of = {int(i): float(p) for i, p in zip(ixs, pos)}
    pl = [i for i in protected if pos_of[i] < (S.IMG - 1) / 2.0]; pr = [i for i in protected if pos_of[i] >= (S.IMG - 1) / 2.0]
    if not pl or not pr:
        raise RuntimeError('no protected column on one side: Omega_g boundary undefined for this a')
    lo = max(pl, key=lambda i: pos_of[i]); hi = min(pr, key=lambda i: pos_of[i])
    pat0 = DC.raster(lay0['c'])
    return dict(name=name, p0_mm=p_par, p_perp_mm=p_perp, a_mm=a_mm, lay_regular=lay0, pat_regular=pat0, movable=movable, protected=protected, omega_ix=(lo, hi), omega_u_px=(pos_of[lo], pos_of[hi]),
                pos0=pos_of, n_dots=lay0['n'], n_columns=lay0['n_columns'], n_rows=len(lay0['rows_present']), n_movable=len(movable), n_protected_left=len(pl), n_protected_right=len(pr), h0_mm=None)


def _component_centroids(pat: np.ndarray):
    b = (np.asarray(pat) > 0).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(b, connectivity=8)
    return cents[1:], stats[1:, cv2.CC_STAT_AREA]


def dot_descriptors(lay: dict, pat: np.ndarray, ctx: dict, disp_px: dict, axis: str = 'u') -> dict:
    k = S.mmpp(); p_perp = ctx['p_perp_mm']
    ixs, pos = DC.column_positions(lay)
    pos_of = {int(i): float(p) for i, p in zip(ixs, pos)}
    lo, hi = ctx['omega_ix']
    inside = [int(i) for i in ixs if lo <= i <= hi]
    qs = np.array([pos_of[i] for i in inside]); gaps = np.diff(qs) * k
    gs = _gap_stats(gaps)
    out = dict(n_elements=lay['n'], n_columns=lay['n_columns'], n_rows=len(lay['rows_present']), n_present_omega=len(inside), n_movable=len(ctx['movable']), n_protected_left=ctx['n_protected_left'], n_protected_right=ctx['n_protected_right'],
               boundary_in=lay.get('boundary_in', 0), boundary_out=lay.get('boundary_out', 0), columns_same_as_regular=bool(list(ixs) == list(sorted(ctx['pos0']))))
    d = np.array([float(disp_px.get(i, 0.0)) for i in ctx['movable']])
    out.update(_disp_stats(d)); out.update(gs)
    if axis == 'u':
        h = 0.5 * math.sqrt(gs['g_max_mm'] ** 2 + p_perp ** 2) if gs['g_max_mm'] is not None else None
        loc = float(0.5 * (qs[gs['g_argmax']] + qs[gs['g_argmax'] + 1]) * k) if gs['g_argmax'] is not None else None
        dmin = min(p_perp, gs['g_min_mm']) if gs['g_min_mm'] is not None else None
        out.update(h_mm=h, h_location_mm=loc, h_method='analytic_rect_cell', dmin_allpairs_mm=dmin, edge_clearance_min_mm=(gs['g_min_mm'] - 2 * S.R_MM) if gs['g_min_mm'] is not None else None)
    else:
        from scipy.spatial import cKDTree
        c = lay['c']; dd, _ = cKDTree(c).query(c, k=2); dmin = float(dd[:, 1].min() * k)
        out.update(h_mm=None, h_location_mm=None, h_method='grid_pending', dmin_allpairs_mm=dmin, edge_clearance_min_mm=dmin - 2 * S.R_MM)

    cents, areas = _component_centroids(pat)
    res = []
    if len(cents) == lay['n']:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(cents).query(lay['c'])
        cents0, _ = _component_centroids(ctx['pat_regular']); lay0 = ctx['lay_regular']
        _, idx0 = cKDTree(cents0).query(lay0['c'])
        key0 = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(lay0['ix'], lay0['iy']))}
        ax = 0 if axis == 'u' else 1
        for j, (a, b) in enumerate(zip(lay['ix'], lay['iy'])):
            j0 = key0.get((int(a), int(b)))
            if j0 is not None and int(a) in ctx['movable']:
                res.append((cents[idx[j], ax] - cents0[idx0[j0], ax]) - float(disp_px.get(int(a), 0.0)))
    res = np.array(res)
    out.update(raster_centre_residual_max_px=float(np.abs(res).max()) if len(res) else None, raster_centre_residual_rms_px=float(np.sqrt(np.mean(res ** 2))) if len(res) else None,
               components_equal_dots=bool(len(cents) == lay['n']))
    out.update(_raster_stats(pat, ctx['pat_regular']))
    return out


def baselines() -> list[dict]:
    out = []
    for i, name in enumerate(S.DOT_BASELINES):
        out.append(dict(index=i, family='dots', name=name, p0_mm=S.DOT_BASELINES[name][0]))
    for i, name in enumerate(S.LINE_BASELINES):
        out.append(dict(index=10 + i, family='lines', name=name, p0_mm=S.LINE_BASELINES[name]))
    return out


# Generate position proposals and save spacing and hole-constraint descriptors.
def run_precheck(limit_blocks: int | None = None, write_rasters: bool = True) -> dict:
    S.ensure_dirs(); RASTERS.mkdir(parents=True, exist_ok=True)
    if MAN.exists():
        raise RuntimeError(f'{MAN} exists: SPAT-01 geometry pre-check already ran (continuation must not overwrite history)')
    t0 = time.time(); n_written = 0; summary_rows = []; regular_rows = []
    with open(MAN, 'w', encoding='utf-8') as fh:
        for bl in baselines():
            fam = bl['family']; name = bl['name']; p0 = bl['p0_mm']

            ctx0 = line_context(p0, 0.0) if fam == 'lines' else dot_context(name, 0.0)
            if fam == 'lines':
                d0 = line_descriptors(ctx0['meta_regular'], ctx0['pat_regular'], ctx0, {})
            else:
                d0 = dot_descriptors(ctx0['lay_regular'], ctx0['pat_regular'], ctx0, {}, 'u')
            h0 = d0['h_mm']
            reg = dict(family=fam, baseline=name, p0_mm=p0, j=0.0, a_mm=0.0, block=-1, proposal=-1, proposal_id=f'{name}_j0.00_regular', seed=None, strategy_accept={s: True for s in S.STRATEGIES}, h0_mm=h0, **d0)
            fh.write(json.dumps(S.clean(reg), ensure_ascii=False) + '\n'); n_written += 1
            regular_rows.append(dict(family=fam, baseline=name, p0_mm=p0, h0_mm=h0, n_elements=d0['n_elements'], lit_px=d0['lit_px'], peak_equal_integral=d0['peak_equal_integral'], raster_sha256=d0['raster_sha256'],
                                     h0_method=('g_max/2 (parallel lines, analytic)' if fam == 'lines' else '0.5*sqrt(g_max^2 + p_perp^2) (aligned rectangular cells, analytic)')))
            if write_rasters:
                np.savez_compressed(RASTERS / f'{name}_j0.00_regular.npz', pattern=(ctx0['pat_regular'] > 0).astype(np.uint8))
            S.log(f'{fam} {name}: regular h0 = {h0:.4f} mm, n = {d0["n_elements"]}, lit {d0["lit_px"]}, peak {d0["peak_equal_integral"]:.4f}', 'geometry')
            for ji, j in enumerate(S.J_LEVELS):
                if j == 0.0:
                    continue
                a_mm = j * p0; a_px = S.px(a_mm)
                ctx = line_context(p0, a_mm) if fam == 'lines' else dot_context(name, a_mm)
                ctx['h0_mm'] = h0
                mov = ctx['movable']; n_mov = len(mov)
                acc_count = {s: 0 for s in S.STRATEGIES}; first_acc = {s: [None] * S.N_BLOCKS for s in S.STRATEGIES}; rows = []
                nb = S.N_BLOCKS if limit_blocks is None else min(S.N_BLOCKS, limit_blocks)
                for block in range(nb):
                    for i in range(S.N_PER_BLOCK):
                        seed = S.seed_for(bl['index'], ji, block, i); rng = np.random.default_rng(seed)
                        d, pinfo = proposal_vector(n_mov, a_px, rng)
                        disp = {int(k): float(v) for k, v in zip(mov, d)}
                        if fam == 'lines':
                            pat, meta = LT.table_spec(p0, S.W_MM, S.REF_PHASE_LINE, disp)
                            desc = line_descriptors(meta, pat, ctx, disp)
                        else:
                            lay = DC.layout(ctx['p0_mm'], ctx['p_perp_mm'], *S.REF_PHASE_DOT, disp, 'u')
                            pat = DC.raster(lay['c'])
                            desc = dot_descriptors(lay, pat, ctx, disp, 'u')
                        masks = accept_masks(desc['g_min_mm'], desc['h_mm'], p0, h0)
                        pid = f'{name}_j{j:.2f}_b{block}_p{i}'
                        rec = dict(family=fam, baseline=name, p0_mm=p0, j=j, a_mm=a_mm, block=block, proposal=i, proposal_id=pid, seed=seed, proposal_info=pinfo, strategy_accept=masks, h0_mm=h0,
                                   disp_px={str(k): round(v, 6) for k, v in disp.items()}, **desc)
                        fh.write(json.dumps(S.clean(rec), ensure_ascii=False) + '\n'); n_written += 1
                        rows.append(rec)
                        for s_, ok in masks.items():
                            if ok:
                                acc_count[s_] += 1
                                if first_acc[s_][block] is None:
                                    first_acc[s_][block] = i
                                    if write_rasters and j in (0.10, 0.20) and sum(1 for b_ in first_acc[s_] if b_ is not None) <= 3:
                                        np.savez_compressed(RASTERS / f'{pid}__{s_}.npz', pattern=(pat > 0).astype(np.uint8))
                n_prop = nb * S.N_PER_BLOCK
                s1 = {r['proposal_id'] for r in rows if r['strategy_accept']['S1_free']}
                for s_ in S.STRATEGIES:
                    acc = [r for r in rows if r['strategy_accept'][s_]]
                    accset = {r['proposal_id'] for r in acc}
                    identical = [t for t in S.STRATEGIES if t != s_ and {r['proposal_id'] for r in rows if r['strategy_accept'][t]} == accset]
                    def agg(key, f):
                        vals = [r[key] for r in acc if r.get(key) is not None]
                        return float(f(vals)) if vals else None
                    summary_rows.append(dict(family=fam, baseline=name, p0_mm=p0, j=j, a_mm=a_mm, strategy=s_, n_proposals=n_prop, n_blocks=nb, n_accepted=len(acc), acceptance_rate=len(acc) / n_prop,
                                             blocks_without_accepted=int(sum(1 for b_ in first_acc[s_][:nb] if b_ is None)), first_accepted_index_mean=(float(np.mean([b_ for b_ in first_acc[s_][:nb] if b_ is not None])) if any(b_ is not None for b_ in first_acc[s_][:nb]) else None),
                                             first_accepted_index_max=(max(b_ for b_ in first_acc[s_][:nb] if b_ is not None) if any(b_ is not None for b_ in first_acc[s_][:nb]) else None),
                                             constraint_active=bool(accset != s1), identical_to=';'.join(identical), n_movable=n_mov, n_elements=(acc[0]['n_elements'] if acc else None),
                                             delta_rms_mm_mean=agg('delta_rms_mm', np.mean), delta_rms_mm_min=agg('delta_rms_mm', np.min), delta_rms_mm_max=agg('delta_rms_mm', np.max),
                                             g_min_mm_mean=agg('g_min_mm', np.mean), g_min_mm_min=agg('g_min_mm', np.min), g_max_mm_mean=agg('g_max_mm', np.mean), g_max_mm_max=agg('g_max_mm', np.max),
                                             h_mm_mean=agg('h_mm', np.mean), h_mm_max=agg('h_mm', np.max), h0_mm=h0, dmin_allpairs_mm_min=agg('dmin_allpairs_mm', np.min), edge_clearance_min_mm_min=agg('edge_clearance_min_mm', np.min),
                                             lit_px_min=agg('lit_px', np.min), lit_px_max=agg('lit_px', np.max), peak_min=agg('peak_equal_integral', np.min), peak_max=agg('peak_equal_integral', np.max),
                                             n_components_min=agg('n_components', np.min), n_components_max=agg('n_components', np.max), xor_px_min=agg('xor_px_vs_regular', np.min),
                                             raster_centre_residual_max_px=agg('raster_centre_residual_max_px', np.max), boundary_in_max=agg('boundary_in', np.max) if fam == 'dots' else 0, boundary_out_max=agg('boundary_out', np.max) if fam == 'dots' else 0,
                                             scaled_fraction=(float(np.mean([r['proposal_info']['scaled'] for r in acc])) if acc else None)))
                S.log(f'{fam} {name} j {j:.2f}: ' + ', '.join(f"{s_}={acc_count[s_]}" for s_ in S.STRATEGIES) + f' ({time.time() - t0:.0f}s)', 'geometry')
    import pandas as pd
    df = pd.DataFrame(summary_rows); df.to_csv(SUM, index=False, encoding='utf-8-sig')
    S.save_json(S.OUT / 'geometry_regular_baselines.json', dict(generated_local=S.now_local(), rows=regular_rows, j_levels=list(S.J_LEVELS), n_blocks=S.N_BLOCKS, n_per_block=S.N_PER_BLOCK, gmin_fracs=list(S.GMIN_FRACS), hmax_fracs=list(S.HMAX_FRACS), seed_base=S.SEED_BASE))
    info = dict(n_layout_records=n_written, seconds=round(time.time() - t0, 1), manifest=S.rel(MAN), summary=S.rel(SUM))
    S.log(f'geometry pre-check done: {info}', 'geometry')
    return info


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--limit-blocks', type=int, default=None); ap.add_argument('--no-rasters', action='store_true')
    a = ap.parse_args()
    print(json.dumps(run_precheck(a.limit_blocks, not a.no_rasters)))
