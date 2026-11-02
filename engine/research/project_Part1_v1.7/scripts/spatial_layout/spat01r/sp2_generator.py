# Generate layouts with a common movable set and protected boundary elements.

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import sp2_common as S2
import sp_geometry as G
import sp_dots_columns as DC
import sp_lines_table as LT1

MAN = S2.OUT / 'geometry_v2.jsonl'
SUM = S2.OUT / 'strategy_summary_v2.csv'
OVERLAP = S2.OUT / 'strategy_overlap_v2.csv'
SETS = S2.OUT / 'movable_sets_v2.json'
RASTERS = S2.OUT / 'rasters'


def dots_present(name: str, u: float, v: float):
    lay = DC.layout(*S2.DOT_BASELINES[name], u, v, None, 'u')
    return lay, sorted(set(int(i) for i in lay['ix']))


def lines_present(s_mm: float, phase: float):
    pat, meta = LT1.table_spec(s_mm, S2.W_MM, phase, None)
    return meta, sorted(int(k) for k in meta['olines']['ids'])


def common_movable_set(bl: dict) -> dict:

    fam, name, p0 = bl['family'], bl['name'], bl['p0_mm']; a_max = S2.J_MAX * p0
    phases = S2.registered_phases(fam)
    if fam == 'dots':
        start = DC.movable_columns(p0, a_max)['ix_movable']
        present = {ph['phase_id']: dots_present(name, ph['u'], ph['v'])[1] for ph in phases}
    else:
        ids_ref = lines_present(p0, S2.REF_PHASE_LINE)[1]
        start = G.line_movable_ids(p0, a_max, ids_ref)['movable']
        present = {ph['phase_id']: lines_present(p0, ph['phase_frac'])[1] for ph in phases}
    mov = list(start); steps = []
    while True:
        L, R = min(mov) - 1, max(mov) + 1
        fail_left = [pid for pid, pr in present.items() if L not in pr or min(mov) not in pr]
        fail_right = [pid for pid, pr in present.items() if R not in pr or max(mov) not in pr]
        if not fail_left and not fail_right:
            break
        if fail_left:
            steps.append(dict(drop=min(mov), side='left', failing_phases=fail_left)); mov = mov[1:]
        if fail_right:
            steps.append(dict(drop=max(mov), side='right', failing_phases=fail_right)); mov = mov[:-1]
        if len(mov) < 3:
            raise RuntimeError(f'{name}: movable set collapsed')
    return dict(baseline=name, family=fam, p0_mm=p0, a_max_mm=a_max, start_set=start, movable=mov, protected_L=min(mov) - 1, protected_R=max(mov) + 1, tightening_steps=steps, n_dropped=len(start) - len(mov),
                presence={pid: dict(n_present=len(pr), movable_all_present=all(i in pr for i in mov), L_present=(min(mov) - 1) in pr, R_present=(max(mov) + 1) in pr) for pid, pr in present.items()},
                rule='movable, L = min-1 and R = max+1 present at every registered phase (reference + pilot + hold-out); protected elements keep displacement 0')


def context_v2(bl: dict, ms: dict, a_mm: float) -> dict:
    fam, name, p0 = bl['family'], bl['name'], bl['p0_mm']
    if fam == 'lines':
        ctx = G.line_context(p0, 0.0)
        ctx.update(a_mm=a_mm, movable=list(ms['movable']), protected=[k for k in ctx['ids'] if k not in ms['movable']], omega_ids=(ms['protected_L'], ms['protected_R']),
                   omega_q_px=(ctx['q0'][ms['protected_L']], ctx['q0'][ms['protected_R']]), n_movable=len(ms['movable']),
                   n_protected_left=len([k for k in ctx['ids'] if k < min(ms['movable'])]), n_protected_right=len([k for k in ctx['ids'] if k > max(ms['movable'])]))
    else:
        ctx = G.dot_context(name, 0.0)
        pos0 = ctx['pos0']
        ctx.update(a_mm=a_mm, movable=list(ms['movable']), protected=[i for i in pos0 if i not in ms['movable']], omega_ix=(ms['protected_L'], ms['protected_R']), omega_u_px=(pos0[ms['protected_L']], pos0[ms['protected_R']]),
                   n_movable=len(ms['movable']), n_protected_left=len([i for i in pos0 if i < min(ms['movable'])]), n_protected_right=len([i for i in pos0 if i > max(ms['movable'])]))
    return ctx


def unit_vector(n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    u = rng.uniform(-1.0, 1.0, n); u = u - u.mean(); m = float(np.abs(u).max()) if n else 0.0
    scale = (1.0 / m) if m > 1.0 else 1.0; u = u * scale
    return u, dict(scaled=bool(scale < 1.0), scale=float(scale), zero_sum_residual=float(abs(u.sum())), max_abs=float(np.abs(u).max()) if n else 0.0)


def masks_v2(g_min_mm: float, h_mm: float, p0: float, h0: float) -> dict:
    s2 = bool(g_min_mm >= S2.GMIN_FRAC * p0 - 1e-12); s3 = bool(h_mm <= S2.HMAX_FRAC * h0 + 1e-12)
    return {'S1_free': True, 'S2_gmin_0.70': s2, 'S3_hmax_1.15': s3, 'S4_joint_0.70_1.15': bool(s2 and s3)}


def build_layout(bl: dict, disp_px: dict, phase=None):

    fam, name, p0 = bl['family'], bl['name'], bl['p0_mm']
    if fam == 'lines':
        ph = S2.REF_PHASE_LINE if phase is None else float(phase)
        pat, meta = LT1.table_spec(p0, S2.W_MM, ph, disp_px or None); return dict(pat=pat, meta=meta)
    u, v = S2.REF_PHASE_DOT if phase is None else phase
    lay = DC.layout(*S2.DOT_BASELINES[name], u, v, disp_px or None, 'u'); return dict(lay=lay, pat=DC.raster(lay['c']))


# Generate layout proposals within the common movable set and save accepted rasters.
def run_library(write_rasters: bool = True) -> dict:
    S2.ensure_dirs(); RASTERS.mkdir(parents=True, exist_ok=True)
    if MAN.exists():
        raise RuntimeError(f'{MAN} exists (no overwrite)')
    t0 = time.time(); sets = {}; rows_sum = []; overlap_rows = []; n_rec = 0; n_rebuild = 0
    with open(MAN, 'w', encoding='utf-8') as fh:
        for bl in S2.BASELINES:
            fam, name, p0 = bl['family'], bl['name'], bl['p0_mm']
            ms = common_movable_set(bl); sets[name] = ms; mov = ms['movable']
            ctx0 = context_v2(bl, ms, 0.0)
            d0 = G.line_descriptors(ctx0['meta_regular'], ctx0['pat_regular'], ctx0, {}) if fam == 'lines' else G.dot_descriptors(ctx0['lay_regular'], ctx0['pat_regular'], ctx0, {}, 'u')
            h0 = d0['h_mm']; ms['h0_mm'] = h0; ms['regular'] = dict(n_elements=d0['n_elements'], lit_px=d0['lit_px'], peak=d0['peak_equal_integral'], raster_sha256=d0['raster_sha256'], g_min_mm=d0['g_min_mm'], g_max_mm=d0['g_max_mm'])
            reg = dict(generator_version=S2.GENERATOR_VERSION, family=fam, baseline=name, p0_mm=p0, j=0.0, a_mm=0.0, block=-1, proposal=-1, proposal_id=f'{name}_v2_j0.00_regular', seed=None, unit_hash=None,
                       strategy_accept={s: True for s in S2.STRATEGIES}, h0_mm=h0, movable=mov, protected_L=ms['protected_L'], protected_R=ms['protected_R'], **d0)
            fh.write(json.dumps(S2.clean(reg), ensure_ascii=False) + '\n'); n_rec += 1; n_rebuild += 1
            if write_rasters:
                np.savez_compressed(RASTERS / f'{name}_v2_j0.00_regular.npz', pattern=(ctx0['pat_regular'] > 0).astype(np.uint8))
            S2.log(f'{name}: movable {len(mov)} (dropped {ms["n_dropped"]}: {ms["tightening_steps"]}), L/R {ms["protected_L"]}/{ms["protected_R"]}, h0 {h0:.4f}', 'geometry')
            ctx = {j: context_v2(bl, ms, j * p0) for j in S2.J_LEVELS}
            lib = {j: [] for j in S2.J_LEVELS}
            for block in range(S2.N_BLOCKS):
                for i in range(S2.N_PER_BLOCK):
                    seed = S2.seed_v2(bl['index'], block, i); rng = np.random.default_rng(seed)
                    u, uinfo = unit_vector(len(mov), rng); uhash = S2.sha_array(u)[:16]
                    for j in S2.J_LEVELS:
                        a_px = S2.px(j * p0); disp = {int(k): float(v) for k, v in zip(mov, u * a_px)}
                        built = build_layout(bl, disp)
                        desc = G.line_descriptors(built['meta'], built['pat'], ctx[j], disp) if fam == 'lines' else G.dot_descriptors(built['lay'], built['pat'], ctx[j], disp, 'u')
                        m = masks_v2(desc['g_min_mm'], desc['h_mm'], p0, h0)
                        pid = f'{name}_v2_j{j:.2f}_b{block}_p{i}'
                        rec = dict(generator_version=S2.GENERATOR_VERSION, family=fam, baseline=name, p0_mm=p0, j=j, a_mm=j * p0, block=block, proposal=i, proposal_id=pid, seed=seed, unit_hash=uhash, unit_info=uinfo,
                                   strategy_accept=m, h0_mm=h0, disp_px={str(k): round(v, 6) for k, v in disp.items()}, **desc)
                        fh.write(json.dumps(S2.clean(rec), ensure_ascii=False) + '\n'); n_rec += 1; n_rebuild += 1
                        lib[j].append(dict(block=block, proposal=i, pid=pid, masks=m, raster=desc['raster_sha256'], disp=disp, desc=desc))
                S2.log(f'{name} block {block} done ({time.time() - t0:.0f}s)', 'geometry')

            for j in S2.J_LEVELS:
                rows = lib[j]; s1 = {r['pid'] for r in rows}
                for s_ in S2.STRATEGIES:
                    acc = [r for r in rows if r['masks'][s_]]; accset = {r['pid'] for r in acc}
                    first = {}
                    for b in range(S2.N_BLOCKS):
                        c = [r for r in acc if r['block'] == b]; first[b] = (min(c, key=lambda r: r['proposal'])['proposal'] if c else None)
                    identical = [t for t in S2.STRATEGIES if t != s_ and {r['pid'] for r in rows if r['masks'][t]} == accset]
                    def agg(key, f):
                        vals = [r['desc'][key] for r in acc if r['desc'].get(key) is not None]; return float(f(vals)) if vals else None
                    rows_sum.append(dict(generator_version=S2.GENERATOR_VERSION, family=fam, baseline=name, p0_mm=p0, j=j, a_mm=j * p0, strategy=s_, n_proposals=len(rows), n_accepted=len(acc), acceptance_rate=len(acc) / len(rows),
                                         blocks_without_accepted=sum(1 for b in first if first[b] is None), first_accepted_index_mean=(float(np.mean([v for v in first.values() if v is not None])) if any(v is not None for v in first.values()) else None),
                                         first_accepted_index_max=(max(v for v in first.values() if v is not None) if any(v is not None for v in first.values()) else None), constraint_active=bool(accset != s1), identical_to=';'.join(identical),
                                         n_movable=len(mov), delta_rms_mm_mean=agg('delta_rms_mm', np.mean), delta_rms_mm_min=agg('delta_rms_mm', np.min), delta_rms_mm_max=agg('delta_rms_mm', np.max),
                                         g_min_mm_mean=agg('g_min_mm', np.mean), g_min_mm_min=agg('g_min_mm', np.min), g_max_mm_mean=agg('g_max_mm', np.mean), g_max_mm_max=agg('g_max_mm', np.max), h_mm_mean=agg('h_mm', np.mean), h_mm_max=agg('h_mm', np.max), h0_mm=h0,
                                         edge_clearance_min_mm_min=agg('edge_clearance_min_mm', np.min), lit_px_min=agg('lit_px', np.min), lit_px_max=agg('lit_px', np.max), peak_min=agg('peak_equal_integral', np.min), peak_max=agg('peak_equal_integral', np.max),
                                         n_components_min=agg('n_components', np.min), n_components_max=agg('n_components', np.max), raster_centre_residual_max_px=agg('raster_centre_residual_max_px', np.max)))

                per_block = {}
                for b in range(S2.N_BLOCKS):
                    for s_ in S2.STRATEGIES:
                        c = sorted([r for r in rows if r['block'] == b and r['masks'][s_]], key=lambda r: r['proposal'])
                        per_block[(b, s_)] = dict(accset=frozenset(r['pid'] for r in c), first=(c[0]['pid'] if c else None), raster=(c[0]['raster'] if c else None), n=len(c))
                        if c and b in S2.PILOT_BLOCKS and write_rasters and not (RASTERS / f"{c[0]['pid']}.npz").exists():
                            pat = build_layout(bl, c[0]['disp'])['pat']; n_rebuild += 1
                            assert S2.sha_array((pat > 0).astype(np.uint8)) == c[0]['raster']
                            np.savez_compressed(RASTERS / f"{c[0]['pid']}.npz", pattern=(pat > 0).astype(np.uint8))
                for b in range(S2.N_BLOCKS):
                    for ia, sa in enumerate(S2.STRATEGIES):
                        for sb in S2.STRATEGIES[ia + 1:]:
                            A_, B_ = per_block[(b, sa)], per_block[(b, sb)]
                            overlap_rows.append(dict(family=fam, baseline=name, j=j, block=b, strategy_a=sa, strategy_b=sb, n_accepted_a=A_['n'], n_accepted_b=B_['n'], same_accept_set=bool(A_['accset'] == B_['accset']),
                                                     first_a=A_['first'], first_b=B_['first'], same_first_proposal=bool(A_['first'] == B_['first'] and A_['first'] is not None), same_raster=bool(A_['raster'] == B_['raster'] and A_['raster'] is not None),
                                                     generation_failed=(A_['first'] is None or B_['first'] is None)))
    pd.DataFrame(rows_sum).to_csv(SUM, index=False, encoding='utf-8-sig'); pd.DataFrame(overlap_rows).to_csv(OVERLAP, index=False, encoding='utf-8-sig')
    S2.save_json(SETS, dict(generated_local=S2.now_local(), generator_version=S2.GENERATOR_VERSION, j_levels=list(S2.J_LEVELS), a_max_rule='0.20 p0', strategies=list(S2.STRATEGIES), gmin_frac=S2.GMIN_FRAC, hmax_frac=S2.HMAX_FRAC, seed_base=S2.SEED_BASE_V2, sets=sets))
    S2.ledger(geometry_rebuilds=n_rebuild, event=dict(kind='geometry_v2', records=n_rec))
    info = dict(n_records=n_rec, geometry_rebuilds=n_rebuild, seconds=round(time.time() - t0, 1)); S2.log(f'geometry v2 done {info}', 'geometry'); return info


if __name__ == '__main__':
    print(json.dumps(run_library()))
