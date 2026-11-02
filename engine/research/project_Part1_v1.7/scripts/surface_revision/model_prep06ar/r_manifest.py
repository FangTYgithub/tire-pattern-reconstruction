# Enumerate candidate, environment, phase and noise combinations.

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import r_common as K
import r_inputs as RI

K5 = K.K5
HALF_MM = 46.08
SURF_SID = {'B_curv': 'B_curv', 'M10_curv': 'A_curv', 'M24_curv': 'C_curv'}
SURF_IDX = {'B_curv': 0, 'M10_curv': 1, 'M24_curv': 2}
ENVS = dict(NOMINAL=(1.0, ('const', 1.0)), BLUR1_5=(1.5, ('const', 1.0)), BLUR2=(2.0, ('const', 1.0)), BLUR2_5=(2.5, ('const', 1.0)), BLUR3=(3.0, ('const', 1.0)),
            G05=(1.0, ('const', 0.5)), G05_BLUR2=(2.0, ('const', 0.5)), GRAD_Y_REV=(1.0, ('gradient_y_reversed', 0.75, 0.25)))
LINE_SET = ('L_w0.5_s8', 'L_w0.75_s8', 'L_w1_s8', 'L_w1.25_s8', 'L_w1.5_s8')
DOT_SET = ('D_r1_p8x2.5', 'D_r1_p8x3', 'D_r1_p8x4', 'D_r1_p8x6', 'D_r1_p8x8', 'D_r1.5_p8x3', 'D_r1.5_p8x4', 'D_r1.5_p8x6')
REPRESENTATIVES = ('D_r1_p8x6', 'D_r1.5_p8x4', 'L_w0.5_s8', 'L_w1.5_s8')
SEED_BASE = dict(point=9_870_000, line=9_880_000)
HOLD_BASE0 = dict(point=9_970_000, line=9_980_000)


def fam_of(cfg: str) -> str:
    return 'point' if cfg.startswith('D_') else 'line'


def seed_of(fam, surface, block, rep, base=None):
    return int((base or SEED_BASE)[fam] + 100 * SURF_IDX[surface] + 10 * int(block) + int(rep))


def phases5(M5: pd.DataFrame) -> dict:
    out = {}
    for fam in ('point', 'line'):
        g = M5[M5.family == fam].drop_duplicates('phase_block').sort_values('phase_block')
        out[fam] = {int(r.phase_block): (dict(block=int(r.phase_block), k=int(r.phase_seq_k), u=float(r.phase_u), v=float(r.phase_v)) if fam == 'point'
                                         else dict(block=int(r.phase_block), k=int(r.phase_seq_k), phi=float(r.phase_frac))) for r in g.itertuples()}
    return out


def field_of(env: str) -> dict:
    import mp3_forward as F
    import r_fields as RF
    f = ENVS[env][1] if env in ENVS else None
    if f is None:
        raise K.Stop('preflight', f'environment {env} not registered for 06C')
    if f[0] == 'const':
        return F.field_const(f[1])
    if f[0] == 'gradient_y_reversed':
        return RF.reversed_y(f[1], f[2], HALF_MM)
    return F.field_gradient(f[0].split('_')[1], f[1], f[2], HALF_MM)


class Planner:
    def __init__(self, M5: pd.DataFrame):
        self.T = RI.template_rows(M5)

    def row(self, part, cfg, surface, ph, sigma, rep, env, seed_base=None, block_label=None) -> dict:
        import mp3_forward as F
        fam = fam_of(cfg); tpl = self.T[cfg]['row']
        inp = RI.inputs(tpl, fam, ph)
        blk = block_label if block_label is not None else ph['block']
        seed = None if float(sigma) == 0.0 else seed_of(fam, surface, ph['block'], rep, seed_base)
        fld = field_of(env)
        r = dict(platform_surface=SURF_SID[surface], sigma=float(sigma), seed=seed, psf_true_px=float(ENVS[env][0]), field=fld)
        spec, rec, route = RI.forward_record(r, inp)
        fk = K5.digest(rec); bind = RI.decoder_binding(fam, tpl, inp)
        return dict(part=part, family=fam, config=cfg, template_cand_id=tpl['cand_id'], template_source=self.T[cfg]['source'], surface=surface, platform_surface=SURF_SID[surface],
                    phase_block=blk, phase_seq_k=ph['k'], phase_u=ph.get('u'), phase_v=ph.get('v'), phase_frac=ph.get('phi'), sigma=float(sigma), rep=int(rep), seed=seed,
                    m3_seed=int(spec['noise']['m3_seed']), noise_stream=spec['noise'].get('stream'), env=env, psf_true_px=float(ENVS[env][0]), field_kind=fld['kind'],
                    field_json=json.dumps(fld, sort_keys=True), field_id=spec['field_id'], rays_per_pixel=int(spec['rays_per_pixel']), forward_route=route, adapter_file_sha256=rec['adapter_file'],
                    raster_key=inp['raster_key'], pattern_sha=rec['pattern'], peak=float(inp['peak']), peak_f32=rec['peak_f32'], integral=rec['integral'], lit_px=float(inp['lit_px']),
                    mesh_sha16=rec['mesh_sha16'], platform=rec['platform'], m3_file_sha256=rec['m3_file'], forward_key=fk, method=bind['method'],
                    method_binding=('P4_C3_cal (five-node table c2_exact_scales_fp1e-4_v1), steger_sigma ' + f"{bind['steger_sigma_px']:.10g} px" if bind['method'] == 'P4'
                                    else f"DF rule, radius {bind['radius_mm']} mm"),
                    case_key=f"{cfg}|{surface}|b{blk}|s{int(sigma)}|r{rep}|{env}")


def plan(M5: pd.DataFrame) -> pd.DataFrame:
    P5 = phases5(M5); PL = Planner(M5); rows = []
    for cfg in REPRESENTATIVES:
        fam = fam_of(cfg)
        for env in ('NOMINAL', 'BLUR2', 'BLUR3'):
            for s in K.SURFS:
                for b in K.BLOCKS:
                    rows.append(PL.row('C1_range', cfg, s, P5[fam][b], 0.0, 0, env))
                    for sg in (20.0, 40.0, 60.0, 80.0, 100.0):
                        for rp in K.REPS:
                            rows.append(PL.row('C1_range', cfg, s, P5[fam][b], sg, rp, env))
    for cfg in LINE_SET:
        for env in ('NOMINAL', 'BLUR1_5', 'BLUR2', 'BLUR2_5', 'BLUR3'):
            for s in K.SURFS:
                for b in K.BLOCKS:
                    for sg in (20.0, 40.0):
                        for rp in K.REPS:
                            rows.append(PL.row('C2_lines', cfg, s, P5['line'][b], sg, rp, env))
    for cfg in DOT_SET:
        for env in ('NOMINAL', 'BLUR2', 'BLUR3'):
            for s in K.SURFS:
                for b in K.BLOCKS:
                    for sg in (20.0, 40.0):
                        for rp in K.REPS:
                            rows.append(PL.row('C2_dots', cfg, s, P5['point'][b], sg, rp, env))
    for cfg in REPRESENTATIVES:
        fam = fam_of(cfg)
        for env in ('G05', 'G05_BLUR2'):
            for s in K.SURFS:
                for b in K.BLOCKS:
                    for sg in (20.0, 40.0):
                        for rp in K.REPS:
                            rows.append(PL.row('C2_g05', cfg, s, P5[fam][b], sg, rp, env))
            rows.append(PL.row('C2_g05_sigma0_diag', cfg, 'M10_curv', P5[fam][0], 0.0, 0, env))
    for cfg in sorted(M5.config.unique()):
        fam = fam_of(cfg)
        for s in K.SURFS:
            for b in K.BLOCKS:
                for sg in (20.0, 40.0):
                    for rp in K.REPS:
                        rows.append(PL.row('C2_grad_y_rev', cfg, s, P5[fam][b], sg, rp, 'GRAD_Y_REV'))
    return pd.DataFrame(rows)


ENTRY = [
    ('L_w0.75_s8', 'BLUR2_5', 'B_curv', 1, 20.0, 1, 'new width 0.75 (S1 template), PSF 2.5, B, lines'),
    ('L_w1_s8', 'BLUR2_5', 'M10_curv', 2, 40.0, 1, 'new width 1.0, PSF 2.5, M10; rep pair (1)'),
    ('L_w1_s8', 'BLUR2_5', 'M10_curv', 2, 40.0, 2, 'new width 1.0, PSF 2.5, M10; rep pair (2): pairing / shared base field'),
    ('L_w1.25_s8', 'BLUR2_5', 'M24_curv', 3, 20.0, 1, 'new width 1.25, PSF 2.5, M24'),
    ('D_r1_p8x2.5', 'BLUR3', 'M24_curv', 1, 40.0, 1, 'new dot spacing 8x2.5 (S2 template), strong blur, M24'),
    ('D_r1_p8x3', 'NOMINAL', 'B_curv', 2, 20.0, 1, 'new dot spacing 8x3 (S2), B'),
    ('D_r1_p8x8', 'BLUR2', 'M10_curv', 3, 40.0, 1, 'new dot spacing 8x8 (S2), M10'),
    ('D_r1.5_p8x3', 'BLUR2', 'B_curv', 0, 20.0, 1, 'new dot spacing r1.5 8x3 (design gap 0 mm; S5 template)'),
    ('L_w0.5_s8', 'G05', 'M24_curv', 1, 40.0, 1, 'g = 0.5 (lines)'),
    ('D_r1_p8x6', 'G05_BLUR2', 'B_curv', 2, 20.0, 1, 'g = 0.5 with PSF 2 (dots)'),
    ('D_r1.5_p8x4', 'G05', 'M10_curv', 0, 0.0, 0, 'g = 0.5 sigma-0 diagnostic row (empty seed path)'),
    ('L_w0.5_s8', 'GRAD_Y_REV', 'M10_curv', 1, 40.0, 1, 'reversed y field via r_forward (lines)'),
    ('D_r1_p8x6', 'GRAD_Y_REV', 'M24_curv', 2, 20.0, 1, 'reversed y field via r_forward (dots)'),
    ('L_w1.5_s8', 'BLUR3', 'B_curv', 1, 0.0, 0, 'range check sigma 0 outside M10 block 0'),
    ('D_r1_p8x6', 'NOMINAL', 'M10_curv', 3, 100.0, 1, 'range check strongest noise (dots)'),
    ('L_w0.5_s8', 'BLUR2', 'M24_curv', 2, 100.0, 1, 'range check strongest noise (lines)'),
]
ENTRY_CHECKS = ['new widths 0.75 / 1.0 / 1.25 at PSF 2.5', 'new dot spacings (r1 2.5 / 3 / 8, r1.5 3)', 'g = 0.5 and the reversed y field', 'B / M10 / M24, dots and lines',
                'sigma 0 and sigma 100', 'one complete rep 1 / rep 2 pair', 'chain render -> formal decode (DF / P4) -> evaluation -> analysis read']


def entry_subset(U: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    flag = pd.Series(False, index=U.index); why = pd.Series('', index=U.index); idx = U.set_index('case_key')
    for cfg, env, s, b, sg, rp, reason in ENTRY:
        ck = f'{cfg}|{s}|b{b}|s{int(sg)}|r{rp}|{env}'
        if ck not in idx.index:
            raise K.Stop('identity', f'entry case {ck} not in the manifest')
        i = U.index[U.case_key == ck][0]
        if U.at[i, 'status'] != 'new':
            raise K.Stop('identity', f'entry case {ck} is not a new image')
        flag[i] = True; why[i] = reason
    E = U[flag]
    cov = dict(n=int(flag.sum()), widths_psf25=sorted(E[(E.env == 'BLUR2_5')].config.unique().tolist()), new_dot_spacings=sorted(E[E.config.isin(['D_r1_p8x2.5', 'D_r1_p8x3', 'D_r1_p8x8', 'D_r1.5_p8x3'])].config.unique().tolist()),
               g05=bool(E.env.isin(['G05', 'G05_BLUR2']).any()), reversed=bool((E.env == 'GRAD_Y_REV').any()), surfaces=sorted(E.surface.unique().tolist()), families=sorted(E.family.unique().tolist()),
               sigma0=int((E.sigma == 0).sum()), sigma100=int((E.sigma == 100).sum()),
               rep_pair=bool(E.groupby(['config', 'surface', 'phase_block', 'sigma', 'env']).rep.apply(lambda s: set(s) == {1, 2}).any()), routes=sorted(E.forward_route.unique().tolist()),
               methods=E.method.value_counts().to_dict())
    cov['ok'] = bool(cov['n'] <= 24 and cov['widths_psf25'] == ['L_w0.75_s8', 'L_w1.25_s8', 'L_w1_s8'] and len(cov['new_dot_spacings']) == 4 and cov['g05'] and cov['reversed']
                     and cov['surfaces'] == list(K.SURFS) and cov['families'] == ['line', 'point'] and cov['sigma0'] >= 1 and cov['sigma100'] >= 1 and cov['rep_pair']
                     and cov['routes'] == ['mp3_forward', 'r_forward'])
    return flag, why, cov


LAUNCH_PLAN = [
    'freeze re-verification (live hashes) and the protocol gate of the later task; a redirect or any drift stops before work',
    'old-input compatibility (counted separately, not science): 2 MODEL-PREP-05 images through r_inputs + frozen mp3_forward (pixel identity) and 2 '
    'MODEL-PREP-05 images with const / gradient_y fields through r_forward (pixel identity) - proves the template path and the adapter on old kinds',
    'entry subset in this order, each image render -> formal decode (DF / P4_C3_cal) -> evaluation -> pack -> read-back -> analysis row: S1 lines (PSF 2.5), '
    'S2 / S5 dots, g = 0.5 (incl. sigma 0), reversed y field, sigma 0 range row, sigma 100 rows, rep pair',
    'stop at the first technical failure (render / decode / evaluation exception, key or identity mismatch, missing field, resource change, wrong route); '
    'scientific results (low coverage, no output) are not stops',
    'only after review: the remaining 3164 - 16 new images, de-duplicated, then the selection freeze, then the holdout (<= 384)']


PLANNED_ONLY = ('outputs/surface_revision/model_prep06a/manifest.csv', 'outputs/surface_revision/model_prep06ar/manifest.csv')


# Select new phases and enumerate the held-out candidate cases.
def holdout_refresh(M5: pd.DataFrame, U: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import mp4_formal as MF
    import tr3_b2 as B2
    reg = B2.registry(); Pz = reg['positions']; F = reg['phases']
    src = pd.DataFrame(reg.get('sources', []))
    src['class'] = ['planned, not imaged' if str(s) in PLANNED_ONLY else 'registry (used / seen)' for s in src['source']] if len(src) else []
    lat = Pz.lattice.astype(str); Rpos = Pz[lat.str.startswith('R') | lat.isin(['unknown', 'nan', 'generic', 'None'])]
    P5 = phases5(M5)
    used_pt = [(float(a), float(b), str(s)) for a, b, s in zip(pd.to_numeric(Rpos.u), pd.to_numeric(Rpos.v), Rpos.source)] + [(q['u'], q['v'], 'model_prep05/formal_manifest.csv') for q in P5['point'].values()]
    Fl = F[(F.pitch.isna() | (pd.to_numeric(F.pitch, errors='coerce').sub(8.0).abs() < 1e-6)) & (F.orientation.isna() | (pd.to_numeric(F.orientation, errors='coerce').abs() < 1e-9))]
    used_ln = [(float(p), str(s)) for p, s in zip(pd.to_numeric(Fl.phase, errors='coerce'), Fl.source) if not np.isnan(p)] + [(q['phi'], 'model_prep05/formal_manifest.csv') for q in P5['line'].values()]
    known = set(reg['raster_keys']) | set(M5.raster_key) | set(U.raster_key)
    T = RI.template_rows(M5); out = {}
    for fam, cfgs in (('point', DOT_SET), ('line', LINE_SET)):
        acc, rej, k = [], {}, 0
        while len(acc) < 4 and k < 20000:
            k += 1
            ph = (dict(u=float((0.5 + k / MF.RHO) % 1.0), v=float((0.5 + k / MF.RHO ** 2) % 1.0)) if fam == 'point' else dict(phi=float(((0.5 + k * MF.SQ2M1) % 1.0) - 0.5)))
            if fam == 'point':
                d = MF._torus(ph['u'] - np.array([a for a, _, _ in used_pt]), ph['v'] - np.array([b for _, b, _ in used_pt])); i = int(np.argmin(d)); dmin = float(d[i]); near = used_pt[i][2]
            else:
                dd = np.abs(((ph['phi'] - np.array([p for p, _ in used_ln])) + 0.5) % 1.0 - 0.5); i = int(np.argmin(dd)); dmin = float(dd[i]); near = used_ln[i][1]
            why = []
            if dmin < MF.MIN_DIST[fam]:
                why.append('near used')
            if not why and acc:
                dm = min((float(MF._torus(ph['u'] - q['u'], ph['v'] - q['v'])) if fam == 'point' else abs(((ph['phi'] - q['phi']) + 0.5) % 1.0 - 0.5)) for q in acc)
                if dm < MF.MIN_DIST[fam]:
                    why.append('near accepted')
            if not why:
                for cfg in cfgs:
                    b = MF.build(T[cfg]['row'], fam, ph)
                    if b['raster_key'] in known:
                        why.append('raster used')
                    if not b['resource_ok']:
                        why.append('resource')
                    if fam == 'line' and b['registered_without_lit']:
                        why.append('registered line not lit')
            if why:
                for w in set(why):
                    rej[w] = rej.get(w, 0) + 1
                continue
            acc.append(dict(block=f'H{len(acc)}', k=k, **ph, min_dist_used=dmin, nearest_source=near))
        out[fam] = dict(accepted=acc, found=len(acc) == 4, rejected=rej, threshold=MF.MIN_DIST[fam])

    used_seeds = set(int(x) for x in reg['seeds']) | set(int(x) for x in pd.to_numeric(M5.seed, errors='coerce').dropna()) | set(int(x) for x in pd.to_numeric(U.seed, errors='coerce').dropna())
    base = {}
    for fam, b0 in HOLD_BASE0.items():
        b = b0
        while {b + 100 * s + 10 * bl + r for s in range(3) for bl in range(4) for r in (1, 2)} & used_seeds:
            b += 100_000
        base[fam] = b
    old = K.load_json(K.SRC6 / 'protocol.json')['C']['holdout_phases']
    same = {f: [q['k'] for q in out[f]['accepted']] == [q['k'] for q in old[f]] for f in out}

    rules = K.load_json(K.SRC6 / 'protocol.json')['rules']['holdout']
    env_of = {'H1': 'NOMINAL', 'H2': 'BLUR2', 'H3': 'BLUR3', 'H4': 'G05'}
    PL = Planner(M5); H = []
    for c in rules['conditions']:
        for fam, ref in (('point', 'D_r1_p8x6'), ('line', 'L_w0.5_s8')):
            if not out[fam]['found']:
                continue
            for i, ph in enumerate(out[fam]['accepted']):
                ph2 = dict(ph, block=i)
                for s in K.SURFS:
                    for rp in K.REPS:
                        r = PL.row('C3_holdout_ref', ref, s, ph2, c['sigma'], rp, env_of[c['name']], seed_base=base, block_label=f'H{i}')
                        r.update(holdout_condition=c['name'], slot='reference (fixed)', status='holdout planned (not imaged)', min_dist_used=ph['min_dist_used'], nearest_source=ph['nearest_source'])
                        H.append(r)
                        H.append(dict(part='C3_holdout_pref', family=fam, config='SLOT: three-surface preferred configuration (filled only after the selection freeze)', surface=s,
                                      phase_block=f'H{i}', phase_seq_k=ph['k'], phase_u=ph.get('u'), phase_v=ph.get('v'), phase_frac=ph.get('phi'), sigma=c['sigma'], rep=rp,
                                      seed=seed_of(fam, s, i, rp, base), env=env_of[c['name']], psf_true_px=c['psf_px'], holdout_condition=c['name'],
                                      slot='preferred (empty; identical to the reference -> dropped)', status='slot (not filled, not imaged)', forward_key=None,
                                      min_dist_used=ph['min_dist_used'], nearest_source=ph['nearest_source']))
    Hd = pd.DataFrame(H)
    hseeds = set(int(x) for x in pd.to_numeric(Hd.seed, errors='coerce').dropna())
    rec = dict(phases={f: v['accepted'] for f, v in out.items()}, found={f: v['found'] for f, v in out.items()}, rejected={f: v['rejected'] for f, v in out.items()},
               same_as_06A=same, seed_base=base, seed_conflicts=sorted(hseeds & used_seeds)[:10], n_seed_conflicts=len(hseeds & used_seeds),
               sources=src.to_dict('records') if len(src) else [], extra_sources=['outputs/surface_revision/model_prep05/formal_manifest.csv (not matched by the registry glob; added explicitly)'],
               formal_manifest_in_registry=bool(len(src) and src.source.astype(str).str.contains('formal_manifest').any()),
               rows_reference=int((Hd.part == 'C3_holdout_ref').sum()), rows_slot=int((Hd.part == 'C3_holdout_pref').sum()))
    return Hd, rec


def build(M5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    D = plan(M5)
    fk5 = M5.set_index('case_key').forward_key.to_dict()
    D['mp5_case_key'] = [ck if ck in fk5 else None for ck in D.case_key]
    D['mp5_forward_key_equal'] = [(fk5.get(ck) == fk) if ck in fk5 else None for ck, fk in zip(D.case_key, D.forward_key)]
    if (D.mp5_forward_key_equal == False).any():
        raise K.Stop('identity', f'planned rows with a MODEL-PREP-05 key but a different forward key: {D[D.mp5_forward_key_equal == False].case_key.tolist()[:5]}')
    if D.forward_key.isna().any():
        raise K.Stop('identity', 'planned rows without a forward key')
    D['status'] = np.where(D.mp5_forward_key_equal == True, 'reuse_mp5', 'new')
    D['reuse_reason'] = np.where(D.status == 'reuse_mp5', 'recomputed forward key == MODEL-PREP-05 frozen key of the same case (template, phase, noise, PSF, field, route)', '')
    parts = D.groupby('forward_key').part.apply(lambda s: ';'.join(sorted(set(s))))
    U = D.drop_duplicates('forward_key').copy(); U['analysis_views'] = U.forward_key.map(parts); U = U.drop(columns=['part'])
    if U.case_key.duplicated().any():
        raise K.Stop('identity', 'two physical images with the same case key')
    U['source_image_key'] = U.mp5_case_key
    U = U.reset_index(drop=True)
    flag, why, cov = entry_subset(U)
    U['entry_subset'] = flag; U['entry_reason'] = why

    O = pd.read_csv(K.SRC6 / 'manifest.csv', low_memory=False)
    J = O[['case_key', 'forward_key', 'status']].merge(U[['case_key', 'forward_key', 'status']], on='case_key', how='outer', suffixes=('_06A', ''), indicator=True)
    both = J[J._merge == 'both']
    cmp = dict(rows_06A=len(O), rows_now=len(U), only_06A=int((J._merge == 'left_only').sum()), only_now=int((J._merge == 'right_only').sum()),
               forward_key_equal_where_06A_had_one=int((both.forward_key_06A.notna() & (both.forward_key_06A == both.forward_key)).sum()),
               forward_key_differs_where_06A_had_one=int((both.forward_key_06A.notna() & (both.forward_key_06A != both.forward_key)).sum()),
               pending_06A_now_keyed=int((both.forward_key_06A.isna() & both.forward_key.notna()).sum()), status_changed=int((both.status_06A != both.status).sum()))
    new = U[U.status == 'new']
    summ = dict(planned_rows=len(D), physical=len(U), reuse=int((U.status == 'reuse_mp5').sum()), new=len(new), new_sigma0=int((new.sigma == 0).sum()),
                new_methods=new.method.value_counts().to_dict(), routes=U.forward_route.value_counts().to_dict(),
                by_part={p: dict(rows=int((D.part == p).sum()), reuse=int(((D.part == p) & (D.status == 'reuse_mp5')).sum()), new=int(((D.part == p) & (D.status == 'new')).sum())) for p in D.part.unique()},
                missing_physical_fields=int(U[['forward_key', 'raster_key', 'pattern_sha', 'mesh_sha16', 'field_id', 'm3_seed', 'adapter_file_sha256']].isna().sum().sum()),
                sigma_gt0_without_seed=int(((U.sigma > 0) & U.seed.isna()).sum()), sigma0_with_seed=int(((U.sigma == 0) & U.seed.notna()).sum()),
                resource_bad=int((U.peak > 1.0 + 1e-12).sum()), entry=cov, comparison_06A=cmp)
    return U, D, summ
