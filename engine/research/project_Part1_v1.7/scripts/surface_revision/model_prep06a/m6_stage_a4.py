# Compare uniform attenuation with noise scaling and summarize illumination bands.

"""MODEL-PREP-06A A4 (should): illumination bridge and bands, stored data only.

bridge   MODEL-PREP-05 UNIFORM (g = 0.75) and NOMINAL levels (glob_Ctau, rep -> block -> four-block mean) against the section-5 (S5-ALIGN-05)
         condition summary of the SAME configuration and front end (DF dots, P4 lines; 8 phases there, 4 blocks here; aggregate levels only,
         never paired). The equivalent noise sigma / g lies between measured S5 levels -> linear interpolation as an approximate display, the
         measured anchors are listed. Light-weighted gain = MODEL-PREP-05 received_light_ratio (shaded intensity of the FOV hits after gain /
         before gain; stage: before PSF, clipping and noise) - it is exactly 0.75 for UNIFORM; for the gradients it is reported, not assumed.
bands    expected positions defined by the stored expected set of the case (lines: reference rows ref_xyz; dots: expected visible dots
         exp_xy / exp_in_image), identical for the paired NOMINAL / UNIFORM / GRAD_X / GRAD_Y cases of one (configuration, surface, block, sigma,
         rep) (asserted). Band of a position = gain of the GRADIENT field at its (x, y) (mp3_forward.gain), endpoints registered before the
         first band output (protocol A4.band_rule). Per band: expected count, recovered with correct id under GRAD, UNIFORM and NOMINAL at the
         SAME positions. Pooled per block over surfaces and reps, then the four-block mean / range.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import m6_common as K
import m6_stats as ST

BAND_EDGES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-12)
CONFIGS = ('D_r1_p8x4', 'D_r1_p8x6', 'D_r1.5_p8x4', 'D_r1.5_p8x6', 'L_w0.5_s6.3', 'L_w0.5_s8', 'L_w1.5_s6.3', 'L_w1.5_s8')


def level(R: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = []
    for (cfg, env, sg), g in R[R.sigma > 0].groupby(['config', 'env', 'sigma']):
        s = ST.blocks(g.groupby(['phase_block', 'rep'], as_index=False)[metric].mean().rename(columns={metric: 'value'}))
        out.append(dict(config=cfg, env=env, sigma=sg, **ST.flat('', s)))
    return pd.DataFrame(out)


def s5_interp(S5: pd.DataFrame, cfg: str, fe: str, sig: float) -> dict:
    q = S5[(S5.combo == cfg) & (S5.front_end == fe)].groupby('sigma').glob_Ctau_mean8.mean().sort_index()
    xs = q.index.to_numpy(float); ys = q.to_numpy(float)
    if sig < xs.min() or sig > xs.max():
        return dict(value=None, lo=None, hi=None, note='outside the measured S5 range (no extrapolation)')
    lo = xs[xs <= sig].max(); hi = xs[xs >= sig].min()
    return dict(value=float(np.interp(sig, xs, ys)), lo=float(lo), hi=float(hi), v_lo=float(q[lo]), v_hi=float(q[hi]), exact=bool(lo == hi))


def bridge() -> pd.DataFrame:
    R = pd.read_csv(K.SRC5 / 'results.csv', float_precision='round_trip', low_memory=False)
    R = R[((R.family == 'point') & (R.method == 'DF')) | ((R.family == 'line') & (R.method == 'P4'))]
    S5 = pd.read_csv(K.S5A / 'condition_summary.csv', low_memory=False)
    rows = []
    for cfg in CONFIGS:
        fe = 'DF' if cfg.startswith('D_') else 'P4'
        for sg in (20.0, 40.0):
            for env in ('NOMINAL', 'UNIFORM', 'GRAD_X', 'GRAD_Y'):
                g = R[(R.config == cfg) & (R.env == env) & (R.sigma == sg)]
                lv = ST.blocks(g.groupby(['phase_block', 'rep'], as_index=False).glob_Ctau.mean().rename(columns={'glob_Ctau': 'value'}))
                geff = float(g.diag_received_light_ratio.mean())
                gmin, gmax = float(g.diag_received_light_ratio.min()), float(g.diag_received_light_ratio.max())
                ref_g = 1.0 if env == 'NOMINAL' else 0.75
                pred_nom = s5_interp(S5, cfg, fe, sg / ref_g); pred_light = s5_interp(S5, cfg, fe, sg / geff)
                rows.append(dict(config=cfg, front_end=fe, sigma=sg, env=env, mp5_level=lv['mean'], mp5_min=lv['vmin'], mp5_max=lv['vmax'],
                                 light_weighted_gain_mean=geff, light_weighted_gain_min=gmin, light_weighted_gain_max=gmax,
                                 sat_frac_pre_clip_max=float(g.diag_frac_pre_clip_above_1.max()), sigma_eq_world=sg / ref_g, s5_at_sigma_eq_world=pred_nom['value'],
                                 s5_anchor_lo=pred_nom.get('lo'), s5_anchor_hi=pred_nom.get('hi'), s5_v_lo=pred_nom.get('v_lo'), s5_v_hi=pred_nom.get('v_hi'),
                                 sigma_eq_light=sg / geff, s5_at_sigma_eq_light=pred_light['value'],
                                 diff_vs_s5_world=(lv['mean'] - pred_nom['value']) if (lv['mean'] is not None and pred_nom['value'] is not None) else None,
                                 diff_vs_s5_light=(lv['mean'] - pred_light['value']) if (lv['mean'] is not None and pred_light['value'] is not None) else None))
    return pd.DataFrame(rows)


def _positions(ck: str, method: str, family: str):
    meta, A = K.read_pack(ck, prefixes=(f'{method}__ev__ref_xyz', f'{method}__ev__ref_rec_id', f'{method}__ev__exp_xy', f'{method}__ev__exp_in_image',
                                        f'{method}__ev__exp_recovered'))
    if family == 'line':
        xy = np.asarray(A[f'{method}__ev__ref_xyz'], np.float64).reshape(-1, 3)[:, :2]; rec = np.asarray(A[f'{method}__ev__ref_rec_id'], bool)
    else:
        m = np.asarray(A[f'{method}__ev__exp_in_image'], bool)
        xy = np.asarray(A[f'{method}__ev__exp_xy'], np.float64).reshape(-1, 2)[m]; rec = np.asarray(A[f'{method}__ev__exp_recovered'], bool)[m]
    return xy, rec


def bands() -> pd.DataFrame:
    import mp3_forward as F
    M = pd.read_csv(K.SRC5 / 'formal_manifest.csv', float_precision='round_trip', low_memory=False)
    M = M[(M.subset == 'core') & M.env.isin(['NOMINAL', 'UNIFORM', 'GRAD_X', 'GRAD_Y'])]
    rows = []
    for key, g in M.groupby(['config', 'surface', 'phase_block', 'sigma', 'rep']):
        cfg = key[0]; fam = 'point' if cfg.startswith('D_') else 'line'; meth = 'DF' if fam == 'point' else 'P4'
        e = {r['env']: r for r in g.to_dict('records')}
        pos = {env: _positions(e[env]['case_key'], meth, fam) for env in e}
        xy0 = pos['NOMINAL'][0]
        for env in e:
            if pos[env][0].shape != xy0.shape or not np.array_equal(pos[env][0], xy0):
                raise K.Stop('pairing', f'{key}: expected positions differ between NOMINAL and {env}')
        for gk in ('GRAD_X', 'GRAD_Y'):
            fld = json.loads(e[gk]['field_json']); gv = F.gain(fld, xy0[:, 0], xy0[:, 1])
            bi = np.digitize(gv, BAND_EDGES) - 1
            for b in range(len(BAND_EDGES) - 1):
                m = bi == b
                rows.append(dict(config=cfg, family=fam, surface=key[1], phase_block=int(key[2]), sigma=float(key[3]), rep=int(key[4]), gradient=gk, band=b,
                                 g_lo=BAND_EDGES[b], g_hi=min(1.0, BAND_EDGES[b + 1]), n_expected=int(m.sum()), g_mean=(float(gv[m].mean()) if m.any() else None),
                                 rec_grad=int(pos[gk][1][m].sum()), rec_uniform=int(pos['UNIFORM'][1][m].sum()), rec_nominal=int(pos['NOMINAL'][1][m].sum())))
    return pd.DataFrame(rows)


def band_summary(B: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (cfg, gk, sg, b), g in B.groupby(['config', 'gradient', 'sigma', 'band']):
        p = g.groupby(['phase_block', 'rep'], as_index=False)[['n_expected', 'rec_grad', 'rec_uniform', 'rec_nominal']].sum()
        rec = dict(config=cfg, gradient=gk, sigma=sg, band=int(b), g_lo=float(g.g_lo.iloc[0]), g_hi=float(g.g_hi.iloc[0]), n_expected_total=int(g.n_expected.sum()))
        for c in ('rec_grad', 'rec_uniform', 'rec_nominal'):
            q = p.assign(value=np.where(p.n_expected > 0, p[c] / p.n_expected.where(p.n_expected > 0, 1), np.nan))
            s = ST.blocks(q[['phase_block', 'rep', 'value']])
            rec.update({f'{c}_rate_mean': s['mean'], f'{c}_rate_min': s['vmin'], f'{c}_rate_max': s['vmax']})
        q = p.assign(value=np.where(p.n_expected > 0, (p.rec_grad - p.rec_uniform) / p.n_expected.where(p.n_expected > 0, 1), np.nan))
        s = ST.blocks(q[['phase_block', 'rep', 'value']]); rec.update(diff_grad_minus_uniform_mean=s['mean'], diff_min=s['vmin'], diff_max=s['vmax'], diff_n_pos=s['n_pos'], diff_n_neg=s['n_neg'])
        out.append(rec)
    return pd.DataFrame(out)


def stage_a4() -> dict:
    K.rules_registered()
    K.protocol_put('A4_band_rule', dict(t=K.now(), edges=list(BAND_EDGES), definition=__doc__.strip()))
    Bg = bridge(); K.save_csv(Bg, K.OUT / '_work' / 'a4_bridge.csv')
    B = bands(); K.save_csv(B, K.OUT / '_work' / 'a4_bands_long.csv')
    S = band_summary(B); K.save_csv(S, K.OUT / '_work' / 'a4_bands.csv')
    rec = dict(t=K.now(), bridge_rows=len(Bg), band_rows=len(B), band_summary_rows=len(S))
    K.protocol_put('A4', rec)
    return rec
