# Plot environment effects and representative reconstruction results.

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import mp5_common as K

TAG = 'MODEL-PREP-05 · formale Validierung: 4 neue Phasen × 2 Rauschrealisierungen, feste Verfahren · keine Gerätekalibrierung'
SURF_DE = {'B_curv': 'B', 'M10_curv': 'M10', 'M24_curv': 'M24'}
CFG_DE = {'D_r1_p8x4': 'Ø2 mm, 8×4', 'D_r1_p8x6': 'Ø2 mm, 8×6', 'D_r1.5_p8x4': 'Ø3 mm, 8×4', 'D_r1.5_p8x6': 'Ø3 mm, 8×6',
          'L_w0.5_s6.3': 'w 0,5 / s 6,3', 'L_w0.5_s8': 'w 0,5 / s 8', 'L_w1.5_s6.3': 'w 1,5 / s 6,3', 'L_w1.5_s8': 'w 1,5 / s 8'}
COL = {'D_r1_p8x4': '#1f77b4', 'D_r1_p8x6': '#ff7f0e', 'D_r1.5_p8x4': '#2ca02c', 'D_r1.5_p8x6': '#d62728', 'L_w0.5_s6.3': '#1f77b4', 'L_w0.5_s8': '#ff7f0e',
       'L_w1.5_s6.3': '#2ca02c', 'L_w1.5_s8': '#d62728'}
ENV_DE = {'NOMINAL': 'nominal', 'BLUR1_5': 'PSF 1,5 px', 'BLUR2': 'PSF 2 px', 'BLUR3': 'PSF 3 px', 'UNIFORM': 'g = 0,75', 'GRAD_X': 'Gradient x', 'GRAD_Y': 'Gradient y'}
PSF_X = {'NOMINAL': 1.0, 'BLUR1_5': 1.5, 'BLUR2': 2.0, 'BLUR3': 3.0}
DOTS = ('D_r1_p8x4', 'D_r1_p8x6', 'D_r1.5_p8x4', 'D_r1.5_p8x6')
LINES = ('L_w0.5_s6.3', 'L_w0.5_s8', 'L_w1.5_s6.3', 'L_w1.5_s8')
SURFS = ('B_curv', 'M10_curv', 'M24_curv')
SCOL = {'B_curv': '#4c72b0', 'M10_curv': '#dd8452', 'M24_curv': '#55a868'}
ERR_LIM = 0.5
RULES = dict(figures=('Abb1_Unschaerfe_Rangfolge', 'Abb2_Punkte_Groesse_Abstand', 'Abb3_Nichtuniformitaet', 'Abb4_Abdeckung_Risiko', 'Abb5_Typische_Topographie', 'Abb6_Fehlerfall'),
             board='analysis.BOARD_CASES (fixed before the run)', failure='analysis.select_failure (rule fixed before the run; skipped if no wrong-id increase)',
             scales='camera grey 0-255; top-view error colour +-0.5 mm, beyond: black + / wrong ids: magenta x; global domain grey, local domain dashed circle')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out_dir, name, plt) -> str:
    from pathlib import Path
    p = Path(out_dir) / f'{name}.png'; p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
    return str(p)


def _pp(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else 100.0 * float(x)


def fig1(Lv: pd.DataFrame, out_dir, tag_extra='') -> str:
    plt = _plt(); fig, axs = plt.subplots(2, 7, figsize=(24, 8.2), squeeze=False)
    L = Lv[(Lv.metric == 'glob_Ctau')]
    for i, (fam, meth, cfgs) in enumerate((('point', 'DF', DOTS), ('line', 'P4', LINES))):
        cols = [(s, sg) for s in SURFS for sg in (20.0, 40.0)] + [('M10_curv', 0.0)]
        for j, (s, sg) in enumerate(cols):
            ax = axs[i, j]
            for c in cfgs:
                q = L[(L.config == c) & (L.surface == s) & (L.sigma == sg) & (L.method == meth) & L.env.isin(list(PSF_X))].copy()
                if not len(q):
                    continue
                q['x'] = q.env.map(PSF_X); q = q.sort_values('x')
                y = q['mean'].map(_pp)
                if sg > 0:
                    lo = (q['mean'] - q['min']).map(_pp).fillna(0); hi = (q['max'] - q['mean']).map(_pp).fillna(0)
                    ax.errorbar(q.x, y, yerr=[lo, hi], marker='o', ms=4, lw=1.4, capsize=3, color=COL[c], label=CFG_DE[c])
                else:
                    ax.plot(q.x, y, marker='s', ms=5, lw=1.0, ls=':', color=COL[c], label=CFG_DE[c])
            ax.set_xticks([1, 1.5, 2, 3]); ax.set_xticklabels(['1', '1,5', '2', '3'], fontsize=8); ax.set_ylim(-3, 100)
            ax.set_title((f'{SURF_DE[s]}, σ = {sg:g}' if sg > 0 else 'σ = 0 (M10, Block 0, Einzelfall)'), fontsize=9.5)
            ax.set_xlabel('wahre PSF σ / px (Decoder nominal 1 px)', fontsize=8)
            if j == 0:
                ax.set_ylabel(f"Cτ global / % ({'Punkte, DF' if fam == 'point' else 'Linien, P4'})")
        h, lab = axs[i, 0].get_legend_handles_labels()
        axs[i, 6].legend(h, lab, fontsize=8, loc='upper right', frameon=False)
    fig.suptitle('Unschärfe und Rangfolge: Mittel über 4 Phasenblöcke (je Mittel aus 2 Rauschrealisierungen), Fehlerbalken = Spannweite der Blöcke; '
                 'PSF 1,5 nur für 8-mm-Linien\n' + TAG + tag_extra, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out_dir, 'Abb1_Unschaerfe_Rangfolge', plt)


def fig2(S: pd.DataFrame, out_dir, tag_extra='') -> str:
    plt = _plt(); labs = ('size@p8x4', 'size@p8x6', 'spacing@r1', 'spacing@r1.5')
    lab_de = {'size@p8x4': 'Größe: Ø3 − Ø2 bei 8×4', 'size@p8x6': 'Größe: Ø3 − Ø2 bei 8×6', 'spacing@r1': 'Abstand: 8×6 − 8×4 bei Ø2', 'spacing@r1.5': 'Abstand: 8×6 − 8×4 bei Ø3'}
    envs = ('NOMINAL', 'BLUR2', 'BLUR3', 'UNIFORM', 'GRAD_X', 'GRAD_Y')
    fig, axs = plt.subplots(4, 3, figsize=(17, 14), squeeze=False)
    Q = S[(S.contrast == 'F2 config_effect') & (S.metric == 'glob_Ctau') & (S.method == 'DF')]
    for i, lb in enumerate(labs):
        for j, sg in enumerate((20.0, 40.0, 0.0)):
            ax = axs[i, j]
            for k, s in enumerate(SURFS if sg > 0 else ('M10_curv',)):
                q = Q[(Q.label == lb) & (Q.surface == s) & (Q.sigma == sg)].set_index('env')
                xs = np.arange(len(envs)) + (k - 1) * 0.18
                y = [(_pp(q.loc[e, 'mean']) if e in q.index else np.nan) for e in envs]
                if sg > 0:
                    lo = [(_pp(q.loc[e, 'mean'] - q.loc[e, 'min']) if e in q.index and q.loc[e, 'min'] is not None else 0) for e in envs]
                    hi = [(_pp(q.loc[e, 'max'] - q.loc[e, 'mean']) if e in q.index and q.loc[e, 'max'] is not None else 0) for e in envs]
                    ax.errorbar(xs, y, yerr=[lo, hi], fmt='o', ms=4, capsize=3, color=SCOL[s], label=SURF_DE[s])
                else:
                    ax.plot(xs, y, 's', ms=5, color=SCOL[s], label='M10 (Einzelfall)')
            ax.axhline(0, color='0.5', lw=0.7)
            ax.set_xticks(np.arange(len(envs))); ax.set_xticklabels([ENV_DE[e] for e in envs], fontsize=8, rotation=20)
            ax.set_title(f"{lab_de[lb]}, σ = {sg:g}" + (' (Einzelfall)' if sg == 0 else ''), fontsize=9.5)
            if j == 0:
                ax.set_ylabel('Δ Cτ global / pp (DF)')
            if i == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Punktraster: Größen- und Abstandseffekt je Umgebung (Mittel über 4 Phasenblöcke, Balken = Spannweite); '
                 'Abstand zum Nominalwert = Differenz der Differenzen\n' + TAG + tag_extra, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_dir, 'Abb2_Punkte_Groesse_Abstand', plt)


def fig3(S: pd.DataFrame, Lv: pd.DataFrame, out_dir, tag_extra='') -> str:
    plt = _plt(); fig, axs = plt.subplots(2, 5, figsize=(24, 9), squeeze=False)
    panels = (('F1 env_vs_parent', 'UNIFORM', 'glob_Ctau', 'g = 0,75 − nominal, global'), ('F1 env_vs_parent', 'UNIFORM', 'loc_Ctau', 'g = 0,75 − nominal, lokal'),
              ('F4 grad_vs_uniform', None, 'glob_Ctau', 'Gradient − g = 0,75, global'), ('F4 grad_vs_uniform', None, 'loc_Ctau', 'Gradient − g = 0,75, lokal'))
    for i, (fam, meth, cfgs) in enumerate((('point', 'DF', DOTS), ('line', 'P4', LINES))):
        for j, (con, env, mt, title) in enumerate(panels):
            ax = axs[i, j]
            envs = (env,) if env else ('GRAD_X', 'GRAD_Y')
            for ei, e in enumerate(envs):
                for si, s in enumerate(SURFS):
                    for gi, sg in enumerate((20.0, 40.0)):
                        q = S[(S.contrast == con) & (S.env == e) & (S.metric == mt) & (S.method == meth) & (S.surface == s) & (S.sigma == sg)].set_index('label')
                        xs = np.arange(len(cfgs)) + (ei - (len(envs) - 1) / 2) * 0.36 + (si - 1) * 0.1 + (gi - 0.5) * 0.04
                        y = [(_pp(q.loc[c, 'mean']) if c in q.index else np.nan) for c in cfgs]
                        lo = [(_pp(q.loc[c, 'mean'] - q.loc[c, 'min']) if c in q.index and q.loc[c, 'min'] is not None else 0) for c in cfgs]
                        hi = [(_pp(q.loc[c, 'max'] - q.loc[c, 'mean']) if c in q.index and q.loc[c, 'max'] is not None else 0) for c in cfgs]
                        ax.errorbar(xs, y, yerr=[lo, hi], fmt=('o' if sg == 20 else 's'), ms=3.5, capsize=2, color=SCOL[s], mfc=(SCOL[s] if e != 'GRAD_Y' else 'none'), lw=0.8)
            ax.axhline(0, color='0.5', lw=0.7); ax.set_xticks(np.arange(len(cfgs))); ax.set_xticklabels([CFG_DE[c] for c in cfgs], fontsize=7.5, rotation=15)
            ax.set_title(f"{'Punkte (DF)' if fam == 'point' else 'Linien (P4)'}: {title}", fontsize=9.5)
            if j == 0:
                ax.set_ylabel('Δ Cτ / pp')
        ax = axs[i, 4]
        q = Lv[(Lv.metric == 'diag_received_light_ratio') & (Lv.family == fam) & (Lv.method == meth) & (Lv.sigma > 0)]
        envs = ('UNIFORM', 'GRAD_X', 'GRAD_Y')
        for k, e in enumerate(envs):
            v = q[q.env == e]['mean'].astype(float)
            ax.scatter(np.full(len(v), k) + np.linspace(-0.2, 0.2, len(v)) if len(v) else [], v, s=10, color='0.3')
        ax.axhline(0.75, color='#d62728', lw=0.8, ls='--')
        ax.set_xticks(range(len(envs))); ax.set_xticklabels([ENV_DE[e] for e in envs]); ax.set_ylabel('empfangene Lichtmenge / ungedämpft')
        ax.set_title('Empfangene Lichtmenge (Mittel der Blöcke je Konfig./Fläche/σ)', fontsize=9)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='none', color=SCOL[s], label=SURF_DE[s]) for s in SURFS] + [Line2D([], [], marker='o', ls='none', color='0.3', label='σ = 20'),
                                                                                                    Line2D([], [], marker='s', ls='none', color='0.3', label='σ = 40'),
                                                                                                    Line2D([], [], marker='o', ls='none', mfc='none', color='0.3', label='offen: Gradient y')]
    fig.legend(handles=h, loc='lower center', ncol=6, frameon=False, fontsize=9)
    fig.suptitle('Gleichmäßige Dämpfung und Gradienten (gleiches Mittel 0,75 über den festen Weltbereich ≠ gleiche empfangene Lichtmenge); Mittel über 4 Blöcke, Balken = Spannweite\n'
                 + TAG + tag_extra, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    return _save(fig, out_dir, 'Abb3_Nichtuniformitaet', plt)


def fig4(T: pd.DataFrame, Rk: pd.DataFrame, out_dir, tag_extra='') -> str:
    plt = _plt(); fig, axs = plt.subplots(1, 2, figsize=(16, 6.6), squeeze=False)
    envc = {'NOMINAL': '0.2', 'BLUR1_5': '#8c564b', 'BLUR2': '#ff7f0e', 'BLUR3': '#d62728', 'UNIFORM': '#1f77b4', 'GRAD_X': '#9467bd', 'GRAD_Y': '#17becf'}
    act = set(Rk[Rk.activation].row_key) if Rk is not None and len(Rk) else set()
    for j, (fam, meth) in enumerate((('point', 'DF'), ('line', 'P4'))):
        ax = axs[0, j]; q = T[(T.family == fam) & (T.method == meth)]
        for e, g in q.groupby('env'):
            ax.scatter(100 * g.glob_Cvalid.astype(float), g.w.astype(float), s=9, color=envc.get(e, 'k'), label=ENV_DE.get(e, e), alpha=0.75)
        a = q[q.row_key.isin(act)]
        ax.scatter(100 * a.glob_Cvalid.astype(float), a.w.astype(float), s=60, facecolors='none', edgecolors='#e000e0', lw=1.2, label='Aktivierung (Eltern w = 0)')
        n_w = int((q.w.astype(float) > 0).sum()); n_a = len(a)
        if n_w:
            ax.set_yscale('symlog', linthresh=1)
        else:
            ax.set_ylim(-1, 10); ax.text(0.5, 0.6, f'keine falschen Zuordnungen: w = 0 in allen {len(q)} Bildern', ha='center', transform=ax.transAxes, fontsize=11)
        ax.set_xlabel('Cvalid global / %'); ax.set_ylabel('falsche Zuordnungen w (Punkte' + (', symlog)' if n_w else ')'))
        ax.set_title(f"{'Punkte (DF)' if fam == 'point' else 'Linien (P4)'}: alle {len(q)} Bilder; w > 0: {n_w}; Aktivierungen: {n_a}", fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle('Abdeckung und Fehlzuordnungsrisiko je Bild (alle Fälle, auch Fehlschläge); viele w-Punkte einer Linie sind kein Vielfaches unabhängiger Fehler\n' + TAG + tag_extra, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out_dir, 'Abb4_Abdeckung_Risiko', plt)


def _case_arrays(P, case_key: str, method: str):
    import mp3_chain as CH3
    meta, A, st = CH3.pack_read(P, case_key)
    if st != 'read':
        return None
    ev = {k[len(f'{method}__ev__'):]: v for k, v in A.items() if k.startswith(f'{method}__ev__')}
    return dict(img=A['image'], ev=ev, res=(meta.get('results') or {}).get(method) or {})


def _topview(ax, d, dom, ms=2.2):
    gx = dom['x0'] + dom['step'] * np.arange(dom['nx']); gy = dom['y0'] + dom['step'] * np.arange(dom['ny'])
    ax.contourf(gx, gy, dom['mask'].astype(float), levels=[0.5, 1.5], colors=['#ececec']); ax.contour(gx, gy, dom['mask'].astype(float), levels=[0.5], colors=['0.5'], linewidths=0.7)
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(15 * np.cos(th), 15 * np.sin(th), '--', color='0.15', lw=0.9)
    ev = d['ev']; sc = None
    if len(ev.get('pts', [])):
        pts = np.asarray(ev['pts'], float); e = pts[:, 2] - np.asarray(ev['zr'], float); v = np.asarray(ev['valid'], bool) & np.asarray(ev['accepted'], bool) & np.isfinite(e)
        w = np.asarray(ev.get('full_point_wrong', ev.get('point_wrong', np.zeros(len(pts), bool))), bool)
        inr = v & (np.abs(e) <= ERR_LIM); outr = v & (np.abs(e) > ERR_LIM) & ~w
        sc = ax.scatter(pts[inr, 0], pts[inr, 1], c=e[inr], cmap='RdBu_r', vmin=-ERR_LIM, vmax=ERR_LIM, s=ms, lw=0)
        ax.scatter(pts[outr, 0], pts[outr, 1], marker='+', s=max(7, 3 * ms), c='k', lw=0.6)
        ax.scatter(pts[v & w, 0], pts[v & w, 1], marker='x', s=max(14, 3 * ms), c='#e000e0', lw=0.8)
    else:
        ax.text(0, 0, 'keine gültige Ausgabe (Cτ = 0)', ha='center', va='center', fontsize=9)
    ax.set_xlim(-47, 47); ax.set_ylim(-47, 47); ax.set_aspect('equal'); ax.set_xlabel('x / mm', fontsize=8); ax.set_ylabel('y / mm', fontsize=8)
    return sc


def _metrics_txt(r: dict, fam: str) -> str:
    m = r.get('metrics') or {}; fi = ((r.get('full_identity') or {}).get('counts') or {}).get('all') if fam == 'line' else None
    c, w, u = (fi['c'], fi['w'], fi['u']) if fi else (m.get('id_correct'), m.get('id_wrong'), m.get('id_unconf'))
    rm = m.get('glob_pt_rmse')
    f = lambda x: f'{100 * x:.1f}'.replace('.', ',') if x is not None else 'undef.'
    return (f"Cτ glob./lok. {f(m.get('glob_Ctau'))} / {f(m.get('loc_Ctau'))} %; gültig {m.get('n_valid')}\n"
            f"Punkt-RMSE {('%.3f' % rm).replace('.', ',') + ' mm' if rm is not None else 'undef.'}; c/w/u {c}/{w}/{u}")


def fig5(P, out_dir, cases=None, tag_extra='') -> str:
    import mp5_analysis as AN
    import s1r_common as S1K
    plt = _plt(); dom = S1K.global_domain(); cases = cases or AN.BOARD_CASES
    fig = plt.figure(figsize=(19, 21)); gs = fig.add_gridspec(4, 5, width_ratios=[1, 1, 1, 1, 0.05], hspace=0.32, wspace=0.18); sc = None
    for i, cfg in enumerate(('D_r1_p8x4', 'L_w1.5_s8')):
        meth = 'DF' if cfg.startswith('D_') else 'P4'; fam = 'point' if cfg.startswith('D_') else 'line'
        for j, env in enumerate(('NOMINAL', 'BLUR3', 'UNIFORM', 'GRAD_Y')):
            ck = f'{cfg}|M10_curv|b0|s40|r1|{env}'
            d = _case_arrays(P, ck, meth); ax = fig.add_subplot(gs[2 * i, j]); ax2 = fig.add_subplot(gs[2 * i + 1, j])
            if d is None:
                ax.text(0.5, 0.5, 'nicht vorhanden', ha='center'); ax.axis('off'); ax2.axis('off'); continue
            ax.imshow(d['img'], cmap='gray', vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{CFG_DE[cfg]} ({meth}), {ENV_DE[env]}\nKamerabild (Grauwerte 0–255)", fontsize=9.5)
            s_ = _topview(ax2, d, dom, ms=(9.0 if fam == 'point' else 2.2)); sc = sc or s_
            ax2.set_title(_metrics_txt(d['res'], fam), fontsize=8.5)
    if sc is not None:
        for r_ in (1, 3):
            cb = fig.colorbar(sc, cax=fig.add_subplot(gs[r_, 4])); cb.set_label('z − z_ref / mm (±0,5 mm)')
    fig.suptitle('Typische Topographie (vorab gewählt): M10, Phasenblock 0, σ = 40, Realisierung 1; grau = globaler Bewertungsbereich, Kreis = lokal r ≤ 15 mm; '
                 '+ = |Fehler| > 0,5 mm, magenta × = falsche Zuordnung\n' + TAG + tag_extra, fontsize=10.5)
    return _save(fig, out_dir, 'Abb5_Typische_Topographie', plt)


def fig6(P, case_key: str | None, parent_key: str | None, out_dir, tag_extra='') -> str | None:
    if case_key is None:
        return None
    import s1r_common as S1K
    plt = _plt(); dom = S1K.global_domain()
    fig, axs = plt.subplots(2, 2, figsize=(13, 13), squeeze=False); sc = None
    for j, (ck, lab) in enumerate(((parent_key, 'Elternfall (nominal)'), (case_key, 'Fall mit größtem w-Anstieg'))):
        d = _case_arrays(P, ck, 'P4'); ax = axs[0, j]
        ax.imshow(d['img'], cmap='gray', vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{lab}\n{ck}', fontsize=9)
        s_ = _topview(axs[1, j], d, dom); sc = sc or s_
        axs[1, j].set_title(_metrics_txt(d['res'], 'line'), fontsize=8.5)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axs[1, :].tolist(), fraction=0.03, pad=0.02); cb.set_label('z − z_ref / mm (Farbskala ±0,5 mm)')
    fig.suptitle('Fehlerfall nach Regel (größter Anstieg falscher Zuordnungen gegenüber dem eigenen Elternfall): nachträgliche Diagnose, kein Häufigkeitsnachweis\n' + TAG + tag_extra, fontsize=10)
    return _save(fig, out_dir, 'Abb6_Fehlerfall', plt)


def all_figures(P, T, S, Lv, Rk, out_dir, preview=False) -> dict:
    import mp5_analysis as AN
    extra = ' · VORSCHAU (Einstiegs-Teilmenge, unvollständige Blöcke)' if preview else ''
    out = dict(F1=fig1(Lv, out_dir, extra), F2=fig2(S, out_dir, extra), F3=fig3(S, Lv, out_dir, extra), F4=fig4(T, Rk, out_dir, extra), F5=fig5(P, out_dir, None, extra))
    fk = AN.select_failure(Rk)
    if fk is not None:
        par = T[T.case_key == fk].parent_case_key.iloc[0]
        out['F6'] = fig6(P, fk, par, out_dir, extra)
    else:
        out['F6'] = None
    out['failure_case'] = fk
    return out
