# Plot detector, precision and illumination diagnostics from saved results.

from __future__ import annotations

import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m6_common as K

W = K.OUT / '_work'
FIG = K.OUT / 'figures'
PSFS = [('NOMINAL', 1.0), ('BLUR1_5', 1.5), ('BLUR2', 2.0), ('BLUR3', 3.0)]
TAG = 'MODEL-PREP-06A · Simulation, feste SURF-02-Geometrie · gesehene Phasenblöcke 0–3 (MODEL-PREP-05) · keine Gerätekalibrierung'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'figure.dpi': 150, 'savefig.dpi': 200, 'font.family': 'DejaVu Sans'})


def de(x, nd=1):
    return 'undef.' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{nd}f}'.replace('.', ',').replace('-', '−')


def _fmt_pp(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: de(v, 0)))


def _save(fig, name):
    fig.text(0.005, 0.003, TAG, fontsize=6.5, color='0.35', ha='left', va='bottom')
    p1 = K.guard(FIG / f'{name}.png'); p2 = K.guard(FIG / f'{name}.pdf')
    fig.savefig(p1, bbox_inches='tight'); fig.savefig(p2, bbox_inches='tight'); plt.close(fig)
    return [K.rel(p1), K.rel(p2)]


def fig1():
    E = pd.read_csv(K.OUT / 'effects.csv', low_memory=False)
    a = E[(E.analysis == 'A1') & E.contrast.str.startswith('Linienbreite') & (E.domain == 'glob')]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    cols = {0.10: '#1b9e77', 0.15: '#000000', 0.20: '#d95f02', 0.30: '#7570b3'}
    for j, sg in enumerate((20.0, 40.0)):
        for i, metric in enumerate(('Ctau', 'Cvalid')):
            ax = axs[i, j]
            for d, ls in ((6.0, '-'), (5.0, '--')):
                taus = K.TAUS if metric == 'Ctau' else (0.15,)
                for tau in taus:
                    s = a[(a.metric == metric) & (a.sigma == sg) & (a.d_mm == d) & (a.tau_mm == tau)]
                    al = s[s.surface == 'alle'].set_index('env'); sf = s[s.surface != 'alle']
                    xs = [p for e, p in PSFS if e in al.index]; ys = [100 * al.loc[e, 'mean'] for e, p in PSFS if e in al.index]
                    lo = [100 * sf[sf.env == e]['mean'].min() for e, p in PSFS if e in al.index]; hi = [100 * sf[sf.env == e]['mean'].max() for e, p in PSFS if e in al.index]
                    off = {0.10: -0.045, 0.15: -0.015, 0.20: 0.015, 0.30: 0.045}[tau] + (0.0 if d == 6.0 else 0.03)
                    c = cols[tau] if metric == 'Ctau' else ('#000000' if d == 6.0 else '#888888')
                    lab = (f'τ = {de(tau, 2)} mm' if metric == 'Ctau' else 'C_valid') + f', d = {int(d)} mm'
                    if metric == 'Ctau' and d == 5.0 and tau != 0.15:
                        continue
                    ax.errorbar(np.array(xs) + off, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], fmt='o', ms=3.2, color=c, ls=ls, lw=0.6, capsize=1.5,
                                label=lab, mfc=('white' if d == 5.0 else c))
            ax.axhline(0, color='0.5', lw=0.6)
            ax.set_title(f"{'C_τ' if metric == 'Ctau' else 'C_valid'} · σ = {int(sg)}")
            ax.set_ylabel('Δ (1,5 mm − 0,5 mm) [pp]'); _fmt_pp(ax)
            if i == 1:
                ax.set_xlabel('wahre PSF [px] (gemessen: 1 / 1,5 / 2 / 3)')
            ax.set_xticks([1, 1.5, 2, 3]); ax.set_xticklabels(['1', '1,5', '2', '3'])
            ax.legend(fontsize=6.3, loc='upper left', frameon=False)
    fig.suptitle('Abb. 1  Linienbreitenvergleich (8 mm, P4) unter verschiedenen Genauigkeitsanforderungen τ und Stützweiten d\n'
                 'Punkte = Mittel der drei Flächen (je 4 Phasenblöcke × 2 Realisierungen); Balken = Spanne der drei Flächenmittel; '
                 'Verbindungslinien nur zur Lesbarkeit', fontsize=8.3)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return _save(fig, 'Abb1_Tau_PSF_Linienbreite')


def fig2():
    A = pd.read_csv(W / 'a2_summary.csv')
    fig, axs = plt.subplots(1, 3, figsize=(10.6, 5.0), gridspec_kw=dict(width_ratios=[1.35, 1.0, 1.0]))
    ax = axs[0]
    conds = [(c, e) for c in ('D_r1.5_p8x4', 'D_r1_p8x6') for e in ('NOMINAL', 'BLUR2', 'BLUR3')]
    keys = [('expected', 'erwartete sichtbare Punkte'), ('n_components', 'Zusammenhangskomponenten'), ('sel_1_rel_area_small', 'abgelehnt: rel. Fläche klein'),
            ('sel_2_rel_area_large', 'abgelehnt: rel. Fläche groß'), ('sel_3_elongation', 'abgelehnt: Streckung'), ('accepted_features', 'akzeptiert'),
            ('recovered_id', 'mit richtiger ID'), ('exp_sharing_nearest_component', 'Verschmelzungsanzeichen*')]
    x = np.arange(len(conds)); wdt = 0.1
    cl = plt.cm.tab10(np.arange(len(keys)))
    for k, (col, lab) in enumerate(keys):
        v = [float(A[(A.config == c) & (A.env == e) & (A.sigma == 20.0)][col].iloc[0]) for c, e in conds]
        ax.bar(x + (k - 3.5) * wdt, v, wdt, color=cl[k], label=lab)
    ax.set_xticks(x); ax.set_xticklabels([('Ø3/8×4' if c.startswith('D_r1.5') else 'Ø2/8×6') + '\n' + {'NOMINAL': 'P1', 'BLUR2': 'P2', 'BLUR3': 'P3'}[e] for c, e in conds], fontsize=7)
    ax.set_ylabel('Anzahl je Bild (Mittel)'); ax.set_title('Merkmalsebene Punkte (DF, σ = 20)\nEinheiten: Punkte bzw. Komponenten je Bild')
    ax.legend(fontsize=5.8, ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    ax = axs[1]
    conds_l = [(c, e) for c in ('L_w0.5_s8', 'L_w1.5_s8') for e in ('NOMINAL', 'BLUR2', 'BLUR3')]
    keys_l = [('expected', 'erwartete Referenzzeilen'), ('detected', 'detektierte Zeilenmerkmale'), ('accepted', 'akzeptiert'), ('recovered_id', 'Referenzzeilen mit richtiger ID')]
    x = np.arange(len(conds_l)); wdt = 0.2
    for k, (col, lab) in enumerate(keys_l):
        v = [float(A[(A.config == c) & (A.env == e) & (A.sigma == 20.0)][col].iloc[0]) for c, e in conds_l]
        ax.bar(x + (k - 1.5) * wdt, v, wdt, color=plt.cm.Set2(k), label=lab)
    ax.set_xticks(x); ax.set_xticklabels([('0,5' if '0.5' in c else '1,5') + '\n' +{'NOMINAL': 'P1', 'BLUR2': 'P2', 'BLUR3': 'P3'}[e] for c, e in conds_l], fontsize=7)
    ax.set_ylabel('Anzahl je Bild (Mittel)'); ax.set_title('Merkmalsebene Linien (P4, σ = 20; Breite in mm)\nEinheit: Bildzeilenabschnitte'); ax.legend(fontsize=5.8, ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    ax = axs[2]
    conds_s = conds + conds_l
    lab_s = [f"{'Ø3/8×4' if c.startswith('D_r1.5') else ('Ø2/8×6' if c.startswith('D_') else ('L 0,5' if '0.5' in c else 'L 1,5'))} {['P1', 'P2', 'P3'][['NOMINAL', 'BLUR2', 'BLUR3'].index(e)]}"
             for c, e in conds_s]
    v = np.array([[float(A[(A.config == c) & (A.env == e) & (A.sigma == 20.0)][k].iloc[0]) for k in ('glob_no_support', 'glob_supported_over_tau', 'glob_Ctau')] for c, e in conds_s]) * 100
    y = np.arange(len(conds_s))
    ax.barh(y, v[:, 2], color='#2c7bb6', label='C_τ (unterstützt, |e| ≤ τ)')
    ax.barh(y, v[:, 1], left=v[:, 2], color='#fdae61', label='unterstützt, |e| > τ')
    ax.barh(y, v[:, 0], left=v[:, 2] + v[:, 1], color='#d7d7d7', label='ohne Stützung (1 − C_valid)')
    ax.set_yticks(y); ax.set_yticklabels(lab_s, fontsize=7); ax.invert_yaxis(); ax.set_xlabel('Anteil der globalen Bewertungsfläche [%]')
    ax.set_title('Flächenebene (gemeinsamer Nenner:\n30947 Knoten, d = 6 mm, τ = 0,15 mm)'); ax.legend(fontsize=5.8, ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    fig.suptitle('Abb. 2  Verlustschichten: Merkmalsebene und Flächenebene getrennt (unterschiedliche Nenner, kein gemeinsamer Trichter)\n'
                 '*erwartete Punkte, deren nächstgelegene Komponente zugleich die nächste eines anderen erwarteten Punktes ist (bildseitige Zuordnung; Anzeichen, kein Kausalbeweis)',
                 fontsize=8.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    return _save(fig, 'Abb2_Verlustschichten')


def fig3():
    L = pd.read_csv(W / 'a3_levels.csv')
    fig, axs = plt.subplots(2, 2, figsize=(8.2, 7.0), sharex=True)
    arm_st = {'N': ('o', '-', 'N: Originalskala, nominale PSF'), 'T': ('v', ':', 'T: nur Schwelle (wahre PSF)'), 'S': ('s', '--', 'S: nur Skala (Knoten s*)'),
              'ST': ('D', '-.', 'ST: Skala + Schwelle')}
    for j, sg in enumerate((20.0, 40.0)):
        ax = axs[0, j]
        for arm, (mk, ls, lab) in arm_st.items():
            s = L[(L.config == 'L_w0.5_s8') & (L.sigma == sg) & (L.arm == arm)].set_index('env')
            xs = [p for e, p in PSFS]; ys = [100 * s.loc[e, 'glob_Ctau'] for e, p in PSFS]
            ax.plot(np.array(xs) + {'N': -0.03, 'T': -0.01, 'S': 0.01, 'ST': 0.03}[arm], ys, marker=mk, ls=ls, lw=0.8, ms=3.5, label=f'0,5 mm, {lab}')
        s = L[(L.config == 'L_w1.5_s8') & (L.sigma == sg) & (L.arm == 'N')].set_index('env')
        ax.plot([p for e, p in PSFS], [100 * s.loc[e, 'glob_Ctau'] for e, p in PSFS], marker='o', color='k', lw=1.2, ms=3.5, label='1,5 mm, N (T/S/ST identisch oder Alias)')
        ax.set_title(f'globale C_τ · σ = {int(sg)}'); ax.set_ylabel('C_τ [%]'); _fmt_pp(ax); ax.set_ylim(-3, 100)
        ax.legend(fontsize=5.6, frameon=True, framealpha=0.85, loc='lower left') if j == 0 else None
        ax = axs[1, j]
        for arm, (mk, ls, lab) in arm_st.items():
            s = L[(L.config == 'L_w0.5_s8') & (L.sigma == sg) & (L.arm == arm)].set_index('env')
            ax.plot(np.array([p for e, p in PSFS]) + {'N': -0.03, 'T': -0.01, 'S': 0.01, 'ST': 0.03}[arm], [s.loc[e, 'margin_mean'] for e, p in PSFS], marker=mk, ls=ls, lw=0.8, ms=3.5,
                    label=f'0,5 mm, {arm}')
            for e, p in PSFS:
                if s.loc[e, 'noise_dominant_share'] < 0.5:
                    ax.plot(p + {'N': -0.03, 'T': -0.01, 'S': 0.01, 'ST': 0.03}[arm], s.loc[e, 'margin_mean'], marker=mk, mfc='white', mec='red', ms=6, ls='none')
        s = L[(L.config == 'L_w1.5_s8') & (L.sigma == sg) & (L.arm == 'N')].set_index('env')
        ax.plot([p for e, p in PSFS], [s.loc[e, 'margin_mean'] for e, p in PSFS], marker='o', color='k', lw=1.2, ms=3.5, label='1,5 mm, N')
        ax.axhline(1.0, color='0.5', lw=0.6, ls='--')
        ax.set_ylabel('Detektionsabstand R / Schwelle'); ax.set_xlabel('wahre PSF [px]')
        ax.set_xticks([1, 1.5, 2, 3]); ax.set_xticklabels(['1', '1,5', '2', '3'])
        ax.set_title(f'Detektionsabstand · σ = {int(sg)}'); ax.legend(fontsize=5.6, frameon=False, loc='upper right') if j == 1 else None
    w_all = int(L.w_sum.sum()); us = L.u_share.max()
    fig.suptitle('Abb. 3  Vier-Arm-Diagnose P4 (gespeicherte Bilder neu dekodiert, wahre PSF bekannt → nur Diagnose, kein Einsatzvorschlag)\n'
                 'unten: rot umrandet = Schwelle überwiegend vom Entwurfsterm bestimmt (sonst Rauschterm); R = analytische Gratantwort unter wahrer PSF\n'
                 f'Mittel der drei Flächen; bestätigte Fehl-IDs in allen Armen: {w_all}; größter mittlerer u-Anteil: {de(100 * us, 2)} %; '
                 'PSF 1: alle Arme identisch (Negativkontrolle)', fontsize=8.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    return _save(fig, 'Abb3_Vier_Arm_Diagnose')


def fig4():
    Bg = pd.read_csv(W / 'a4_bridge.csv'); Bs = pd.read_csv(W / 'a4_bands.csv')
    fig, axs = plt.subplots(1, 3, figsize=(10.6, 4.0), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
    ax = axs[0]
    for env, mk, c in (('UNIFORM', 'o', '#2c7bb6'), ('NOMINAL', 's', '#999999')):
        q = Bg[Bg.env == env]
        ax.scatter(100 * q.s5_at_sigma_eq_world, 100 * q.mp5_level, marker=mk, s=16, color=c, label=('g = 0,75, Abschnitt 3.5 bei σ/0,75' if env == 'UNIFORM' else 'nominal, Abschnitt 3.5 bei gleichem σ'))
    ax.plot([0, 100], [0, 100], color='0.5', lw=0.6)
    ax.set_xlabel('Abschnitt 3.5 (S5-ALIGN-05), interpoliert [%]'); ax.set_ylabel('MODEL-PREP-05, gemessen [%]')
    ax.set_title('Brücke: aggregierte globale C_τ\n(8 Konfigurationen × σ 20 / 40; keine Fallpaare)'); ax.legend(fontsize=6, frameon=False)
    for k, (cfg, sg, ttl) in enumerate((('L_w0.5_s8', 40.0, 'Linie 0,5 / 8 mm, σ = 40'), ('D_r1_p8x6', 40.0, 'Punkte Ø2 / 8×6, σ = 40'))):
        ax = axs[k + 1]
        q = Bs[(Bs.config == cfg) & (Bs.gradient == 'GRAD_Y') & (Bs.sigma == sg)].sort_values('band')
        xc = (q.g_lo + q.g_hi) / 2
        ax.plot(xc, 100 * q.rec_grad_rate_mean, 'o-', color='#d7191c', lw=0.9, ms=3.5, label='Gradient y')
        ax.plot(xc, 100 * q.rec_uniform_rate_mean, 's--', color='#2c7bb6', lw=0.9, ms=3.5, label='gleichmäßig g = 0,75')
        ax.plot(xc, 100 * q.rec_nominal_rate_mean, '^:', color='0.3', lw=0.9, ms=3.5, label='nominal g = 1')
        ax.fill_between(xc, 100 * q.rec_grad_rate_min, 100 * q.rec_grad_rate_max, color='#d7191c', alpha=0.12, lw=0)
        ax.set_xlabel('lokale Verstärkung g (Gradientenfeld)'); ax.set_ylabel('Wiedergewinnung, richtige ID [%]')
        ax.set_title(f'{ttl}\n(gleiche erwartete Positionen in allen drei Fällen)'); ax.legend(fontsize=6, frameon=False); _fmt_pp(ax)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: de(v, 2)))
    fig.suptitle('Abb. 4  Beleuchtung: Brücke zu Abschnitt 3.5 und Bänder gleicher lokaler Verstärkung (y-Gradient)\n'
                 'Band = Verstärkung des Gradientenfeldes; Fläche = Spanne der vier Blockwerte des Gradienten; nominale Kurve = geometrisch bedingtes Eigenprofil', fontsize=8.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    return _save(fig, 'Abb4_Beleuchtung_Bruecke_Baender')


def fig5():
    B = pd.read_csv(W / 'b_pairs.csv'); F = pd.read_csv(W / 'b_warning_rule.csv')
    fig = plt.figure(figsize=(11.0, 5.0)); gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])
    ax = fig.add_subplot(gs[0])
    c = F[F.level.str.startswith('Zelle')].copy(); c['lab'] = [f"{ {'NOMINAL': 'P1', 'BLUR1_5': 'P1,5', 'BLUR2': 'P2', 'BLUR3': 'P3'}[e]}\nσ{int(s)}" for e, s in zip(c.env, c.sigma)]
    order = [(e, s) for s in (20.0, 40.0) for e, _ in PSFS]
    c = c.set_index(['env', 'sigma']).loc[order].reset_index()
    x = np.arange(len(c))
    ax.bar(x - 0.18, 100 * c.dw_1x1, 0.36, color='#bbbbbb', label='1×1 (Elternbild)'); ax.bar(x + 0.18, 100 * c.dw_3x3, 0.36, color='#2c7bb6', label='3×3')
    s = F[F.level.str.startswith('Fl')]
    for i, (e, sg) in enumerate(order):
        q = s[(s.env == e) & (s.sigma == sg)]
        ax.scatter(np.full(len(q), i + 0.18), 100 * q.dw_3x3, s=8, color='k', zorder=3)
        if q.change_ge_5pp.any():
            ax.annotate('Warnung (Fläche)', (i + 0.18, 100 * q.dw_3x3.max()), xytext=(i - 0.9, 100 * q.dw_3x3.max() + 14), fontsize=6, color='red', arrowprops=dict(arrowstyle='->', color='red', lw=0.6))
    ax.axhline(0, color='0.5', lw=0.6); ax.set_xticks(x); ax.set_xticklabels(c.lab, fontsize=7)
    ax.set_ylabel('Δ C_τ (1,5 − 0,5 mm) [pp]'); _fmt_pp(ax); ax.legend(fontsize=6.3, frameon=False, loc='upper left')
    ax.set_title('Linienbreitenunterschied, global\nBalken: 3 Flächen × Blöcke 0/2 · Punkte: Flächen (3×3)', fontsize=7.8)

    B['ad'] = (B.glob_Ctau_3 - B.glob_Ctau_1).abs()
    r = B.sort_values(['ad', 'case_key'], ascending=[False, True]).iloc[0]
    cfg, env, sg = r.config, r.env, int(r.sigma)
    pf = K.OUT / 'data' / 'b' / f'{cfg}__{env}__s{sg}.npz'; tag = f"{r.surface}__b{int(r.phase_block)}"
    with np.load(pf, allow_pickle=False) as z:
        ev3 = {k[len(f'{tag}__ev__'):]: z[k] for k in z.files if k.startswith(f'{tag}__ev__')}
    _, A1 = K.read_pack(r.case_key, prefixes=('P4__ev__',)); ev1 = K.sub(A1, 'P4__ev__')
    lim = 0.5
    for k, (ev, ttl) in enumerate(((ev1, '1×1 (MODEL-PREP-05)'), (ev3, '3×3'))):
        ax = fig.add_subplot(gs[k + 1])
        ref = np.asarray(ev['ref_xyz'], float).reshape(-1, 3); ax.scatter(ref[:, 0], ref[:, 1], s=0.15, color='0.82', lw=0, label='Referenzzeilen (Wahrheit)')
        p = np.asarray(ev['pts'], float).reshape(-1, 3); v = np.asarray(ev['valid'], bool); e = p[:, 2] - np.asarray(ev['zr'], float)
        sc = ax.scatter(p[v, 0], p[v, 1], c=np.clip(e[v], -lim, lim), s=1.6, cmap='coolwarm', vmin=-lim, vmax=lim, lw=0)
        ww = np.asarray(ev.get('full_point_wrong', np.zeros(len(p), bool)), bool) & v
        if ww.any():
            ax.scatter(p[ww, 0], p[ww, 1], s=4, color='#e000e0', lw=0, label='bestätigte Fehl-ID')
        ax.set_aspect('equal'); ax.set_xlim(-46, 46); ax.set_ylim(-46, 46); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
        g1 = r.glob_Ctau_1 if k == 0 else r.glob_Ctau_3; rm = r.glob_pt_rmse_1 if k == 0 else r.glob_pt_rmse_3
        ax.set_title(f'{ttl}\nC_τ = {de(100 * g1, 1)} %, RMSE = {de(rm, 3)} mm, gültig: {int(v.sum())}', fontsize=7.5)
        if k == 1:
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02); cb.set_label('Höhenfehler e [mm] (Skala ±0,5)')
    fig.suptitle(f'Abb. 5  Teilstrahlprüfung (1×1 vs. 3×3 Strahlen je Projektorpixel, gleiches Rauschfeld, P4 unverändert)\n'
                 f'Fallbeispiel nach größter |Δ C_τ| (nachträglich gewählt, kein typischer Mittelwert): {cfg.replace("_", " ")}, {env}, σ = {sg}, {r.surface}, Block {int(r.phase_block)}', fontsize=7.8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    return _save(fig, 'Abb5_Teilstrahlpruefung'), dict(case=r.case_key, d_glob_Ctau=float(r.glob_Ctau_3 - r.glob_Ctau_1))


def all_figures() -> dict:
    out = dict(Abb1=fig1(), Abb2=fig2(), Abb3=fig3(), Abb4=fig4())
    f5, case = fig5(); out['Abb5'] = f5; out['Abb5_case'] = case
    K.protocol_put('figures', dict(t=K.now(), files=out))
    return out
