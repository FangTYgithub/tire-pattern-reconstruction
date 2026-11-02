# Plot joint-study effects and case comparisons from validated tables.

from __future__ import annotations

import numpy as np
import pandas as pd

import s5v_common as K
import s5u_figs as FG

GID = 'data'


def _save(fig, name) -> str:
    p = K.guard(K.FIGS / name); p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return K.rel(p)


def count_finite(axes) -> int:
    n = 0
    for ax in np.ravel(axes):
        for ln in ax.get_lines():
            if ln.get_gid() == GID:
                n += int(np.isfinite(np.asarray(ln.get_ydata(), float)).sum())
    return n


def _de(v, nd=1) -> str:
    return '–' if v is None or pd.isna(v) else f'{float(v):.{nd}f}'.replace('.', ',')


def f2(S: pd.DataFrame) -> dict:
    from matplotlib.ticker import LogLocator, NullFormatter
    plt, CF = FG._plt(); combos = FG.LINES; fe = 'P4'
    rows = [('glob_Ctau_mean8', 'C$_τ$ global / %', 100, 'lin'), ('loc_Ctau_mean8', 'C$_τ$ lokal (r ≤ 15 mm) / %', 100, 'lin'),
            ('glob_pt_rmse_median', 'Punkt-RMSE global\n(Median def. Fälle) / mm', 1, 'log'), ('sup_surf_rmse_median', 'Flächen-RMSE gestützt\n(Median def. Fälle) / mm', 1, 'log')]
    fig, axs = plt.subplots(len(rows), 3, figsize=(6.5, 9.0), sharex=True)
    for j, s in enumerate(FG.SURF):
        for i, (m, ylab, f, sc) in enumerate(rows):
            ax = axs[i, j]
            for c in combos:
                g = S[(S.combo == c) & (S.surface == s) & (S.front_end == fe)].set_index('sigma').reindex(FG.SIG)
                y = pd.to_numeric(g[m], errors='coerce') * f; st = FG.style(c)
                ax.plot(FG.SIG, y, marker='o', ms=2.6, lw=0.9, label=FG.lab(c), gid=GID, **st)
                if m.endswith('_median'):
                    nd = g[m.replace('_median', '_n_defined')]; part = (nd > 0) & (nd < g.n_cases)
                    ax.plot(np.array(FG.SIG)[part.to_numpy()], y[part].to_numpy(), 'o', mfc='white', mec=st['color'], ms=3.2, lw=0)
            if sc == 'log':
                ax.set_yscale('log'); ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0))); ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_ylim(-3, 103)
            ax.grid(lw=0.3, alpha=0.5); ax.yaxis.set_major_formatter(CF)
            if i == 0:
                ax.set_title(FG.SH[s])
            if j == 0:
                ax.set_ylabel(ylab, fontsize=6.3)
            if i == len(rows) - 1:
                ax.set_xlabel('Rauschen σ'); ax.set_xticks(FG.SIG)
    h, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, ncol=3, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=6)
    fig.text(0.02, 0.012, 'Linienmuster, Frontend P4; b = Linienbreite, s = Linienabstand in mm.\n'
             'C$_τ$: 8-Phasen-Mittel (Wiederholungen je Phase gemittelt, kein Ausgang = 0). Fehler: Median der Fälle mit definiertem Wert;\n'
             'hohler Marker = nicht alle Fälle definiert, fehlender Punkt = kein definierter Fall. Gemessene Stufen, keine Interpolation.', fontsize=5.6, va='bottom')
    fig.tight_layout(rect=(0, 0.065, 1, 0.955))
    n = count_finite(axs)
    return dict(file=_save(fig, 'F2_Linien_Gesamtbereich.png'), n_plotted=n)


def f3(E: pd.DataFrame) -> dict:
    plt, CF = FG._plt()
    sets = [('dots_parallel', 'registered', 'D0', 'Punkte (r 1,0 / 1,5 mm): Abstand längs 6 → 10 mm'),
            ('dots_perp', 'registered', 'D0', 'Punkte (r 1,0 / 1,5 mm): Abstand quer 4 → 6 mm'),
            ('lines', 'registered', 'P4', 'Linien (b 0,5 / 1,5 mm): Abstand 4,45 → 8 mm'),
            ('lines_6.3_8', 'posthoc', 'P4', 'Linien (b 0,5 / 1,5 mm): Abstand 6,3 → 8 mm – nachträglich ergänzt, nicht vorregistriert')]
    fig, axs = plt.subplots(4, 3, figsize=(6.5, 8.8), sharex=True)
    labels = {'E_small': 'E$_{klein}$ (kleinere Größe)', 'E_large': 'E$_{groß}$ (größere Größe)', 'I': 'I = E$_{groß}$ − E$_{klein}$'}
    for i, (setn, an, fe, title) in enumerate(sets):
        for j, s in enumerate(FG.SURF):
            ax = axs[i, j]
            for eff, col, dx in (('E_small', '#1f77b4', -2.2), ('E_large', '#d62728', 0.0), ('I', 'k', 2.2)):
                g = E[(E.set == setn) & E.analysis.str.startswith(an) & (E.front_end == fe) & (E.surface == s) & (E.effect == eff) & (E.metric == 'glob_Ctau') & (E.record_type == 'interaction')]
                g = g[g.sigma.isin(K.SIG_LABELS)].copy(); g['sg'] = g.sigma.astype(float); g = g.set_index('sg').reindex(FG.SIG)
                y = 100 * pd.to_numeric(g['mean'], errors='coerce'); lo = 100 * pd.to_numeric(g['res_lo_5'], errors='coerce'); hi = 100 * pd.to_numeric(g['res_hi_95'], errors='coerce')
                cont = ax.errorbar(np.array(FG.SIG) + dx, y, yerr=[y - lo, hi - y], fmt='o-' if eff == 'I' else 'o--', color=col, ms=2.6, lw=0.9, capsize=1.5, label=labels[eff])
                cont[0].set_gid(GID)
            ax.axhline(0, color='0.4', lw=0.5); ax.grid(lw=0.3, alpha=0.5); ax.yaxis.set_major_formatter(CF)
            if j == 0:
                ax.set_ylabel('Δ C$_τ$ global / pp', fontsize=6.3)
            ax.set_title(f'{title}\n{FG.SH[s]}' if j == 1 else f'\n{FG.SH[s]}', fontsize=6.2)
            if i == 3:
                ax.set_xlabel('Rauschen σ'); ax.set_xticks(FG.SIG)
    h, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l_, ncol=3, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.02, 0.01, 'E = C$_τ$(größerer Abstand) − C$_τ$(kleinerer Abstand) bei gleicher Punkt-/Liniengröße; E > 0: der größere Abstand erreicht mehr Abdeckung.\n'
             'I = E$_{groß}$ − E$_{klein}$. Phasenmittel (Wiederholungen je Phase gemittelt); Balken = deskriptives 5–95-%-Intervall (Phasen-Resampling 10 000×,\n'
             'gleiche Indizes für alle Zellen); Punkte D0, Linien P4; kein Signifikanztest. Gemeinsames Grauwert-Integral des Musters (Q = 11695), keine Lichtleistung.', fontsize=5.4, va='bottom')
    fig.tight_layout(rect=(0, 0.06, 1, 0.965))
    return dict(file=_save(fig, 'F3_Vierfelder_Abwaegung.png'), n_plotted=count_finite(axs))


def f4(FC: pd.DataFrame, C: pd.DataFrame, expect_top: int = 324, save: bool = True) -> dict:
    if FC.sigma.dtype.kind != 'f':
        raise K.Stop('refused', f'front-end table sigma is {FC.sigma.dtype}, not numeric: read it with s5v_effects.load_paired')
    plt, CF = FG._plt()
    fig, axs = plt.subplots(4, 3, figsize=(6.5, 8.8), sharex=True)
    d = C[C.family == 'point']
    for j, s in enumerate(FG.SURF):
        for c in FG.DOTS:
            g = FC[(FC.combo == c) & (FC.surface == s)].set_index('sigma').reindex(FG.SIG); st = FG.style(c)
            axs[0, j].plot(FG.SIG, 100 * pd.to_numeric(g.d_glob_Ctau_mean8, errors='coerce'), marker='o', ms=2.4, lw=0.8, label=FG.lab(c), gid=GID, **st)
            axs[1, j].plot(FG.SIG, pd.to_numeric(g.d_n_valid_mean8, errors='coerce'), marker='o', ms=2.4, lw=0.8, gid=GID, **st)
        for fe, ls in (('D0', '-'), ('DF', '--')):
            for num, den, row in ((('id_wrong',), ('id_correct', 'id_wrong'), 2), (('id_unconf',), ('id_correct', 'id_wrong', 'id_unconf'), 3)):
                y = []
                for sg in FG.SIG:
                    g = d[(d.surface == s) & (d.front_end == fe) & (d.sigma == sg)]
                    nu = sum(pd.to_numeric(g[x]).sum() for x in num); de = sum(pd.to_numeric(g[x]).sum() for x in den)
                    y.append(100 * nu / de if de else np.nan)
                axs[row, j].plot(FG.SIG, y, ls, marker='s', ms=2.6, color='k' if fe == 'D0' else '#ff7f0e', lw=1, label=fe, gid=GID)
        axs[0, j].set_title(FG.SH[s])
    for i, yl in enumerate(('Δ C$_τ$ global (DF − D0) / pp', 'Δ gültige Punkte, alle\n(DF − D0)', 'falsch bestätigt w/(c+w) / %', 'nicht bestätigbar u/(c+w+u) / %')):
        axs[i, 0].set_ylabel(yl, fontsize=6.2)
        for j in range(3):
            axs[i, j].grid(lw=0.3, alpha=0.5); axs[i, j].axhline(0, color='0.4', lw=0.5); axs[i, j].yaxis.set_major_formatter(CF)
    for j in range(3):
        axs[3, j].set_xlabel('Rauschen σ'); axs[3, j].set_xticks(FG.SIG)
    h, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l_, ncol=3, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=5.8)
    axs[2, 0].legend(frameon=False, fontsize=6)
    fig.text(0.02, 0.01, 'Gleiche Bilder, nur das Frontend unterscheidet sich (D0 / eingefrorenes DF). Zeilen 1–2: je Konfiguration Phasenmittel der Paardifferenzen\n'
             '(Wiederholungen je Phase gemittelt); gültige Punkte = alle gültigen Rekonstruktionspunkte ohne Bereichsfilter. Zeilen 3–4: über alle 9 Punktkonfigurationen\n'
             'gepoolte Anteile der akzeptierten Punkte (c korrekt, w falsch bestätigt, u nicht bestätigbar; Nenner c+w bzw. c+w+u); Nenner 0 → kein Punkt.', fontsize=5.4, va='bottom')
    fig.tight_layout(rect=(0, 0.06, 1, 0.945))
    n_top = count_finite(axs[0:2, :]); n_zero = sum(int((np.asarray(ln.get_ydata(), float) == 0).sum()) for ax in np.ravel(axs[0:2, :]) for ln in ax.get_lines() if ln.get_gid() == GID)
    if n_top != expect_top:
        import matplotlib.pyplot as plt2
        plt2.close(fig)
        raise K.Stop('refused', f'F4 rows 1-2 carry {n_top} finite plotted values, expected {expect_top} - figure not saved')
    res = dict(n_top_finite=n_top, n_top_exact_zero=n_zero, n_rows34_finite=count_finite(axs[2:4, :]))
    if not save:
        plt.close(fig); return res
    return dict(file=_save(fig, 'F4_Frontend_Identitaetsrisiko.png'), **res)


F4B_ROWS = (('glob_pt_rmse', 'Δ Punkt-RMSE global\n(DF − D0, Median def. Paare) / mm'), ('sup_surf_rmse', 'Δ Flächen-RMSE gestützt\n(DF − D0, Median def. Paare) / mm'))


def f4b(FC: pd.DataFrame, save: bool = True) -> dict:
    if FC.sigma.dtype.kind != 'f':
        raise K.Stop('refused', f'front-end table sigma is {FC.sigma.dtype}, not numeric: read it with s5v_effects.load_paired')
    plt, CF = FG._plt()
    fig, axs = plt.subplots(len(F4B_ROWS), 3, figsize=(6.5, 5.0), sharex=True)
    for j, s in enumerate(FG.SURF):
        for c in FG.DOTS:
            g = FC[(FC.combo == c) & (FC.surface == s)].set_index('sigma').reindex(FG.SIG); st = FG.style(c)
            for i, (m, _) in enumerate(F4B_ROWS):
                y = pd.to_numeric(g[f'd_{m}_median_defined_pairs'], errors='coerce'); nd = pd.to_numeric(g[f'{m}_n_defined_pairs'], errors='coerce')
                axs[i, j].plot(FG.SIG, y, marker='o', ms=2.4, lw=0.8, label=FG.lab(c) if i == 0 else None, gid=GID, **st)
                part = ((nd > 0) & (nd < pd.to_numeric(g.n_pairs, errors='coerce'))).to_numpy()
                axs[i, j].plot(np.array(FG.SIG)[part], y.to_numpy()[part], 'o', mfc='white', mec=st['color'], ms=3.0, lw=0)
        axs[0, j].set_title(FG.SH[s])
    for i, (_, yl) in enumerate(F4B_ROWS):
        axs[i, 0].set_ylabel(yl, fontsize=6.2)
        for j in range(3):
            axs[i, j].grid(lw=0.3, alpha=0.5); axs[i, j].axhline(0, color='0.4', lw=0.5); axs[i, j].yaxis.set_major_formatter(CF)
    for j in range(3):
        axs[-1, j].set_xlabel('Rauschen σ'); axs[-1, j].set_xticks(FG.SIG)
    h, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l_, ncol=3, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=5.8)
    fig.text(0.02, 0.012, 'Ergänzung zu F4 (nach dem Einfrieren als Darstellung ergänzt): gleiche Bilder, nur das Frontend unterscheidet sich. Median der Paardifferenzen DF − D0\n'
             'über Paare, in denen beide Fehler definiert sind; negativ = DF kleinerer Fehler. Punkt-RMSE global: gültige Punkte im globalen Bewertungsbereich;\n'
             'Flächen-RMSE gestützt: Knoten mit Stützung (d = 6 mm). Hohler Marker = nicht alle Paare definiert, fehlender Punkt = kein definiertes Paar.', fontsize=5.3, va='bottom')
    fig.tight_layout(rect=(0, 0.11, 1, 0.88))
    n = count_finite(axs)
    want = int(sum(pd.to_numeric(FC[f'd_{m}_median_defined_pairs'], errors='coerce').notna().sum() for m, _ in F4B_ROWS))
    if n != want:
        import matplotlib.pyplot as plt2
        plt2.close(fig)
        raise K.Stop('refused', f'F4b plots {n} finite values, the saved pair fields define {want} - figure not saved')
    res = dict(n_plotted=n, n_defined_in_table=want, n_undefined_in_table=2 * len(FC) - want)
    if not save:
        plt.close(fig); return res
    return dict(file=_save(fig, 'F4b_Frontend_Fehler.png'), **res)


# Draw selected joint-study camera images and reconstructed point clouds.
def f_cases(C: pd.DataFrame, sel: dict, arrays) -> list:
    plt, CF = FG._plt(); out = []
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    nz, ne = Normalize(-16, 4), Normalize(-0.5, 0.5)
    jobs = [(fam, c['role'], c['series'], c['front_end']) for fam in ('point', 'line') for c in sel[fam]['cells']] + [('line', 'Diagnose', sel['diagnostic']['series'], 'P4')]
    for fam, role, series, fe in jobs:
        n = len(series); H = 2.0 * n + 1.6; fig = plt.figure(figsize=(6.5, H))
        gs = fig.add_gridspec(n, 4, wspace=0.22, hspace=1.0, left=0.05, right=0.98, top=1 - 0.48 / H, bottom=1.45 / H)
        info = []
        for i, key in enumerate(series):
            r = C[(C.combo == key[0]) & (C.surface == key[1]) & (C.phase_label == key[2]) & (C.sigma == float(key[3])) & (C.rep == int(key[4])) & (C.front_end == fe)]
            if len(r) != 1:
                raise K.Stop('refused', f'case {key} {fe}: {len(r)} rows')
            r = r.iloc[0]; img, z = arrays(r, fe)
            val = z['valid'].astype(bool); P = z['pts'][val]; zt = z['zr'][val].astype(float); e = P[:, 2] - zt if len(P) else np.zeros(0)
            a0 = fig.add_subplot(gs[i, 0]); a0.imshow(img, cmap='gray', vmin=0, vmax=255); a0.set_xticks([]); a0.set_yticks([])
            a0.set_title(f'Kamerabild · {FG.SH[key[1]]} · {FG.lab(key[0])}\n{key[2]} · σ {key[3]:g} · Wdh. {key[4]}', fontsize=5.2, loc='left')
            axk = [fig.add_subplot(gs[i, c]) for c in (1, 2, 3)]
            for k, (vals, norm, cmap, t) in enumerate(((zt, nz, 'viridis', 'wahre Höhe\n(nur an gültigen Punkten)'),
                                                       (P[:, 2] if len(P) else zt, nz, 'viridis', f'rekonstruierte Höhe ({fe})'),
                                                       (np.clip(e, -0.5, 0.5), ne, 'RdBu_r', 'z$_{rek}$ − z$_{wahr}$\n(±0,5 mm begrenzt)'))):
                ax = axk[k]; ax.set_facecolor('#dcdcdc')
                if len(P):
                    ax.scatter(P[:, 0], P[:, 1], c=vals, cmap=cmap, norm=norm, s=(3 if fam == 'point' else 0.5), linewidths=0, rasterized=True)
                else:
                    ax.text(0.5, 0.5, 'keine gültigen\nPunkte', ha='center', va='center', fontsize=6, transform=ax.transAxes)
                ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.set_aspect('equal'); ax.tick_params(labelsize=4.8, labelleft=(k == 0)); ax.set_title(t, fontsize=5.2)
                ax.xaxis.set_major_formatter(CF); ax.yaxis.set_major_formatter(CF)
            n_over = int((np.abs(e) > 0.5).sum()); emax = float(np.abs(e).max()) if len(e) else None
            nv, ng, nl, na = int(r.n_valid), int(r.glob_n_valid), int(r.loc_n_valid), int(r.n_accepted)
            if nv != len(P):
                raise K.Stop('refused', f'case {key}: table n_valid {nv} != saved valid points {len(P)}')
            pct = (lambda v: '–' if pd.isna(v) else f'{100 * float(v):.1f}'.replace('.', ','))
            txt = (f'Punkte: akzeptiert {na} · gültig (alle) {nv} · davon global {ng} · lokal (r ≤ 15 mm) {nl}\n'
                   f'C$_τ$ global {pct(r.glob_Ctau)} % · lokal {pct(r.loc_Ctau)} % · Punkt-RMSE global {_de(r.glob_pt_rmse, 3)} mm ({ng} P.) · lokal {_de(r.loc_pt_rmse, 3)} mm ({nl} P.)\n'
                   f'alle gültigen: |Fehler| > 0,5 mm {n_over} von {nv}, max {_de(emax, 2)} mm · Identität (akzeptierte P.): c {int(r.id_correct)} / w {int(r.id_wrong)} / u {int(r.id_unconf)}')
            fig.text(0.05, axk[0].get_position().y0 - 0.20 / H, txt, fontsize=5.0, va='top')
            info.append(dict(key=key, front_end=fe, n_accepted=na, n_valid=nv, glob_n_valid=ng, loc_n_valid=nl, n_abs_err_gt_0p5=n_over, max_abs_err_mm=emax))
        x1, x3 = axk[0].get_position(), axk[2].get_position(); x2 = axk[1].get_position()
        for rect, norm, cmap, lab_ in (([x1.x0, 0.66 / H, x2.x1 - x1.x0, 0.05 / H], nz, 'viridis', 'Höhe z / mm (Fenster −16…4; Felder 2–3)'),
                                       ([x3.x0, 0.66 / H, x3.x1 - x3.x0, 0.05 / H], ne, 'RdBu_r', 'z$_{rek}$ − z$_{wahr}$ / mm (Feld 4)')):
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(rect), orientation='horizontal')
            cb.set_label(lab_, fontsize=5.2, labelpad=1); cb.ax.tick_params(labelsize=4.8, pad=1); cb.ax.xaxis.set_major_formatter(CF)
        fam_t = 'Punkte (D0)' if fam == 'point' else 'Linien (P4)'
        fig.suptitle(f'{fam_t} – {role}: ' + ('Zelle mit ' + ('höchstem' if role == 'hoch' else 'niedrigstem') + ' Median-C$_τ$ bei σ 40' if role != 'Diagnose' else 'Randanomalie aus S5-ALIGN-04'),
                     fontsize=7, x=0.01, ha='left')
        fig.text(0.01, 0.04 / H, 'Zweidimensionale Draufsicht (x–y-Projektion, mm) der gespeicherten 3D-Punkte; dargestellt sind alle gültigen Punkte. Feld 2 zeigt die wahre Höhe nur an den\n'
                 'Positionen gültiger Punkte, keine vollständige Referenzfläche (siehe F7/F8). Leeres Feld = keine gültigen Punkte (die Oberfläche existiert dennoch).\n'
                 'Gleiche Achsen und Farbskalen in allen Fällen; nachträgliche Auswahl nach festgelegter Regel, keine Validierung.', fontsize=5.0, va='bottom')
        name = f"{'F5_Punkte' if fam == 'point' else 'F6_Linien'}_{role}.png"
        out.append(dict(file=_save(fig, name), cases=info))
    return out
