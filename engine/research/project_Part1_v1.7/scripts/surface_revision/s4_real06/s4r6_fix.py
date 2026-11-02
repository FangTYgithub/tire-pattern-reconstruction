# Check layout result records and derived analysis inputs.

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import s4r6_common as H

K = H.K
ALPHA_STAR, BETA = 0.80, 1.15


def a1() -> dict:
    t0 = time.time()
    R = pd.read_csv(H.reg(H.R5 / '_parts' / 'B_rasters.csv'), low_memory=False)
    reg = json.loads(H.reg(H.R2 / 'layouts.json').read_text(encoding='utf-8'))
    pr = {n: (float(reg['baselines'][n]['p0_mm']), float(reg['baselines'][n]['h0_mm'])) for n in ('R8x4', 'L6.3')}
    rows = []
    for r in R[R.layout != 'regular'].itertuples():
        p0, h0 = pr[r.baseline]
        s2 = bool(pd.notna(r.g_min_raster_mm) and r.g_min_raster_mm >= ALPHA_STAR * p0 - 1e-12); s3 = bool(pd.notna(r.h_raster_mm) and r.h_raster_mm <= BETA * h0 + 1e-12)
        for a in str(r.arms).split(';'):
            ok = {'S2*': s2, 'S3': s3, 'S4*': s2 and s3}.get(a)
            if ok is None:
                continue
            cont = {'S2*': r.mask_S2star, 'S3': r.mask_S3, 'S4*': r.mask_S4star}[a]
            rows.append(dict(record_type='A1_strategy_raster', baseline=r.baseline, layout=r.layout, phase=r.phase, arm=a, raster_key=r.raster_key, continuous_ok=bool(cont),
                             raster_ok=bool(ok), g_min_raster_mm=r.g_min_raster_mm, h_raster_mm=r.h_raster_mm, alpha_p0=ALPHA_STAR * p0, beta_h0=BETA * h0,
                             violation=bool(cont and not ok), source='outputs/surface_revision/s4_real05/_parts/B_rasters.csv'))
    D = pd.DataFrame(rows); V = D[D.violation]
    by_arm = V.groupby('arm').size().to_dict(); uniq = V.drop_duplicates('raster_key')
    s3_keys = set(V[V.arm == 'S3'].raster_key)
    out = dict(n_strategy_raster_rows=len(D), violations_by_arm={a: int(by_arm.get(a, 0)) for a in ('S2*', 'S3', 'S4*')}, n_violation_records=len(V), n_unique_rasters=int(len(uniq)),
               by_family_unique=uniq.groupby('baseline').size().to_dict(), S4star_sharing_S3_raster=int(V[(V.arm == 'S4*') & V.raster_key.isin(s3_keys)].shape[0]),
               not_in_S3=uniq[~uniq.raster_key.isin(s3_keys)][['baseline', 'layout', 'phase', 'h_raster_mm', 'beta_h0']].to_dict('records'),
               anchor=dict(expected=dict(S2star=0, S3=9, S4star=4, records=13, unique=11), ok=bool(by_arm.get('S2*', 0) == 0 and by_arm.get('S3', 0) == 9 and by_arm.get('S4*', 0) == 4 and len(V) == 13 and len(uniq) == 11)),
               policy='generation keeps the continuous-coordinate definition; these are pixel-quantisation diagnostics, no reselection, no threshold change',
               seconds=round(time.time() - t0, 1), t_local=K.now_local())
    H.part('A1', D); H.run_put('A1', out); H.flush_reads('A1')
    return out


def a2() -> dict:
    t0 = time.time(); r5 = json.loads(H.reg(H.R5 / 'run.json').read_text(encoding='utf-8'))
    M = pd.read_csv(H.reg(H.R5 / 'replay_metrics.csv'), low_memory=False, float_precision='round_trip'); M = M[M.domain == 'glob']
    rows = []
    for x in M.itertuples():
        ej = json.loads(H.reg(H.R5 / 'evals' / f'{x.new_decode}.json').read_text(encoding='utf-8')); m = ej['metrics']
        z = K.load_npz(H.reg(H.R5 / 'evals' / f'{x.new_decode}.npz')); st = np.asarray(z['status'], np.int64)
        old_sum = int(sum(int(v) for k, v in m.items() if k.startswith('n_status_')))
        n11 = int((st == 11).sum()); new_sum = old_sum + n11
        unchanged = dict(c=int(m['glob_id_correct']) == int(x.new_c), w=int(m['glob_id_wrong']) == int(x.new_w), u=int(m['glob_id_unconf']) == int(x.new_u),
                         Ctau=float(m['glob_Ctau']) == float(x.new_Ctau), n_accepted=int(m['n_accepted']) == int(np.asarray(z['accepted'], bool).sum()))
        rows.append(dict(record_type='A2_eval', eval_record=f'outputs/surface_revision/s4_real05/evals/{x.new_decode}.json', image_key=x.image_key, surface=x.surface, phase=x.phase,
                         sigma=x.sigma, role=x.role, n_detected=int(m['n_detected']), status_sum_stored=old_sum, gap_stored=int(m['n_detected']) - old_sum,
                         n_status_clipped_source_abstain=n11, status_sum_new=new_sum, closed_new=bool(new_sum == int(m['n_detected'])), gap_equals_status11=bool(int(m['n_detected']) - old_sum == n11),
                         **{f'unchanged_{k}': v for k, v in unchanged.items()}))
    D = pd.DataFrame(rows)
    out = dict(n_evals=len(D), evals_with_gap=int((D.gap_stored > 0).sum()), gap_total=int(D.gap_stored.sum()), status11_total=int(D.n_status_clipped_source_abstain.sum()),
               all_gaps_equal_status11=bool(D.gap_equals_status11.all()), all_closed_new=bool(D.closed_new.all()),
               metrics_unchanged=bool(D[[c for c in D.columns if c.startswith('unchanged_')]].all().all()),
               anchor=dict(expected=dict(evals_with_gap=9, total=1192), ok=bool((D.gap_stored > 0).sum() == 9 and D.gap_stored.sum() == 1192)),
               note='field added on the report / new evaluation layer only; historical evaluation JSON not rewritten, no re-decoding', seconds=round(time.time() - t0, 1), t_local=K.now_local())
    H.part('A2', D); H.run_put('A2', out); H.flush_reads('A2')
    return out


def a3() -> dict:

    r5 = json.loads(H.reg(H.R5 / 'run.json').read_text(encoding='utf-8'))
    snap = r5['config_C']['harness']; fin = r5['final']['final_code']
    diff = sorted(k for k in set(snap) | set(fin) if snap.get(k) != fin.get(k))
    t = {s: (r5[s].get('t_local') if isinstance(r5.get(s), dict) else None) for s in ('config_C', 'C_regression', 'C_stage1', 'C_stage2', 'C_gate5', 'C_pairs', 'A2_reserve', 'figures', 'final')}
    changes = [
        dict(file='scripts/surface_revision/s4_real05/s4r5_replay.py', layer='execution framework + statistics',
             change=['decode_one: return path for an already complete record (dict(...) with duplicate identity key raised TypeError) fixed',
                     'pair_metrics() with CATS / _cats / _dom added (old / new pair table, compact npz)'],
             stage='after C_stage2 (02:43:07): fix before the gate-5 call (C_gate5 02:43:30); pair_metrics before C_pairs (02:44:10)',
             affects='no decode or evaluation array: the failing path ran before any budget reservation; replay_metrics.csv / replay_compact.npz are produced by pair_metrics',
             recheck='reference recomputed the 72 pairs from the raw arrays (counts identical); A2 below re-reads the stored evaluations'),
        dict(file='scripts/surface_revision/s4_real05/s4r5_stats.py', layer='statistics / registration',
             change=['phase_scan() + REG_NAME / PER_RECORD added (targeted register scan replacing the stopped full JSON scan)', 'a2_history: nearest-used phase with provenance; own output folder excluded from the scan'],
             stage='after C_pairs (02:44:10), before the final A2_reserve run (02:51:12); exact edit times not recoverable',
             affects='run.json A2_reserve, _parts/A2_history.csv, _parts/phase_scan.json only; a1 / a2_counts / a3 code unchanged after their runs (02:27:53-02:28:01)',
             recheck='A1 S2 median and A3 totals re-derived in this round from stored inputs (a3_recheck)'),
        dict(file='scripts/surface_revision/s4_real05/s4r5_main.py', layer='execution framework (dispatcher)',
             change=['commands c_metrics, a2s, figs, assemble, final added; assemble() / final() added'],
             stage='after C_stage2: c_metrics before 02:44:10, a2s before 02:51:12, figs before 02:53:29, assemble / final at the end',
             affects='dispatch and packaging (checks.csv, final hashes); no method or evaluation code path', recheck='checks.csv rows re-read in A2 / A1'),
        dict(file='scripts/surface_revision/s4_real05/s4r5_figs.py', layer='display', change=['new file after the snapshot'], stage='before figures (02:53:29)', affects='figures only', recheck='visual check')]
    out = dict(files_differing=diff, method_core_unchanged=bool(all(r5['config_C']['code'][k] == H.code_hashes([k])[k] for k in r5['config_C']['code'])),
               section_times=t, changes=changes, note='the initial snapshot is not back-filled; exact minute of each edit is unknown beyond the run.json section times')

    D = pd.read_csv(H.reg(H.SR / 's4_real04' / '_parts' / 'A1_deletion_sensitivity.csv'), low_memory=False).set_index('eval_key')
    reg = json.loads(H.reg(H.R2 / 'layouts.json').read_text(encoding='utf-8'))
    M = pd.read_csv(H.reg(H.R2 / 'manifest.csv'), low_memory=False, keep_default_na=False)
    m = M[(M.baseline == 'L6.3') & (M.surface == 'M10_curv') & (M.phase == 'H0') & (M.sigma.astype(float) == 0.0)].drop_duplicates('layout').set_index('layout').eval_key
    s0 = D.loc[m['regular']]; v = [D.loc[m[x['layout']]] for x in reg['baselines']['L6.3']['views'] if x['strategy'] == 'S2']
    med = [float(np.median([100 * (y[c] - s0[c]) for y in v])) for c in ('glob_Ctau_official', 'glob_Ctau_without_clipped')]
    P = pd.read_csv(H.reg(H.SR / 's345_ready' / 'plan_s345.csv'), low_memory=False); P = P[P.set_formal == 'S5-main']
    tot = {}
    for x in P.itertuples():
        mm = json.loads(H.reg(H.SR / 's345_ready' / 'run' / 'eval' / f'{x.decode_key}.json').read_text(encoding='utf-8'))['metrics']
        t_ = tot.setdefault(x.family, [0, 0]); t_[0] += int(mm['glob_n_gross']); t_[1] += int(mm['loc_n_gross'])
    out['a3_recheck'] = dict(S2_median_pp=med, S2_ok=bool(abs(med[0] + 14.249984) < 5e-7 and abs(med[1] + 14.529142) < 5e-7), S5_gross=tot,
                             S5_ok=bool(tot.get('line') == [5257, 129] and tot.get('point') == [469, 22]))
    out['t_local'] = K.now_local()
    H.run_put('A3', out); H.flush_reads('A3')
    return out
