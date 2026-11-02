# Generate constrained layouts in the reserved proposal blocks.

"""S4-REAL-07 B2: layout generation on the reserved blocks B024-B029 at j = 0.30 (no imaging).

Registered generator s3_generate.evaluate (seed_s3 namespace, unit vector U[-1, 1] -> minus mean -> scaled, displacement j * p0 * u, descriptors at
the reference phase, inherited general validity). One proposal stream per family x block, candidates 0..999 in order, shared by the four arms:
  S1   candidate 0; not generally valid -> generation failure (no next candidate)
  S2*  first generally valid candidate with g_min >= 0.80 p0 (dots: g_min = registered movable-column neighbour gap, not the all-pair distance)
  S3   first generally valid candidate with h <= 1.15 h0
  S4*  first generally valid candidate meeting both
The stream is walked until every arm is resolved or 1000 candidates were evaluated (an arm without a candidate = generation failure, denominator 1000).
Continuous-coordinate policy; raster deviations are diagnosed later and never used to redraw.
"""
from __future__ import annotations

import time
from multiprocessing import Pool

import s4r7_common as H

K = H.K
BLOCKS = (24, 25, 26, 27, 28, 29)
J = 0.30
ALPHA, BETA, N_CAND = 0.80, 1.15, 1000
ARMS = ('S1', 'S2*', 'S3', 'S4*')


def _block(args) -> dict:
    name, block = args
    import s3_generate as SG
    import sp2_generator as GEN
    bl = SG.baseline(name); ms = SG.sets()[name]; p0, h0 = float(bl['p0_mm']), float(ms['h0_mm'])
    ctx = GEN.context_v2(bl, ms, J * p0)
    sel = {a: None for a in ARMS}; rows = []; t0 = time.time(); evaluated = 0
    for c in range(N_CAND):
        ev = SG.evaluate(name, block, c, J, ctx); evaluated += 1
        ok = bool(ev['general_ok']); s2 = bool(ev['g_min_mm'] >= ALPHA * p0 - 1e-12); s3 = bool(ev['h_mm'] <= BETA * h0 + 1e-12)
        rows.append(dict(cand=c, seed=int(ev['seed']), general_ok=ok, general_fail=';'.join(ev['general_fail']), g_min_mm=float(ev['g_min_mm']), h_mm=float(ev['h_mm']),
                         rms_mm=float(ev['delta_rms_mm']), s2=s2, s3=s3))
        if c == 0:
            sel['S1'] = dict(cand=0, ev=ev) if ok else 'failed'
        for a, cond in (('S2*', ok and s2), ('S3', ok and s3), ('S4*', ok and s2 and s3)):
            if sel[a] is None and cond:
                sel[a] = dict(cand=c, ev=ev)
        if all(sel[a] is not None for a in ARMS):
            break
    out = dict(baseline=name, block=block, p0_mm=p0, h0_mm=h0, n_evaluated=evaluated, stopped='all arms resolved' if all(sel[a] is not None for a in ARMS) else 'candidate budget exhausted (1000)',
               pool=dict(general_ok=sum(r['general_ok'] for r in rows), s2=sum(r['general_ok'] and r['s2'] for r in rows), s3=sum(r['general_ok'] and r['s3'] for r in rows),
                         s4=sum(r['general_ok'] and r['s2'] and r['s3'] for r in rows)), seconds=round(time.time() - t0, 1), arms={}, candidates=rows)
    for a in ARMS:
        s = sel[a]
        if s is None or s == 'failed':
            out['arms'][a] = dict(status='generation_failure', reason=('candidate 0 not generally valid' if a == 'S1' else 'no qualifying candidate in 1000'), denominator=(1 if a == 'S1' else N_CAND))
            continue
        ev = s['ev']
        out['arms'][a] = dict(status='selected', cand=s['cand'], seed=int(ev['seed']), unit_hash=ev['unit_hash'], disp_px={str(k): float(v) for k, v in ev['disp_px'].items()},
                              g_min_mm=float(ev['g_min_mm']), h_mm=float(ev['h_mm']), rms_mm=float(ev['delta_rms_mm']), delta_max_mm=float(ev['delta_max_mm']),
                              edge_clearance_min_mm=float(ev['edge_clearance_min_mm']), raster_ref_phase=ev['raster_sha256'], general_ok=bool(ev['general_ok']))
    return out


def generate(workers: int = 12) -> dict:
    t0 = time.time(); jobs = [(n, b) for n in ('R8x4', 'L6.3') for b in BLOCKS]
    with Pool(min(workers, len(jobs))) as pool:
        res = pool.map(_block, jobs)
    gen = dict(rule=__doc__.strip(), j=J, alpha=ALPHA, beta=BETA, blocks=list(BLOCKS), n_cand=N_CAND, blocks_out=res, code=H.code_hashes(H.GEN_DEPS), t_local=K.now_local(),
               seconds=round(time.time() - t0, 1), proposals_evaluated=int(sum(r['n_evaluated'] for r in res)))
    H.save_json(H.OUT / 'generation.json', gen)
    summ = [dict(baseline=r['baseline'], block=r['block'], n_evaluated=r['n_evaluated'], **{a: (r['arms'][a].get('cand') if r['arms'][a]['status'] == 'selected' else 'FAIL') for a in ARMS})
            for r in res]
    H.run_put('generation', dict(summary=summ, proposals_evaluated=gen['proposals_evaluated'], seconds=gen['seconds'], file='outputs/surface_revision/s4_real07/generation.json',
                                 sha256=K.sha_file(H.OUT / 'generation.json'), t_local=K.now_local()))
    return dict(summary=summ, proposals=gen['proposals_evaluated'], seconds=gen['seconds'])
