# Select geometric layouts from bounded proposal blocks and record rejected proposals.

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s3_common as C
import numpy as np
import sp2_common as S2
import sp2_generator as GEN
import sp_geometry as G

KEEP = ('g_min_mm', 'g_max_mm', 'g_p05_mm', 'g_p50_mm', 'h_mm', 'delta_rms_mm', 'delta_max_mm', 'delta_mean_mm', 'edge_clearance_min_mm', 'dmin_allpairs_mm', 'lit_px', 'peak_equal_integral',
        'n_elements', 'n_components', 'raster_sha256', 'equal_integral_feasible', 'boundary_in', 'boundary_out', 'columns_same_as_regular', 'components_equal_dots', 'ids_present_same_as_regular', 'n_clipped_lines',
        'raster_centre_residual_max_px', 'xor_px_vs_regular')
_SETS = None


def sets() -> dict:
    global _SETS
    if _SETS is None:
        _SETS = C.load_json(S2.OUT / 'movable_sets_v2.json')['sets']
    return _SETS


def baseline(name: str) -> dict:
    return next(b for b in S2.BASELINES if b['name'] == name)


def general_ok(fam: str, d: dict) -> tuple[bool, list[str]]:

    bad = []
    if not d.get('equal_integral_feasible'):
        bad.append('equal_integral_infeasible')
    if fam == 'dots':
        if d.get('boundary_in', 0) or d.get('boundary_out', 0):
            bad.append('boundary_in_out')
        if not d.get('columns_same_as_regular', True):
            bad.append('columns_changed')
        if not d.get('components_equal_dots', True):
            bad.append('components_changed')
    else:
        if not d.get('ids_present_same_as_regular', True):
            bad.append('line_ids_changed')
        if d.get('n_clipped_lines', 0):
            bad.append('clipped_lines')
    return (not bad), bad


def evaluate(name: str, block: int, cand: int, j: float, ctx=None) -> dict:
    bl = baseline(name); ms = sets()[name]; fam, p0 = bl['family'], bl['p0_mm']
    ctx = ctx or GEN.context_v2(bl, ms, j * p0)
    seed = C.seed_s3(bl['index'], block, cand); rng = np.random.default_rng(seed)
    u, uinfo = GEN.unit_vector(len(ms['movable']), rng)
    a_px = S2.px(j * p0); disp = {int(k): float(v) for k, v in zip(ms['movable'], u * a_px)}
    built = GEN.build_layout(bl, disp)
    d = G.line_descriptors(built['meta'], built['pat'], ctx, disp) if fam == 'lines' else G.dot_descriptors(built['lay'], built['pat'], ctx, disp, 'u')
    m = GEN.masks_v2(d['g_min_mm'], d['h_mm'], p0, ms['h0_mm'])
    ok, why = general_ok(fam, d)
    return dict(baseline=name, family=fam, block=block, block_id=C.block_id(block), cand=cand, j=j, seed=seed, unit_hash=S2.sha_array(u)[:16], unit_info=uinfo, general_ok=ok, general_fail=why,
                accept={s: bool(m[s] and ok) for s in C.STRATEGIES}, mask_raw=m, h0_mm=ms['h0_mm'], **{k: d.get(k) for k in KEEP}, disp_px={str(k): v for k, v in disp.items()})


def eval_block(args) -> list[dict]:
    name, block, n = args
    bl = baseline(name); ctx = GEN.context_v2(bl, sets()[name], C.J_STRONG * bl['p0_mm'])
    return [evaluate(name, block, c, C.J_STRONG, ctx) for c in range(n)]


def select(name: str, recs: list[dict]) -> tuple[list[dict], list[dict], dict]:

    fam = baseline(name)['family']; layouts = {}; fails = []; table = {}
    for b in C.BLOCKS:
        rb = sorted([r for r in recs if r['block'] == b], key=lambda r: r['cand'])
        if not rb:
            continue
        valid = [r for r in rb if r['general_ok']]
        sel = {}
        arms = [('S1_free', C.J_STRONG)] + ([('S1_free', C.J_WEAK)] + [(s, C.J_STRONG) for s in C.STRATEGIES[1:]] if name in C.CORE else [])
        for s, j in arms:
            pool = valid if s == 'S1_free' else [r for r in valid if r['accept'][s]]
            if not pool:
                fails.append(dict(baseline=name, family=fam, block=b, block_id=C.block_id(b), strategy=s, j=j, n_candidates=len(rb), n_general_valid=len(valid), status='generation_failed')); sel[(s, j)] = None; continue
            r = pool[0]
            if j == C.J_WEAK:
                r = evaluate(name, b, r['cand'], C.J_WEAK)
            pid = f"{name}_s3_j{j:.2f}_{C.block_id(b)}_c{r['cand']}"
            if pid not in layouts:
                layouts[pid] = dict(proposal_id=pid, j=j, block=b, block_id=C.block_id(b), proposal=r['cand'], seed=r['seed'], unit_hash=r['unit_hash'], disp_px=r['disp_px'], arms=[], arm_blocks=[],
                                    geometry={k: r.get(k) for k in KEEP}, masks_at_selection=r['accept'])
            layouts[pid]['arms'].append(f'{s}@j{j:.2f}'); layouts[pid]['arm_blocks'].append((f'{s}@j{j:.2f}', b)); sel[(s, j)] = r['cand']
        table[C.block_id(b)] = {f'{s}@j{j:.2f}': v for (s, j), v in sel.items()}
    return list(layouts.values()), fails, table


def summarise(name: str, recs: list[dict]) -> dict:
    out = {}
    for s in C.STRATEGIES:
        per = {}
        for b in C.BLOCKS:
            rb = sorted([r for r in recs if r['block'] == b], key=lambda r: r['cand'])
            acc = [r['cand'] for r in rb if r['accept'][s]]
            per[C.block_id(b)] = dict(n_candidates=len(rb), n_general_valid=sum(1 for r in rb if r['general_ok']), n_accepted=len(acc), first=(acc[0] if acc else None))
        firsts = [v['first'] for v in per.values() if v['first'] is not None]
        nc = sum(v['n_candidates'] for v in per.values()); na = sum(v['n_accepted'] for v in per.values())
        out[s] = dict(n_candidates=nc, n_accepted=na, acceptance_rate=(na / nc if nc else None), blocks_success={str(k): sum(1 for v in per.values() if v['first'] is not None and v['first'] < k) for k in C.PREFIX_BUDGETS},
                      first_index_median=(float(np.median(firsts)) if firsts else None), first_index_max=(max(firsts) if firsts else None), per_block=per)
    return out


# Select layouts from proposal blocks and record acceptance outcomes.
def main():
    if C.GEN_LEDGER.exists() or C.LAYOUTS_JSON.exists():
        raise SystemExit('generation outputs exist (no overwrite)')
    C.OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()

    formal = [C.seed_s3(b['index'], blk, c) for b in S2.BASELINES for blk in C.BLOCKS for c in range(C.N_CAND)]
    old = {S2.seed_v2(b['index'], blk, c) for b in S2.BASELINES for blk in range(S2.N_BLOCKS) for c in range(S2.N_PER_BLOCK)}
    old |= {S2.S1.SEED_BASE + 100000 * b['index'] + 10000 * ji + 100 * blk + c for b in S2.BASELINES for ji in range(len(S2.S1.J_LEVELS)) for blk in range(S2.S1.N_BLOCKS) for c in range(S2.S1.N_PER_BLOCK)}
    noise_old = {S2.NOISE_SEED_BASE_V2 + off + i for off in (0, 1000) for i in range(100)}
    seed_check = dict(n_formal=len(formal), n_unique=len(set(formal)), injective=(len(set(formal)) == len(formal)), overlap_with_v1_v2=len(set(formal) & old), overlap_with_pilot_noise=len(set(formal) & noise_old),
                      min=min(formal), max=max(formal), rule='3 000 000 000 + 100 000 * baseline_index + 1000 * block + candidate')
    if not (seed_check['injective'] and seed_check['overlap_with_v1_v2'] == 0 and seed_check['overlap_with_pilot_noise'] == 0):
        raise SystemExit(f'seed check failed {seed_check}')
    jobs = [(name, b, C.N_CAND) for name in C.CORE for b in C.BLOCKS]
    with Pool(processes=16) as pool:
        res = pool.map(eval_block, jobs, chunksize=1)
    recs = {name: [] for name in C.CORE + C.COST}
    for (name, b, _), rr in zip(jobs, res):
        recs[name].extend(rr)
    for name in C.COST:
        bl = baseline(name); ctx = GEN.context_v2(bl, sets()[name], C.J_STRONG * bl['p0_mm'])
        for b in C.BLOCKS:
            c = 0
            while True:
                r = evaluate(name, b, c, C.J_STRONG, ctx); recs[name].append(r)
                if r['general_ok'] or c >= C.N_CAND - 1:
                    break
                c += 1
    with open(C.GEN_LEDGER, 'w', encoding='utf-8') as fh:
        for name in C.CORE + C.COST:
            for r in sorted(recs[name], key=lambda r: (r['block'], r['cand'])):
                fh.write(json.dumps(C.clean({k: v for k, v in r.items() if k != 'disp_px'}), ensure_ascii=False) + '\n')
    layouts = {}; failures = []; tables = {}; summaries = {}
    sets_ = sets()
    for name in C.CORE + C.COST:
        bl = baseline(name)
        lay, fails, table = select(name, recs[name])
        reg = dict(proposal_id=f'{name}_s3_j0.00_regular', j=0.0, block=-1, block_id='S0', proposal=-1, seed=None, unit_hash=None, disp_px={}, arms=['S0_regular'], arm_blocks=[('S0_regular', -1)])
        layouts[name] = dict(family=bl['family'], p0_mm=bl['p0_mm'], movable=sets_[name]['movable'], protected_L=sets_[name]['protected_L'], protected_R=sets_[name]['protected_R'], h0_mm=sets_[name]['h0_mm'],
                             n_unique_layouts=len(lay) + 1, generation_failures=fails, layouts=[reg] + lay)
        failures += fails; tables[name] = table
        if name in C.CORE:
            summaries[name] = summarise(name, recs[name])
        else:
            summaries[name] = dict(S1_free=dict(per_block={C.block_id(b): dict(first=next((r['cand'] for r in sorted(recs[name], key=lambda r: r['cand']) if r['block'] == b and r['general_ok']), None),
                                                                              n_candidates_evaluated=sum(1 for r in recs[name] if r['block'] == b)) for b in C.BLOCKS}))
    C.save_json(C.LAYOUTS_JSON, dict(generated_local=C.now_local(), package=C.PACKAGE_VERSION, rule='formal blocks B000-B023; S1 first generally valid candidate (j 0.10 and 0.20 share the unit vector); S2 / S3 / S4 first generally valid candidate meeting the j 0.20 mask; identical layouts merged with their strategy membership; disp_px full precision',
                                     seed_check=seed_check, baselines=layouts))
    C.save_json(C.GEN_SUMMARY, dict(generated_local=C.now_local(), seconds=round(time.time() - t0, 1), seed_check=seed_check, n_ledger_rows=sum(len(v) for v in recs.values()), summaries=summaries, selection=tables,
                                    generation_failures=failures, general_invalid={name: sum(1 for r in recs[name] if not r['general_ok']) for name in recs}))
    print(json.dumps(dict(seconds=round(time.time() - t0, 1), rows={k: len(v) for k, v in recs.items()}, n_layouts={k: v['n_unique_layouts'] for k, v in layouts.items()}, n_failures=len(failures),
                          accept={name: {s: summaries[name][s]['acceptance_rate'] for s in C.STRATEGIES} for name in C.CORE},
                          blocks_success={name: {s: summaries[name][s]['blocks_success'] for s in C.STRATEGIES} for name in C.CORE})))


if __name__ == '__main__':
    main()
