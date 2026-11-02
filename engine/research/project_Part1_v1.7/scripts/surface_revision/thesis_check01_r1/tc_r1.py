# Recompute supplementary diagnostics from stored study outputs.

"""Recompute amplitude summaries and numerical source checks from stored results."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = ROOT / 'outputs' / 'surface_revision'
OUT = SR / 'thesis_check01_r1'
PARENT = SR / 'thesis_check01'
RUN = OUT / 'run.json'
DRAFTS = ROOT.parents[1] / 'protocols'
SURF = ('B_curv', 'M10_curv', 'M24_curv')
SHORT = {'B_curv': 'B', 'M10_curv': 'M10', 'M24_curv': 'M24'}
READ: dict = {}


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(Path(p).resolve())


def reg(p) -> Path:

    p = Path(p)
    if not p.exists():
        raise SystemExit(f'STOP missing source {rel(p)}')
    r = rel(p)
    if r not in READ:
        READ[r] = sha(p)
    return p


def guard(p) -> Path:
    q = Path(p).resolve()
    if OUT.resolve() not in q.parents and q != OUT.resolve():
        raise SystemExit(f'STOP write outside {OUT}: {q}')
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_load() -> dict:
    return json.loads(RUN.read_text(encoding='utf-8')) if RUN.exists() else dict(task='THESIS-CHECK-01R',  acceptance='reference (independent)', created=now())


def run_put(section: str, obj) -> None:
    r = run_load(); r[section] = clean(obj); r['updated'] = now()
    q = guard(RUN); t = q.with_name(q.name + '.tmp'); t.write_text(json.dumps(clean(r), ensure_ascii=False, indent=1), encoding='utf-8'); os.replace(t, q)


def flush(section: str) -> None:
    r = run_load(); s = r.setdefault('sources_read', {})
    for k, v in READ.items():
        if k in s and s[k] != v:
            raise SystemExit(f'STOP {k} changed during this round ({section})')
        s.setdefault(k, v)
    r['updated'] = now(); q = guard(RUN); t = q.with_name(q.name + '.tmp'); t.write_text(json.dumps(clean(r), ensure_ascii=False, indent=1), encoding='utf-8'); os.replace(t, q)


SOURCES = ['outputs/surface_revision/thesis_check01/report_zh.md', 'outputs/surface_revision/thesis_check01/run.json', 'outputs/surface_revision/thesis_check01/results.csv',
           'outputs/surface_revision/thesis_check01/_parts/B_results.csv', 'outputs/surface_revision/thesis_check01/_parts/C_gauss.csv',
           'outputs/surface_revision/thesis_check01/_parts/C_window.csv', 'outputs/surface_revision/thesis_check01/_parts/A_errata.csv',
           'outputs/surface_revision/thesis_check01/_parts/C_identity.csv', 'outputs/surface_revision/thesis_check01/_parts/C_regimes.csv',
           'outputs/surface_revision/thesis_check01/_parts/C_resource.csv', 'outputs/surface_revision/s12_resource01/results/views.csv',
           'outputs/surface_revision/s4_real07/results.csv', 'outputs/surface_revision/s4_real07/_parts/views.csv', 'outputs/surface_revision/s3_code05/README_study.md',
           'outputs/surface_revision/s3_code04/results/C1_verdicts.csv', 'scripts/surface_revision/thesis_check01/tc_cross.py']
DOCS = ['THESIS-CHECK-01_review.md', 'THESIS-CHECK-01R_protocol.md']


def parent_identity() -> dict:
    r = json.loads(reg(PARENT / 'run.json').read_text(encoding='utf-8'))
    fin = r['final']
    out = dict(parent=rel(PARENT), parent_final_t=fin['t_local'], scope=r['scope'], unique_references=r['B_link']['total_unique_references'],
               source_inputs=r['B_link']['total_source_inputs'])
    out['parent_outputs_unchanged'] = {k: sha(ROOT / k) == v for k, v in fin['outputs'].items()}
    out['parent_code_unchanged'] = {k: sha(ROOT / k) == v for k, v in fin['code'].items()}
    out['all_parent_outputs_unchanged'] = bool(all(out['parent_outputs_unchanged'].values())); out['all_parent_code_unchanged'] = bool(all(out['parent_code_unchanged'].values()))
    for k in ('parent_outputs_unchanged', 'parent_code_unchanged'):
        out[f'n_{k}'] = len(out.pop(k))
    return out


def start() -> dict:
    for p in SOURCES:
        reg(ROOT / p)
    for d in DOCS:
        reg(DRAFTS / d)
    pid = parent_identity()
    if not (pid['all_parent_outputs_unchanged'] and pid['all_parent_code_unchanged']):
        raise SystemExit('STOP parent THESIS-CHECK-01 files changed after its final record')
    run_put('start', dict(t_local=now(), parent_identity=pid, revision_scope=['R1 report corrections (CODE-05 errata, section 4 summary, causal wording)', 'R2 Gaussian amplitude summary with denominators',
                                                                             'R3 post-hoc source check and registration', 'R4 redraw of F1-F4'],
                          imaging=0, decoding=0, new_references=0, note='parent files read only; nothing copied; references by path and hash'))
    flush('start')
    return pid


def amp() -> dict:
    V = pd.read_csv(reg(SR / 's12_resource01' / 'results' / 'views.csv'), low_memory=False, float_precision='round_trip')
    v = V[(V.surface_class == 'supplement') & (V.regime == 'resource_cap_11695')].copy()
    if set(v.surface) != {'M10_curv_gauss'}:
        raise SystemExit(f'STOP unexpected supplement surfaces {set(v.surface)}')
    old = pd.read_csv(reg(PARENT / '_parts' / 'C_gauss.csv'), low_memory=False, float_precision='round_trip')
    rows = []
    for (sec, fam, lay, sz, sg), g in v.groupby(['section', 'family', 'layout', 'size_mm', 'sigma']):
        a = pd.to_numeric(g.amp_err_peak_mm, errors='coerce'); fin = np.isfinite(a); no = g.state == 'no_output'
        rec = dict(section=sec, family=fam, layout=lay, size_mm=sz, size_meaning=('dot radius (diameter = 2 x size_mm)' if fam == 'point' else 'line width'),
                   diameter_mm=(2 * sz if fam == 'point' else None), sigma=sg, n_total=len(g), n_no_output=int(no.sum()), n_output=int((~no).sum()), n_amp_defined=int(fin.sum()),
                   n_amp_undefined=int((~fin).sum()), n_output_but_amp_undefined=int(((~no) & ~fin).sum()), n_no_output_but_amp_defined=int((no & fin).sum()),
                   amp_err_peak_median_mm=(float(a[fin].median()) if fin.any() else None), amp_err_peak_min_mm=(float(a[fin].min()) if fin.any() else None),
                   amp_err_peak_max_mm=(float(a[fin].max()) if fin.any() else None), glob_Ctau_mean_no_output_as_0=float(g.glob_Ctau.fillna(0).mean()),
                   loc_Ctau_mean_no_output_as_0=float(g.loc_Ctau.fillna(0).mean()))
        o = old[(old.section == sec) & (old.family == fam) & (old.layout == lay) & (old.size_mm == sz) & (old.sigma == sg)]
        rec['parent_median_mm'] = (float(o.amp_err_peak_median_mm.iloc[0]) if len(o) and pd.notna(o.amp_err_peak_median_mm.iloc[0]) else None)
        m0, m1 = rec['parent_median_mm'], rec['amp_err_peak_median_mm']
        rec['median_equals_parent'] = bool((m0 is None and m1 is None) or (m0 is not None and m1 is not None and abs(m0 - m1) <= 1e-12))
        rows.append(rec)
    D = pd.DataFrame(rows)
    chk = dict(n_groups=len(D), n_groups_parent=len(old), denominators_closed=bool((D.n_amp_defined + D.n_amp_undefined == D.n_total).all()),
               output_split_closed=bool((D.n_output + D.n_no_output == D.n_total).all()), no_output_but_amp_defined=int(D.n_no_output_but_amp_defined.sum()),
               medians_equal_parent=bool(D.median_equals_parent.all()), groups_all_undefined=int((D.n_amp_defined == 0).sum()))
    p = guard(OUT / 'amplitude_summary.csv'); D.to_csv(p, index=False, encoding='utf-8-sig')

    rng = []
    for (sec, fam, sg), g in D.groupby(['section', 'family', 'sigma']):
        d = g[g.n_amp_defined > 0]
        rng.append(dict(section=sec, family=fam, sigma=sg, n_configs=len(g), n_configs_defined=len(d), cfg_median_min=(float(d.amp_err_peak_median_mm.min()) if len(d) else None),
                        cfg_median_max=(float(d.amp_err_peak_median_mm.max()) if len(d) else None), case_min=(float(d.amp_err_peak_min_mm.min()) if len(d) else None),
                        case_max=(float(d.amp_err_peak_max_mm.max()) if len(d) else None), n_total=int(g.n_total.sum()), n_no_output=int(g.n_no_output.sum()),
                        n_amp_defined=int(g.n_amp_defined.sum())))
    run_put('R2_amplitude', dict(checks=chk, ranges=rng, file=rel(p), sha256=sha(p), estimator='pandas median of the finite amp_err_peak_mm values of the group (as parent C_gauss.csv)',
                                 t_local=now()))
    flush('amp')
    if not (chk['denominators_closed'] and chk['output_split_closed'] and chk['medians_equal_parent']):
        raise SystemExit(f'STOP amplitude summary checks failed: {chk}')
    return chk


def src() -> dict:
    pr = json.loads(reg(PARENT / 'run.json').read_text(encoding='utf-8')); old_read = set(pr['sources_read'])
    V = pd.read_csv(reg(SR / 's12_resource01' / 'results' / 'views.csv'), low_memory=False, float_precision='round_trip')
    v = V[(V.surface_class == 'main') & (V.regime == 'resource_cap_11695')].drop_duplicates('input_key')
    v = v[v.id_unconf > 0]
    rows = []
    for r in v.itertuples():
        jp = ROOT / r.eval_record; zp = ROOT / r.eval_record.replace('.json', '.npz')
        in_old = rel(zp) in old_read
        rec_sha = json.loads(reg(jp).read_text(encoding='utf-8'))['npz_sha256']; cur = sha(reg(zp))
        rows.append(dict(npz=rel(zp), in_parent_sources_read=in_old, record_sha_equal=bool(cur == rec_sha), parent_registered_sha_equal=(bool(pr['sources_read'][rel(zp)] == cur) if in_old else None)))
    D = pd.DataFrame(rows)

    dep = {}
    man = reg(SR / 'surf02' / 'release' / 'dependencies_surf02_v1_1.json'); rel_json = json.loads(reg(SR / 'surf02' / 'release' / 'release_surf02.json').read_text(encoding='utf-8'))
    m = json.loads(man.read_text(encoding='utf-8'))
    dep['surf02_manifest'] = dict(file=rel(man), manifest_sha_equals_release=bool(sha(man) == rel_json['dependency_manifest_sha256']), n_files=len(m['files']),
                                  all_equal=bool(all(sha(reg(ROOT / k)) == h for k, h in m['files'].items())),
                                  covers=[k for k in m['files'] if any(s in k for s in ('s1_model', 'Dataset.csv', 'pm014', 'surface_eval_r3', 's2_common', 'sn_lib', 'sl_lib'))])
    r6 = json.loads(reg(SR / 's4_real06' / 'run.json').read_text(encoding='utf-8'))['final']['final_code']
    k6 = 'scripts/surface_revision/s4_real06/s4r6_analysis.py'
    dep['s4r6_analysis'] = dict(file=k6, frozen_list='outputs/surface_revision/s4_real06/run.json final.final_code', equal=bool(sha(reg(ROOT / k6)) == r6[k6]))
    dep['parent_code'] = dict(note='parent tc_*.py hashes recorded in thesis_check01/run.json final.code; checked unchanged at start of this round')
    out = dict(n_inputs_with_unconfirmable=len(D), n_missing_from_parent_sources_read=int((~D.in_parent_sources_read).sum()),
               n_registered_in_parent=int(D.in_parent_sources_read.sum()), all_record_sha_equal=bool(D.record_sha_equal.all()),
               parent_registered_still_equal=bool(D[D.in_parent_sources_read].parent_registered_sha_equal.all()), dependencies=dep,
               statement='The registered set is unchanged. Additional sets were checked post hoc against their records; this does not establish prospective registration.',
               t_local=now())
    run_put('R3_sources', dict(out, missing_files=D[~D.in_parent_sources_read].npz.tolist()))
    flush('src')
    return {k: v for k, v in out.items() if k != 'dependencies'} | dict(deps_ok=bool(dep['surf02_manifest']['all_equal'] and dep['s4r6_analysis']['equal']))


def extra() -> dict:
    V = pd.read_csv(reg(SR / 's12_resource01' / 'results' / 'views.csv'), low_memory=False, float_precision='round_trip')
    v = V[(V.surface_class == 'main') & (V.regime == 'resource_cap_11695') & (V.sigma == 0)]
    out = {}

    B = pd.read_csv(reg(PARENT / '_parts' / 'B_results.csv'), low_memory=False, float_precision='round_trip')
    S = B[(B.record_type == 'summary') & (B.batch == 'S12') & (B.section == 'S1') & (B.domain == 'glob') & (B.sigma == 0)]
    out['S1_sigma0_3surf'] = {f'{f}|{s:g}': dict(actual=float(g.act_C_tau_mean.mean()), reference=float(g.ref_C_tau_mean.mean())) for (f, s), g in S.groupby(['family', 'size_mm'])}

    amb = v[(v.section == 'S2') & (v.family == 'point')].groupby(['layout', 'surface']).n_ambiguous.mean().unstack()
    out['S2_dots_n_ambiguous'] = {k: {SHORT[s]: float(x) for s, x in r.items()} for k, r in amb.loc[['R6x4', 'R7.35x4', 'R8x4', 'T8x4']].iterrows()}
    lu = v[(v.section == 'S2') & (v.family == 'line')].groupby(['layout', 'surface'])[['n_status_segment_unknown', 'n_detected', 'n_accepted']].mean()
    out['S2_lines_states'] = {f'{a}|{SHORT[b]}': {k: float(x) for k, x in r.items()} for (a, b), r in lu.loc[['s4.45_phi0', 's5_phi0', 's5.6_phi0', 's6.3_phi0', 's8_phi0', 's4.45_phi45', 's5_phi45']].iterrows()}

    R7 = pd.read_csv(reg(SR / 's4_real07' / 'results.csv'), low_memory=False, float_precision='round_trip')
    c = R7[(R7.record_type == 'cond') & (R7.domain == 'glob') & (R7.family == 'line') & (R7.set == 'anchor') & (R7.method == 'CLIP')]
    out['R7_anchor_CLIP_gross'] = {f'{SHORT[s]}|{sg:g}': int(g.n_gross.sum()) for (s, sg), g in c.groupby(['surface', 'sigma'])}
    sm = B[(B.record_type == 'summary') & (B.batch == 'R7') & (B.set == 'main') & (B.domain == 'glob') & (B.family == 'line') & (B.method == 'CLIP') & (B.sigma == 40.0)]
    out['R7_lines_sigma40'] = {f'{a}|{SHORT[s]}': dict(median=100 * r.d_act_median, min=100 * r.d_act_min, max=100 * r.d_act_max, n_pos=int(r.d_act_n_pos), n_neg=int(r.d_act_n_neg), n_blocks=int(r.n_blocks))
                               for (a, s), r in sm.set_index(['arm', 'surface']).iterrows()}
    run_put('extra_summaries', dict(out, note='read from existing tables only (S12 views, S4-REAL-07 results, parent summaries)', t_local=now()))
    flush('extra')
    return out


def final() -> dict:
    r = run_load(); bad = [k for k, h in r.get('sources_read', {}).items() if sha(ROOT / k if not k.startswith('C:') else k) != h]
    pid = parent_identity()
    outs = {rel(p): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'run.json' and '.tmp' not in p.name}
    code = {rel(p): sha(p) for p in sorted(HERE.glob('*.py'))}
    res = dict(sources_unchanged=not bad, changed=bad[:10], n_sources=len(r.get('sources_read', {})), parent_outputs_unchanged=pid['all_parent_outputs_unchanged'],
               parent_code_unchanged=pid['all_parent_code_unchanged'], parent_unique_references=pid['unique_references'], parent_source_inputs=pid['source_inputs'],
               imaging=0, decoding=0, new_references=0, outputs=outs, code=code, t_local=now())
    run_put('final', res)
    return {k: v for k, v in res.items() if k not in ('outputs', 'code')}


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    fn = dict(start=start, amp=amp, src=src, extra=extra, final=final).get(cmd)
    if fn is None:
        raise SystemExit(__doc__)
    print(json.dumps(clean(fn()), ensure_ascii=False)[:5000])
