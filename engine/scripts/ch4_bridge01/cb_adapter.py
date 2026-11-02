# Build the Chapter 4 candidate rasters and bind each case to its forward-model inputs.

from __future__ import annotations
import json, copy
import numpy as np
import pandas as pd
import cb_common as G

C, B = G.C, G.B
_S: dict = {}
NEW_SOURCES = {
    'D_r0.75_p8x4': dict(filter=dict(section='S5', family='point', regime='equal_integral', radius_mm=0.75, p_parallel_mm=8.0, p_perp_mm=4.0, arrangement='R'), change={}),
    'D_r0.75_p8x6': dict(filter=dict(section='S5', family='point', regime='equal_integral', radius_mm=0.75, p_parallel_mm=8.0, p_perp_mm=4.0, arrangement='R'),
                         change=dict(p_perp_mm=6.0, layout_id='R8x6', brightness_rule='peak=1 (CH4-BRIDGE-01 resource rule A = min(1, Q / S0) = 1 since S0 < Q)')),
    'D_r1_T8x4': dict(filter=dict(section='S2', family='point', regime='equal_integral', radius_mm=1.0, p_parallel_mm=8.0, p_perp_mm=4.0, arrangement='T'), change={}),
    'L_w1_s10': dict(filter=dict(section='S2', family='line', regime='equal_integral', line_width_mm=1.0, line_pitch_mm=10.0, orientation_deg=0.0), change={}),
    'L_w0.5_s10': dict(filter=dict(section='S2', family='line', regime='equal_integral', line_width_mm=1.0, line_pitch_mm=10.0, orientation_deg=0.0),
                       change=dict(line_width_mm=0.5)),
}
NUM = ('radius_mm', 'p_parallel_mm', 'p_perp_mm', 'line_width_mm', 'line_pitch_mm', 'orientation_deg')


def platform():

    if 'p' not in _S:
        ch = B.init(); p = ch.platform()
        import c_common as K
        import r_manifest as RM
        import adp_patterns as PT
        import rdy_inputs as RDY

        RM.ENVS.update(UNIFORM=(1., ('const', .75)), GRAD_X=(1., ('gradient_x', .75, .25)), GRAD_Y=(1., ('gradient_y', .75, .25)))
        T = {cfg: dict(source=v['source'], row=v['row']) for cfg, v in p['T'].items() if cfg in G.OLD_DOTS + G.OLD_LINES}
        G.require(sorted(T) == sorted(G.OLD_DOTS + G.OLD_LINES), 'Old template set incomplete')
        T.update(new_templates(p['RI']))
        _S.update(p=p, ch=ch, K=K, RM=RM, PT=PT, RDY=RDY, T=T, M5=pd.read_csv(K.SRC5 / 'formal_manifest.csv', float_precision='round_trip', low_memory=False))
    return _S


def new_templates(RI) -> dict:
    Cd = RI.cands(); out = {}
    for cfg, spec in NEW_SOURCES.items():
        q = Cd
        for k, v in spec['filter'].items():
            q = q[(q[f'_{k}'] == v)] if k in NUM else q[q[k].astype(str) == str(v)]
        G.require(len(q) > 0, f'No registered source row for {cfg}')
        base = RI._row(q)
        row = copy.deepcopy(base)
        for k, v in spec['change'].items():
            row[k] = v
        if spec['change']:
            row['cand_id'] = f'CB01:{cfg}:from:{base["cand_id"]}'
            row['raster_sha256_registered'] = ''
            for k in ('peak_registered', 'forward_key_pre', 'decode_key_pre'):
                row[k] = ''
        out[cfg] = dict(source=f'{base["section"]} {base["regime"]} row {base["cand_id"]}' + (f'; explicit change {spec["change"]}' if spec['change'] else '; unchanged'), row=row,
                        base_cand_id=base['cand_id'], change=spec['change'])
    return out


# Apply the shared integral cap and peak ceiling to the phase-specific candidate raster.
def inputs(tpl: dict, fam: str, ph: dict) -> dict:

    S = platform(); RI = S['p']['RI']
    if not str(tpl['brightness_rule']).startswith('peak=1'):
        inp = RI.inputs(tpl, fam, ph)
        G.require(float(inp['lit_px']) > G.Q and abs(float(inp['peak']) - G.Q / float(inp['lit_px'])) <= 1e-9, 'A < 1 branch with S0 <= Q')
        return inp
    key = C.digest(dict(cand=tpl['cand_id'], family=fam, phase=ph))
    cache = _S.setdefault('cap_cache', {})
    if key not in cache:
        b = S['PT'].build(RI.with_phase(tpl, fam, ph))
        binary = (np.asarray(b['binary']) > 0).astype(np.float32); lit = float(binary.sum(dtype=np.float64))
        G.require(float(b['peak']) == 1.0 and lit < G.Q, f'A = 1 branch requires S0 < Q (S0 = {lit})')
        cache[key] = dict(binary=binary, peak=1.0, pattern=(binary * np.float32(1.0)).astype(np.float32), raster_key=S['RDY'].raster_key(binary), lit_px=lit,
                          centres=b['centres'], classes=b['classes'], meta_full=b['meta_full'], brightness=b['brightness'], registered_hash_ok=b['registered_hash_ok'],
                          table=None, enc_spec=None, ordinary=None)
    return cache[key]


def phases() -> dict:
    S = platform()
    return S['RM'].phases5(S['M5'])


# Construct a complete candidate case with phase, noise and forward-model identities.
def plan_row(cfg, surface, b, sigma, rep, env, part='CB01_MAIN', ph=None, seed_base=None, block_label=None) -> dict:

    S = platform(); RM, RI, K5 = S['RM'], S['p']['RI'], S['K'].K5
    fam = G.fam_of(cfg); tpl = S['T'][cfg]['row']
    ph = ph or phases()[fam][b]
    inp = inputs(tpl, fam, ph)
    blk = block_label if block_label is not None else ph['block']
    seed = None if float(sigma) == 0.0 else G.seed_of(fam, surface, ph['block'], rep, seed_base)
    fld = RM.field_of(env)
    G.require(fld == G.FIELDS[env], f'Field definition drift for {env}')
    r = dict(platform_surface=RM.SURF_SID[surface], sigma=float(sigma), seed=seed, psf_true_px=float(RM.ENVS[env][0]), field=fld)
    G.require(r['psf_true_px'] == G.PSF[env], 'PSF drift')
    spec, rec, route = RI.forward_record(r, inp)
    fk = K5.digest(rec); bind = RI.decoder_binding(fam, tpl, inp)
    return dict(part=part, family=fam, config=cfg, template_cand_id=tpl['cand_id'], template_source=S['T'][cfg]['source'], surface=surface, platform_surface=RM.SURF_SID[surface],
                phase_block=blk, phase_seq_k=ph['k'], phase_u=ph.get('u'), phase_v=ph.get('v'), phase_frac=ph.get('phi'), sigma=float(sigma), rep=int(rep), seed=seed,
                m3_seed=int(spec['noise']['m3_seed']), noise_stream=spec['noise'].get('stream'), env=env, psf_true_px=float(RM.ENVS[env][0]), field_kind=fld['kind'],
                field_json=json.dumps(fld, sort_keys=True), field_id=spec['field_id'], rays_per_pixel=int(spec['rays_per_pixel']), forward_route=route, adapter_file_sha256=rec['adapter_file'],
                raster_key=inp['raster_key'], pattern_sha=rec['pattern'], peak=float(inp['peak']), peak_f32=rec['peak_f32'], integral=rec['integral'], lit_px=float(inp['lit_px']),
                mesh_sha16=rec['mesh_sha16'], platform=rec['platform'], m3_file_sha256=rec['m3_file'], forward_key=fk, method=bind['method'],
                method_binding=('P4_C3_cal (five-node table c2_exact_scales_fp1e-4_v1), steger_sigma ' + f"{bind['steger_sigma_px']:.10g} px" if bind['method'] == 'P4'
                                else f"DF rule, radius {bind['radius_mm']} mm"),
                case_key=G.case_key(cfg, surface, blk, sigma, rep, env))


# Build a candidate raster and verify its image conditions against the registered manifest row.
def prep(row: dict) -> dict:

    S = platform(); p = S['p']; RI, IN, K = p['RI'], p['IN'], S['K']; K5 = K.K5
    fam = row['family']; cfg = row['config']
    G.require(cfg in S['T'], f'{row["case_key"]}: no template for {cfg}')
    tpl = S['T'][cfg]['row']
    ph = dict(u=float(row['phase_u']), v=float(row['phase_v'])) if fam == 'point' else dict(phi=float(row['phase_frac']))
    inp = inputs(tpl, fam, ph)
    pat = np.asarray(inp['pattern'], np.float32)
    chk = dict(raster=inp['raster_key'] == row['raster_key'], pattern=K5.content_sha(pat) == row['pattern_sha'], peak=float(np.float32(inp['peak'])) == float(row['peak_f32']),
               integral=abs(float(pat.sum(dtype=np.float64)) - float(row['integral'])) <= 1e-6, template=tpl['cand_id'] == row['template_cand_id'],
               resource=abs(float(inp['peak']) - min(1.0, G.Q / float(inp['lit_px']))) <= 1e-9 and float(pat.sum(dtype=np.float64)) <= G.Q * (1 + 2 ** -23))
    G.require(all(chk.values()), f'{row["case_key"]}: inputs differ from the manifest {chk}')
    r = dict(platform_surface=row['platform_surface'], sigma=float(row['sigma']), seed=row.get('seed'), psf_true_px=float(row['psf_true_px']), field_json=row['field_json'])
    spec, rec, route = RI.forward_record(r, inp)
    fk = K5.digest(rec)
    G.require(fk == row['forward_key'] and route == row['forward_route'], f'{row["case_key"]}: recomputed forward key / route differs from the manifest')
    ik = IN.input_key(tpl['cand_id'], fam, ph, inp); p['q'].register(ik, inp); K.set_sandbox_env(); B.sandbox()
    radius = float(tpl['radius_mm']) if fam == 'point' else float('nan')
    return dict(row=row, case_id=row['case_key'], family=fam, sid=row['platform_surface'], surface=row['surface'], pattern=pat, spec=spec, forward_key=fk, route=route,
                forward_record=rec, decoder_row=dict(input_key=ik, radius_mm=radius, family=fam), ctx=dict(meta_full=inp['meta_full'], binary=inp['binary']), method=row['method'],
                input_key=ik, inputs=dict(template=tpl['cand_id'], raster_key=inp['raster_key'], peak=float(inp['peak']), lit_px=float(inp['lit_px'])))


def raster_info(cfg, ph) -> dict:

    S = platform(); fam = G.fam_of(cfg); tpl = S['T'][cfg]['row']; inp = inputs(tpl, fam, ph)
    out = dict(config=cfg, raster_key=inp['raster_key'], lit_px=float(inp['lit_px']), peak=float(inp['peak']), A=min(1.0, G.Q / float(inp['lit_px'])),
               integral=float(np.asarray(inp['pattern'], np.float64).sum()), integral_share_of_Q=float(np.asarray(inp['pattern'], np.float64).sum()) / G.Q,
               active_limit='peak_limited (A = 1, S0 < Q)' if float(inp['lit_px']) < G.Q else 'integral_limited (A = Q / S0)')
    if fam == 'point':
        c = np.asarray(inp['centres'], float); out.update(n_features=int(len(c)), registered_identities=int(len(c)))
        if inp.get('classes') is not None:
            out['n_classes'] = int(len(set(np.asarray(inp['classes']).tolist())))
    else:
        ol = inp['meta_full']['olines']
        ids = ol.get('ids') if isinstance(ol, dict) else None
        out.update(n_features=int(ol['n_lines']), width_px=float(ol['width_px']), spacing_px=float(ol['spacing_px']), registered_identities=(len(ids) if ids is not None else None))
    return out
