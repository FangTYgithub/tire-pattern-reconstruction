# Convert phase specifications into dot and line raster inputs.

from __future__ import annotations

import json

import numpy as np

import s4r7_common as H

K = H.K


def phase_table() -> dict:
    import sp2_common as S2
    r5 = json.loads(H.reg(H.R5 / 'run.json').read_text(encoding='utf-8'))['A2_reserve']
    out = {'dots': {}, 'lines': {}}
    for fam in ('dots', 'lines'):
        for p in S2.registered_phases(fam):
            if p['phase_id'] in ('H0', 'H1'):
                out[fam][p['phase_id']] = dict(dict(u=p['u'], v=p['v']) if fam == 'dots' else dict(phase_frac=p['phase_frac'], k64=p['k64']), source='sp2_common.registered_phases (seen, regression / anchor)')
    for i, d in enumerate(r5['dots_phases'][:2]):
        out['dots'][f'R{i}'] = dict(u=float(d['u']), v=float(d['v']), source='s4_real05 A2_reserve dots_phases[%d] = holdout32[%d]' % (i, 24 + i))
    for i, d in enumerate(r5['lines_phases'][:2]):
        out['lines'][f'R{i}'] = dict(phase_frac=float(d['phase_frac']), k64=d['k64'], source='s4_real05 A2_reserve lines_phases[%d]' % i)
    return out


_PT = None


def ph(fam: str, phase_id: str) -> dict:
    global _PT
    if _PT is None:
        _PT = phase_table()
    return _PT[fam][phase_id]


def table(name: str, layout: dict, phase_id: str) -> dict:

    import sp2_common as S2
    fam = 'dots' if name.startswith('R') else 'lines'; p = ph(fam, phase_id)
    disp = {int(k): float(v) for k, v in layout['disp_px'].items()}
    out = dict(baseline=name, proposal_id=layout['layout_id'], phase_id=phase_id, disp=disp, j=float(layout.get('j', 0.0) if disp else 0.0), arms=[s for s, _b in layout.get('strategies', [])],
               family=fam, phase=dict(p, phase_id=phase_id))
    if fam == 'dots':
        import sp_dots_columns as DC
        lay = DC.layout(*S2.DOT_BASELINES[name], p['u'], p['v'], disp or None, 'u'); lay0 = DC.layout(*S2.DOT_BASELINES[name], p['u'], p['v'], None, 'u')
        out.update(pat=np.asarray(DC.raster(lay['c'])), centres=np.asarray(lay['c'], np.float64), centres_regular=np.asarray(lay0['c'], np.float64), ix=np.asarray(lay['ix']), spec=None,
                   present=sorted(set(int(i) for i in lay['ix'])), n_elements=int(lay['n']), boundary_in=int(lay.get('boundary_in', 0)), boundary_out=int(lay.get('boundary_out', 0)))
    else:
        import sp2_lines_table as LT2
        pat, meta = LT2.table_spec(S2.LINE_BASELINES[name], S2.W_MM, p['phase_frac'], disp or None)
        ids = sorted(int(k) for k in meta['olines']['ids'])
        out.update(pat=np.asarray(pat), centres=None, spec=meta['olines'], meta=meta, present=ids, n_elements=len(ids))
    return out


def build_input(name: str, layout: dict, phase_id: str) -> dict:

    import rdy_inputs as RI
    T = table(name, layout, phase_id)
    binary = (np.asarray(T['pat']) > 0).astype(np.float32); lit = float(binary.sum(dtype=np.float64))
    peak = H.Q / lit if lit > 0 else float('inf'); pk32 = np.float32(peak); pattern = (binary * pk32).astype(np.float32); integral = float(pattern.sum(dtype=np.float64))
    rel_err = abs(integral - H.Q) / H.Q
    return dict(binary=binary, peak=float(peak), pattern=pattern, raster_key=RI.raster_key(binary), lit_px=lit, integral_f32=integral, integral_rel_err=rel_err, centres=T['centres'], classes=None,
                meta_full=(None if T['family'] == 'dots' else dict(olines=T['spec'])), brightness='peak = Q 11695 / lit px (S4 rule)', registered_hash_ok=None, table=T, enc_spec=None, ordinary=None,
                resource_feasible=bool(peak <= 1.0 + 1e-12 and rel_err <= 1e-6))
