# Expose new-phase, sub-ray, identity and ideal-sampling studies through the public runner.

import json
import sys
import time
import numpy as np
from runtime import ENGINE, PROJECT, path


# Run one new-phase case and add complete-line identity counts where applicable.
def new_phases(row):
    path('thesis_reinforce05')
    import tr5_manifest as M
    import tr5_chain as C
    context = M.context(row)
    backend = C.RealBackend()
    image, diagnostics = backend.render(row, context)
    out, clouds = [], {}
    for method in row['methods'].split(';'):
        dec = backend.decode(method, image, row, context)
        metrics, arrays, info = backend.evaluate(row['family'], dec, row, context)
        full, identity = C.full_identity(row, context, arrays)
        arrays.update(full)
        out.append(dict(method=method, result=dict(row, **metrics),
                        full_identity=identity, diagnostics=diagnostics))
        clouds[method] = arrays
    return out, image, clouds


# Render the requested ray-sampling setting and evaluate the registered methods.
def subrays_chapter2(row):
    path('thesis_reinforce04')
    import tr4_prep as P
    import tr4_chain as C
    record = json.loads((ENGINE / row['source_metadata']).read_text('utf-8'))
    prep = P.prepare(row, int(row['rays_per_pixel']), record=record)
    backend = C.RealBackend()
    image, diagnostics = backend.render(prep)
    out, clouds = [], {}
    for method in row['methods'].split(';'):
        dec = backend.decode(method, image, prep)
        metrics, arrays, info = backend.evaluate(row['family'], dec, prep)
        out.append(dict(method=method, result=dict(row, **metrics), diagnostics=diagnostics))
        clouds[method] = arrays
    return out, image, clouds


# Evaluate an environment case, optionally with the four PSF diagnostic settings.
def environment_pilot(row, adapted=False):
    path('model_prep06a')
    import m6_platform as P
    import mp5_chain as C
    prep = P.prep(row)
    backend = C.RealBackend()
    forward = backend.render(prep)
    image = forward['image']
    out, clouds = [], {}
    if adapted:
        if row['family'] != 'line':
            raise ValueError('PSF diagnostic is registered for lines only')
        table = json.loads((PROJECT/'outputs/surface_revision/model_prep06a/arm_table.json').read_text('utf-8'))
        arms = [a for a in table if a['config'] == row['config'] and a['env'] == row['env']]
        if {a['arm'] for a in arms} != {'N', 'T', 'S', 'ST'}:
            raise ValueError('No complete registered PSF diagnostic for this condition')
        cache = {}
        for a in arms:
            settings = (float(a['steger_sigma_px']), float(a['psf_design_px']))
            if settings not in cache:
                raw, meta, thresholds = P.p4_decode(image, prep, *settings)
                arrays, rec, identity = P.evaluate(prep, 'P4', raw, meta)
                cache[settings] = arrays, rec, thresholds
            arrays, rec, thresholds = cache[settings]
            result = dict(row, **rec['metrics'], **P.cwu(rec, 'line'))
            out.append(dict(method=a['arm'], arm=a['arm'], result=result,
                            settings=a, thresholds=thresholds, full_identity=rec['full_identity']))
            clouds[a['arm']] = arrays
    else:
        for method in row['methods'].split(';'):
            raw, meta = backend.decode(prep, method, image)
            arrays, rec, identity = P.evaluate(prep, method, raw, meta)
            out.append(dict(method=method, result=dict(row, **rec['metrics'], **P.cwu(rec, row['family'])),
                            full_identity=identity, diagnostics=forward['diag']))
            clouds[method] = arrays
    return out, image, clouds


# Render the sub-ray counterpart of a registered Chapter 4 parent case.
def subrays_chapter4(row):
    sys.path.insert(0, str(ENGINE/'scripts/ch4_bridge01'))
    import cb_adapter as A
    import cb_run as R
    parent = dict(row, case_key=row['parent_case_key'], forward_key=row['parent_forward_key'], rays_per_pixel=1)
    prep = A.prep(parent)
    prep['case_id'] = row['case_key']
    chain = A.platform()['ch']
    started = time.monotonic()
    image, diagnostics, forward_key = R.render(prep, int(row['rays_per_pixel']))
    if forward_key != row['forward_key']:
        raise ValueError('Sub-ray forward identity differs from registration')
    decoded = chain.decode_eval(prep, image)
    result = chain.result_row(prep, image, diagnostics, decoded, time.monotonic()-started)
    result.update({k: row[k] for k in A.C.IDENTITY if k in row})
    result.update(forward_key=forward_key, rays_per_pixel=row['rays_per_pixel'])
    return [dict(method=row['method'], result=result,
                 sensitivity=A.B.multi_eval(decoded['ev'], row['surface']))], image, {row['method']: decoded['ev']}


# Run a candidate or environment extension through the fixed processing chain.
def environment_extension(row):
    sys.path.insert(0, str(ENGINE/'scripts/ch4_bridge01'))
    import cb_adapter as A
    chain = A.platform()['ch']

    prep = chain.prep(row)
    started = time.monotonic()
    image, diagnostics = chain.render(prep)
    decoded = chain.decode_eval(prep, image)
    result = chain.result_row(prep, image, diagnostics, decoded, time.monotonic()-started)
    return [dict(method=row['method'], result=result,
                 sensitivity=A.B.multi_eval(decoded['ev'], row['surface']))], image, {row['method']: decoded['ev']}


# Evaluate the three-by-three ray counterpart of an environment pilot case.
def subrays_pilot(row):
    path('model_prep06a')
    import m6_stage_b as B
    import m6_platform as P
    out = B._work3(row)
    if not out['ok']:
        raise ValueError(out['error'])
    result = dict(row, **out['rec']['metrics'], **P.cwu(out['rec'], row['family']),
                  rays_per_pixel=3, rendered_forward_key=out['fk3'])
    return [dict(method='P4', result=result, diagnostics=out['diag'])], out['img'], {'P4': out['ev']}


# Place exact heights at expected feature locations and evaluate the resulting sampling reference.
def sampling_reference(row):
    path('s345_ready'); path('s12_resource01'); path('thesis_check01')
    import s12_case as SC
    import rdy_platform as RP
    import tc_ref as T
    raster = SC.raster_of(row)
    context = SC.context(row, raster, float(row['peak_target']))
    sid = row['platform_surface']
    mesh = T.mesh(sid)
    S = RP.s2()
    if row['family'] == 'point':
        expected = S.SL.expected_dots(mesh, context['centres'])
        uv = np.asarray(expected['uv'], np.float64)
        visible = (np.asarray(expected['hit'], bool) & np.isfinite(uv).all(axis=1)
                   & (uv[:, 0] >= 0) & (uv[:, 0] <= S.IMG-1)
                   & (uv[:, 1] >= 0) & (uv[:, 1] <= S.IMG-1))

        xy = np.asarray(expected['xy'], np.float32).astype(np.float64)[visible]
        z = T.truth(sid)(np.column_stack([xy[:, 1], xy[:, 0]]))
        finite = np.isfinite(z)
        points = np.column_stack([xy[finite], z[finite]])
    else:
        reference = S.T.line_reference(mesh, context['meta_full']['olines'], 1., pattern=context['pattern'])
        points = np.asarray(reference['rows']['xyz'], np.float32).astype(np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
    metrics = dict(row, reference_key=T.ref_key('S12', sid, points),
                   n_out_of_window=int(((points[:, 2] < -16) | (points[:, 2] > 4)).sum()))
    for domain in ('glob', 'loc'):
        metrics.update({domain+'_'+k: v for k, v in T.eval_cloud(points, T.domain('S12', sid, domain)).items()})
    return [dict(method='ideal_sampling', result=metrics)], None, {'ideal_sampling': {'pts': points}}


# Re-evaluate line identities against all illuminated projector lines.
def complete_line_identity(row):
    import numerics as N
    out, image, clouds = N.size_spacing(row)
    path('thesis_reinforce05')
    import tr5_chain as T
    import s12_case as SC
    context = SC.context(row, SC.raster_of(row), float(row['peak_target']))
    for item in out:
        arrays = clouds[item['method']]
        full, identity = T.full_identity(row, context, arrays)
        arrays.update(full)
        item['full_identity'] = identity
        for domain, counts in identity.get('counts', {}).items():
            item['result'].update({f'full_{domain}_{k}': v for k, v in counts.items()})
    return out, image, clouds


# Compare ideal sampling, correctly identified samples and actual layout reconstruction.
def layout_diagnostic(row):
    import numerics as N
    out, image, clouds = N.layout(row)
    path('thesis_check01')
    import tc_ref as T
    import rdy_platform as RP
    sid = RP.K.SURF_MAP[row['surface']]
    for item in out:
        arrays = clouds[item['method']]
        if row['family'] == 'point':
            xy = np.asarray(arrays['exp_xy'], np.float64)[arrays['exp_in_image']]
            z = T.truth(sid)(np.column_stack([xy[:, 1], xy[:, 0]]))
            ok = np.isfinite(z)
            reference = np.column_stack([xy[ok], z[ok]])
        else:
            reference = np.asarray(arrays['ref_xyz'], np.float64)
        points = np.asarray(arrays['pts'], np.float64)
        valid = np.asarray(arrays['valid'], bool)
        correct = valid & np.asarray(arrays['point_correct'], bool)

        layers = {'L1_reference': reference, 'L2_confirmed_correct': points[correct], 'L3_official': points[valid]}
        diagnostics = []
        for name, points in layers.items():
            for domain in ('glob', 'loc'):
                metrics = T.eval_cloud(points, T.domain('R7', sid, domain), (3.5, 4., 5., 6.))
                diagnostics.append(dict(layer=name, domain=domain, **metrics))
        item['sampling_layers'] = diagnostics
    return out, image, clouds


BACKENDS = {'new_phases': new_phases, 'subray2': subrays_chapter2,
            'pilot': environment_pilot, 'psf_diagnostic': lambda row: environment_pilot(row, True),
            'subray4': subrays_chapter4, 'sampling': sampling_reference,
            'identity': complete_line_identity, 'layout_diagnostic': layout_diagnostic,
            'environment_extension': environment_extension, 'subray_pilot': subrays_pilot}
