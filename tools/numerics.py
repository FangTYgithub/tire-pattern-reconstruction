# Connect public experiment entries to the scientific rendering and evaluation functions.

import sys
import time
import numpy as np
from runtime import ENGINE, SR, path


# Render and decode one environment case, then evaluate the registered tolerance grid.
def chapter4(row):
    sys.path.insert(0, str(ENGINE / 'scripts/ch4_bridge01'))
    import cb_adapter as A
    p = A.prep(row)
    chain = A.platform()['ch']
    started = time.monotonic()
    image, diagnostics = chain.render(p)
    decoded = chain.decode_eval(p, image)
    result = chain.result_row(p, image, diagnostics, decoded, time.monotonic()-started)
    result.update({k: row[k] for k in A.C.IDENTITY if k in row})
    sensitivity = A.B.multi_eval(decoded['ev'], row['surface'])
    main = next(x for x in sensitivity if x['d_mm'] == 6 and x['tau_mm'] == .15)
    if abs(main['Ctau']-result['glob_Ctau']) > 1e-10:
        raise ValueError('Sensitivity and main evaluation disagree')
    return [dict(method=row['method'], result=result, sensitivity=sensitivity)], image, {row['method']: decoded['ev']}


# Rebuild a registered raster and run D0/DF for points or P4 for lines.
def size_spacing(row, size_extension=False):
    path('s345_ready')
    path('s12_resource01')
    import s12_common as K
    import s12_case as SC12
    import rdy_platform as P
    import s1r_common as S1
    import s1r_eval as E
    path('s3_code02'); path('s3_code01')
    import s3c_dev as DV
    if size_extension:
        raster = S1.build_raster('point', float(row['size_mm']), row['phase_label'])
        if raster['key'] != row['raster_key']:
            raise ValueError('Size-extension raster differs')
    else:
        raster = SC12.raster_of(row)
        SC12.peak_of(row, raster)
    peak = float(row['peak_target'])
    context = SC12.context(row, raster, peak)
    sigma = float(row['sigma'])
    seed = 0 if sigma == 0 else int(row['noise_seed'])
    bound = P.bound()
    image = bound.call('render', bound.sid(row['surface']), context['pattern'], peak, sigma, seed, K.PSF_PX)['image']
    methods = ['D0', 'DF'] if row['family'] == 'point' else ['P4']
    out = []; clouds = {}
    for method in methods:
        if method == 'DF':
            detected = DV.dot_decode(image, raster['binary'], 1, [float(row['size_mm'])], 'DF', 'binding')
            decoded = DV.dot_dec(detected, 'single')
        else:
            decoded = bound.call('decode', bound.method(K.ARM[row['family']]), image, context)
        sid = row['platform_surface']
        metrics, arrays, _, _ = E.evaluate_case(sid, row['family'], decoded, context, P.mesh(row['surface']), P.dom_local(row['surface']), P.dom_global(), sid == 'A_curv_gauss')
        out.append(dict(method=method, result=dict(row, **metrics)))
        clouds[method] = arrays
    return out, image, clouds


# Compare the registered decoding and class-use arms on the same image.
def coding(row, complete_identity=False):
    path('s3_code04')
    import c4_common
    import c4_store as S
    import c4_run as R
    import c3_eval as E
    import rdy_platform as P
    inputs = S.inputs_of(row)
    _, pattern, peak, sigma, seed = R.render_contract(row, inputs)
    bound = P.bound()
    image = bound.call('render', bound.sid(row['surface']), pattern, peak, sigma, seed, 1.)['image']
    reference = image
    if row['family'] == 'dots' and sigma > 0:
        reference = bound.call('render', bound.sid(row['surface']), pattern, peak, 0., 0, 1.)['image']
    context = E.ctx_of(row, inputs)
    front_cache = {}; lines = {}; out = []; clouds = {}
    for name, method, params in R.rules_for(row):
        packed, metadata = S.decode_one(image, inputs, row, method, params, front_cache)
        decoded = S.unpack(row['family'], packed, metadata)
        if method == 'dot':
            evaluated = E.eval_dots(image, reference, inputs, row, decoded, context)
            for arm, (metrics, arrays) in evaluated.items():
                out.append(dict(method=name, arm=arm, result=dict(row, **metrics)))
                clouds[name+'_'+arm] = arrays
        else:
            lines[name] = decoded
    if lines:
        evaluated, truth, confirmable = E.eval_lines(row, inputs, lines, context)
        for arm, (metrics, arrays) in evaluated.items():
            result = dict(row, **metrics)
            if complete_identity:
                path('thesis_reinforce01')
                import tr_a4 as ID
                import s1r_common as S1
                mesh = P.mesh(row['surface'])
                olines = inputs['meta_full']['olines']
                scan = ID.scan_raster(inputs['binary'], olines)
                points = np.asarray(arrays['pts'], np.float64)
                valid = arrays['valid']
                identity = ID.reidentify(mesh, lines[arm]['features'], olines, scan['extra'], inputs['binary'] > 0,
                                         arrays['acc'], arrays['lid'], confirmable, truth, valid,
                                         valid & S1.in_global(points[:, :2], P.dom_global()), valid & S1.in_local(points[:, :2]))
                if not identity['closure_old_identity']:
                    raise ValueError('Complete identity failed to reproduce the original identity rule')
                result.update({'full_'+k: v for k, v in identity.items()})
            out.append(dict(method=arm, arm=arm, result=result))
            clouds[arm] = arrays
    return out, image, clouds


# Render a position-adjusted pattern and evaluate each registered strategy view.
def layout(row):
    path('s4_real02')
    import r2_common as K
    import r2_run as R
    import r2_eval as E
    import rdy_platform as P
    import rdy_decode as D
    import rdy_eval as RE
    import pb_m3
    inputs = R.inputs_of(row)
    image = pb_m3.render(P.bound().sid(row['surface']), inputs['pattern'], float(inputs['peak']), float(row['sigma']), int(row['render_seed']), K.RK.PSF_PX, K.BOUNDARY)['image']
    decoded = D.decode(row['method'], image, inputs, R._rowlike(row))['arms'][row['arm']]
    metrics, arrays, _ = E.evaluate(row['family'], row['surface'], decoded, inputs, K.BOUNDARY, E.domain('omega1'))
    result = dict(row, **metrics)
    if row['family'] == 'point':
        reference = image if float(row['sigma']) == 0 else pb_m3.render(P.bound().sid(row['surface']), inputs['pattern'], float(inputs['peak']), 0., 0, K.RK.PSF_PX, K.BOUNDARY)['image']
        identity, code = RE.source_identity(image, reference, inputs, decoded, P.mesh(row['surface']))
        result['source_identity'] = identity
    out = [dict(method=row['method'], result=result)]
    clouds = {row['method']: arrays}
    if row['family'] == 'point':
        path('s3_code01')
        import s3c_dev as DV
        import sp2_common
        df = DV.dot_dec(DV.dot_decode(image, np.asarray(inputs['binary'], np.float32), 1, [float(sp2_common.R_MM)], 'DF', 'binding'), 'single')
        metrics, arrays, _ = E.evaluate('point', row['surface'], df, inputs, K.BOUNDARY, E.domain('omega1'))
        out.append(dict(method='DF', result=dict(row, **metrics)))
        clouds['DF'] = arrays
    return out, image, clouds


# Run one case of the joint size-spacing experiment.
def joint(row):
    import pandas as pd
    path('s5_align03'); path('s5_align04')
    import s5r_main as R
    import s5t_platform as P
    key = (row['combo'], row['surface'], row['phase_label'], row['sigma'], row['rep'])
    request = R.manifest_rows([key], pd.DataFrame([row]))[0]
    executor = P.RealQ()
    if not executor.check_inputs(request, row)['ok']:
        raise ValueError('Joint-study generated pattern differs from registration')
    images, _ = executor.render(request)
    out = []; clouds = {}
    for method in R.methods_of(request):
        decoded, metadata = executor.decode(request, method, images)
        arrays, evaluated = executor.evaluate(dict(request, _decode_meta=metadata), method, decoded)
        out.append(dict(method=method, result=dict(row, **evaluated['metrics']), support=evaluated['support']))
        clouds[method] = arrays
    return out, images['image'], clouds


# Evaluate a held-out layout with its registered decoding arms.
def layout_validation(row):
    path('s4_real07')
    import s4r7_run as R
    import s4r7_eval as E
    import rdy_platform as P
    import rdy_decode as D
    import pb_m3
    inputs = R.inputs(row)
    image = pb_m3.render(P.bound().sid(row['surface']), inputs['pattern'], float(inputs['peak']), float(row['sigma']), int(row['render_seed']), R.K.RK.PSF_PX, R.K.BOUNDARY)['image']
    reference = image
    if row['family'] == 'point' and float(row['sigma']) > 0:
        reference = pb_m3.render(P.bound().sid(row['surface']), inputs['pattern'], float(inputs['peak']), 0., 0, R.K.RK.PSF_PX, R.K.BOUNDARY)['image']
    out = []; clouds = {}
    methods = ['D0','DF'] if row['family']=='point' else ['P4TABLE','CLIP']
    for method in methods:
        if method == 'DF':
            path('s3_code01')
            import s3c_dev as DV
            import sp2_common
            decoded = DV.dot_dec(DV.dot_decode(image, inputs['binary'], 1, [float(sp2_common.R_MM)], 'DF', 'binding'), 'single')
        elif method == 'CLIP':
            import s4r5_clip
            params, config = R._params(row, inputs, method)
            raw = s4r5_clip.run_p4_table_clip(image, inputs['table']['spec'], config, inputs['pattern'], params, 'abstain')
            decoded = dict(pts=np.asarray(raw['points'],np.float64).reshape(-1,3),accepted=np.asarray(raw['accepted'],bool),aux=dict(r=raw))
        else:
            params, _ = R._params(row, inputs, method)
            decoded = D.decode(method, image, inputs, dict(section='S4',family=row['family'],arm=R.ARM[method],radius_mm=params.get('radius_mm',''),strategy=row['layout']))['arms'][R.ARM[method]]
        if method == 'DF':

            import r2_eval as E2
            metrics, arrays, _ = E2.evaluate('point', row['surface'], decoded, inputs, R.K.BOUNDARY, E2.domain('omega1'))
            extra = {'nodes': None}
        else:
            metrics, arrays, extra = E.evaluate(row['family'], row['surface'], decoded, inputs, image, reference)
        out.append(dict(method=method,result=dict(row,**metrics),nodes=extra['nodes'],source_identity=extra.get('source_identity')))
        clouds[method] = arrays
    return out,image,clouds
