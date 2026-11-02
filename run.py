"""Thesis experiments: plan, run, export."""
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
from runtime import ENGINE, setup, clean

P = 'research/project_Part1_v1.7/outputs/surface_revision/'
STUDIES = {
    'S12': (P+'s12_resource01/inputs.csv', 'input_key', 's12'),
    'SIZE02': ('outputs/s1_size02/manifest.csv', 'input_key', 'size'),
    'CODE07': (P+'s3_code04/formal/manifest.csv', 'image_key', 'coding'),
    'LAYOUT02': (P+'s4_real02/manifest.csv', 'image_key', 'layout'),
    'LAYOUT07': (P+'s4_real07/manifest.csv', 'image_key', 'layout_validation'),
    'JOINT05': (P+'s5_align02/manifest_next.csv', 'manifest_key', 'joint'),
    'CH4-BLUR': ('outputs/mp06c_zero_close_2026-10-21/blur_manifest_7920.csv', 'case_key', 'ch4'),
    'CH4-LIGHT': ('outputs/mp06f_lighting/manifest.csv', 'case_key', 'ch4'),
    'CH4-BRIDGE': ('outputs/ch4_bridge01/manifest_main.csv', 'case_key', 'ch4'),
    'CH4-VALID': ('outputs/ch4_valid02/manifest.csv', 'case_key', 'ch4'),
    'TR5': (P+'thesis_reinforce05/manifest.csv', 'forward_key', 'new_phases'),
    'SUBRAY-CH2': ('delivery_inputs/subray_ch2.csv', 'case_key', 'subray2'),
    'SAMPLING-S12': ('delivery_inputs/sampling_s12.csv', 'input_key', 'sampling'),
    'MP05': (P+'model_prep05/formal_manifest.csv', 'case_key', 'pilot'),
    'PSF-DIAG': ('delivery_inputs/psf_diagnostic.csv', 'case_key', 'psf_diagnostic'),
    'MP06C': (P+'model_prep06ar/manifest.csv', 'case_key', 'environment_extension'),
    'MP06C-HOLDOUT': (P+'model_prep06c/holdout_manifest.csv', 'case_key', 'environment_extension'),
    'MP06E-HOLDOUT': ('outputs/mp06e_holdout/manifest.csv', 'case_key', 'ch4'),
    'BRIDGE-HOLDOUT': ('outputs/ch4_bridge01/manifest_holdout.csv', 'case_key', 'ch4'),
    'SUBRAY-CH4': ('outputs/ch4_bridge01/manifest_subray.csv', 'case_key', 'subray4'),
    'S12-IDENTITY': ('delivery_inputs/line_identity.csv', 'input_key', 'identity'),
    'LAYOUT-DIAG': ('delivery_inputs/layout_diagnostic.csv', 'image_key', 'layout_diagnostic'),
    'CODE-IDENTITY': ('delivery_inputs/code_identity.csv', 'image_key', 'coding_identity'),
    'SUBRAY-MP05': ('delivery_inputs/subray_pilot.csv', 'case_key', 'subray_pilot'),
}


# Load the selected experiment manifest and its case-key definition.
def load(name):
    import pandas as pd
    rel, key, backend = STUDIES[name]
    frame = pd.read_csv(ENGINE / rel, float_precision='round_trip', low_memory=False)
    return frame, key, backend


def filter_rows(frame, expressions):
    import pandas as pd
    for expression in expressions:
        if '=' not in expression:
            raise ValueError('Use COLUMN=VALUE')
        col, val = expression.split('=', 1)
        if col not in frame:
            raise ValueError('Unknown filter column: '+col)
        frame = frame[frame[col].eq(float(val))] if pd.api.types.is_numeric_dtype(frame[col]) else frame[frame[col].astype(str).eq(val)]
    return frame


# Dispatch one case to its scientific computation adapter.
def compute(row, backend):
    import numerics as N
    if backend == 'ch4': return N.chapter4(row)
    if backend == 's12': return N.size_spacing(row)
    if backend == 'size': return N.size_spacing(row, True)
    if backend == 'coding': return N.coding(row)
    if backend == 'coding_identity': return N.coding(row, True)
    if backend == 'layout': return N.layout(row)
    if backend == 'joint': return N.joint(row)
    if backend == 'layout_validation': return N.layout_validation(row)
    from supplement import BACKENDS
    if backend in BACKENDS: return BACKENDS[backend](row)
    raise ValueError(backend)


# Run selected cases and store their results, with optional images and point clouds.
def run(args):
    from integrity import verify
    binding = verify()['binding']
    frame, key, backend = load(args.study)
    frame = filter_rows(frame, args.where).drop_duplicates(key)
    if args.limit is not None:
        if args.limit < 1: raise ValueError('Limit must be positive')
        frame = frame.head(args.limit)
    if frame.empty: raise ValueError('No matching cases')
    output = setup(args.output)
    con = sqlite3.connect(output/'results.sqlite')
    con.execute('CREATE TABLE IF NOT EXISTS cases(study TEXT,k TEXT,binding TEXT,request TEXT,result TEXT,PRIMARY KEY(study,k))')
    done = 0
    for row in frame.to_dict('records'):
        row = clean(row); case_key = str(row[key])
        request = json.dumps(row, sort_keys=True, ensure_ascii=False)
        prior = con.execute('SELECT binding,request FROM cases WHERE study=? AND k=?', (args.study,case_key)).fetchone()
        if prior:
            if prior != (binding,request): raise ValueError('Existing result belongs to different code or inputs')
            print('REUSE', case_key, flush=True); continue
        started = time.monotonic()
        results, image, clouds = compute(row, backend)
        for result in results:
            result['image_sha256'] = hashlib.sha256(image.tobytes()).hexdigest() if image is not None else None
        ident = hashlib.sha256((args.study+case_key).encode()).hexdigest()[:20]
        if args.save_images and image is not None:
            from PIL import Image
            Image.fromarray(image).save(output/(ident+'.png'))
        if args.save_clouds or args.save_details:
            import numpy as np
            for method, arrays in clouds.items():
                keep = {k:arrays[k] for k in ('pts','valid','accepted','acc','zr','point_correct','point_wrong','point_unconf') if arrays is not None and k in arrays}
                if args.save_details:
                    keep = {k:np.asarray(v) for k,v in arrays.items()}
                    if any(v.dtype.hasobject for v in keep.values()):
                        raise ValueError('Evaluation details must be numeric arrays, not pickled objects')
                label = ''.join(c if c.isalnum() or c in '-_' else '_' for c in method)
                np.savez_compressed(output/(ident+'_'+label+'.npz'), **keep)
        con.execute('INSERT INTO cases VALUES(?,?,?,?,?)',(args.study,case_key,binding,request,json.dumps(clean(results),ensure_ascii=False,allow_nan=False)))
        con.commit();done += 1
        print(f'OK {case_key} {time.monotonic()-started:.1f}s', flush=True)
    con.close()
    print(json.dumps({'selected':len(frame),'computed':done,'reused':len(frame)-done,'output':str(output)}))


# Compare representative Chapter 4 outputs with stored numerical references.
def check(args):
    from integrity import verify
    verified = verify()
    output = setup(args.output)
    fixtures = json.loads((ROOT/'tests/cases.json').read_text('utf-8'))
    metrics = ['glob_Ctau','glob_Cvalid','glob_pt_rmse','glob_pt_p95','glob_surf_rmse','n_valid','glob_n_valid','n_accepted','c','w','u','image_sha']
    records = []
    for item in fixtures:
        row = item['row']; result = compute(row,'ch4')[0][0]['result']; expected = item['expected_result']
        for name in metrics:
            a,b = result.get(name),expected.get(name)
            equal = a==b if isinstance(a,str) or a is None or b is None else abs(float(a)-float(b)) <= 1e-12
            if not equal: raise AssertionError(f'{row["case_key"]} {name}: {a} != {b}')
        records.append(row['case_key']); print('PASS',row['case_key'],flush=True)
    (output/'check.json').write_text(json.dumps(dict(status='passed',binding=verified['binding'],cases=len(records),new_renders=len(records),historical_arrays_used=0,keys=records),indent=2),'utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    sub.add_parser('verify')
    q=sub.add_parser('plan');q.add_argument('study',choices=STUDIES);q.add_argument('--where',action='append',default=[])
    q=sub.add_parser('run');q.add_argument('study',choices=STUDIES);q.add_argument('--where',action='append',default=[])
    scope=q.add_mutually_exclusive_group(required=True);scope.add_argument('--all',action='store_true');scope.add_argument('--limit',type=int)
    q.add_argument('--output',type=Path,default=ROOT/'outputs');q.add_argument('--save-images',action='store_true');q.add_argument('--save-clouds',action='store_true');q.add_argument('--save-details',action='store_true')
    q=sub.add_parser('check');q.add_argument('--output',type=Path,default=ROOT/'outputs/check')
    q=sub.add_parser('export');q.add_argument('--database',type=Path,required=True);q.add_argument('--output',type=Path,default=ROOT/'outputs/export')
    q=sub.add_parser('select-ch4');q.add_argument('study',choices=['CH4-BLUR','CH4-LIGHT','CH4-VALID']);q.add_argument('--database',type=Path,required=True);q.add_argument('--output',type=Path,default=ROOT/'outputs/selection')
    args=p.parse_args()
    if args.command=='verify':
        from integrity import verify
        print(json.dumps(verify()))
    elif args.command=='list':
        for name in STUDIES:
            f,k,b=load(name);print(name,len(f),'views;',f[k].nunique(),'cases;',b)
    elif args.command=='plan':
        f,k,b=load(args.study);f=filter_rows(f,args.where)
        print(json.dumps({'views':len(f),'cases':f[k].nunique(),'images':0 if b=='sampling' else f[k].nunique(),'columns':list(f)}))
    elif args.command=='check':check(args)
    elif args.command in ('export','select-ch4'):
        database=args.database.resolve();output=setup(args.output)
        from reporting import export,select_ch4
        result=export(database,output)[2] if args.command=='export' else select_ch4(database,args.study,output)
        print(json.dumps(clean(result)))
    else:run(args)


if __name__=='__main__':main()
