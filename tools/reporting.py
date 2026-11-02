# Export case tables and derive Chapter 4 parameter-selection summaries.

from pathlib import Path
import json
import sqlite3
import sys

import pandas as pd
from runtime import ENGINE, clean


# Export stored scalar results and available diagnostic arrays to analysis tables.
def export(database, output):
    database = Path(database).resolve()
    con = sqlite3.connect('file:'+database.as_posix()+'?mode=ro', uri=True)
    records = []; sensitivity = []; diagnostics = []; coverage = []; identities = {}
    for study, key, binding, request, result in con.execute('SELECT * FROM cases ORDER BY study,k'):
        identities.setdefault(study,set()).add(key)
        for item in json.loads(result):
            row = dict(item['result'], delivery_study=study, delivery_key=key,
                       delivery_method=item['method'], delivery_arm=item.get('arm'), code_binding=binding)
            for domain, counts in (item.get('full_identity') or {}).get('counts', {}).items():
                row.update({f'full_{domain}_{k}': v for k,v in counts.items()})
            row.update({'detector_'+k: v for k,v in item.get('thresholds', {}).items()})
            records.append(row)
            for diagnostic in item.get('sampling_layers', []):
                diagnostics.append(dict(diagnostic, delivery_study=study, delivery_key=key, delivery_method=item['method']))
            for s in item.get('sensitivity',[]):
                sensitivity.append(dict(s,case_key=key,delivery_study=study))
    con.close()
    if not records:
        raise ValueError('Database contains no completed cases')

    import run
    for study, keys in identities.items():
        frame,k,_ = run.load(study)
        expected=set(frame[k].astype(str))
        if not keys.issubset(expected):
            raise ValueError('Result key absent from registered manifest')
        coverage.append(dict(study=study,computed=len(keys),registered=len(expected),complete=keys==expected))
    R=pd.DataFrame(records);R.to_csv(output/'case_metrics.csv',index=False)
    S=pd.DataFrame(sensitivity)
    if len(S):S.to_csv(output/'sensitivity.csv',index=False)
    if diagnostics:pd.DataFrame(diagnostics).to_csv(output/'diagnostics.csv',index=False)
    (output/'coverage.json').write_text(json.dumps(coverage,indent=2),'utf-8')
    return R,S,coverage


# Check matrix completeness before deriving the Chapter 4 selection tables.
def select_ch4(database, study, output):
    if study not in ('CH4-BLUR','CH4-LIGHT','CH4-VALID'):
        raise ValueError('Selection requires the full blur, lighting or validation matrix')
    R,S,coverage=export(database,output)
    state=next((r for r in coverage if r['study']==study),None)
    if not state or not state['complete']:
        raise ValueError('Whole registered matrix required for selection; no partial ranking emitted')
    R=R[R.delivery_study==study].copy();S=S[S.delivery_study==study].copy()
    import run
    expected,_,_=run.load(study)
    sys.path.insert(0,str(ENGINE/'scripts/ch4_bridge01'))
    import cb_common as G
    import analysis as A
    long,surfaces,choices=A.tables(R,S,expected[G.C.IDENTITY])
    surfaces.to_csv(output/'surface_summary.csv',index=False)
    choices.to_csv(output/'choices.csv',index=False)
    return {'status':'passed','study':study,'image_cases':len(R),'choice_rows':len(choices)}
