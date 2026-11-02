# Connect blur cases to rendering, decoding and multi-tolerance surface evaluation.

from __future__ import annotations
import os,sys,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import core as C

_chain=None
_domain=None
_old_tables=None

def sandbox():
    for k,sub in [('LINE01RB_OUT_DIR','line_out'),('LINE01RB_CASES_DIR','line_cases'),('LINE01RB_LEDGER_DIR','line_ledger')]:
        os.environ[k]=str(C.OUT/'scratch'/sub)
    os.environ['LINE01RB_RENDER_CAP']='0'

def init():
    global _chain
    C.install_guard()
    if _chain is None:
        sys.path.insert(0,str(C.PROJECT/'scripts/surface_revision/model_prep06c'))
        import c_chain
        _chain=c_chain
        _chain.platform()

        import rdy_platform as RP
        RP.bound(); RP.dom_global()
        for sf in C.SURFS: RP.dom_local(sf)

        import ep_patterns
        import ep_detect
        import rdy_decode
    sandbox()
    return _chain

def prepare(r):
    ch=init();p=ch.prep(C.rowdict(r));sandbox()
    C.require(p['route']=='mp3_forward','Unexpected rendering route in blur module')
    return p

def multi_eval(ev,surface):
    global _domain
    if _domain is None:
        with np.load(C.DOMAIN,allow_pickle=False) as z:_domain={k:z[k] for k in z.files}
    from surface_eval_r3 import surface_eval_r3_dev
    pts=ev['pts'][ev['valid']].astype(np.float64)
    sid=dict(B_curv='B_curv',M10_curv='A_curv',M24_curv='C_curv')[surface]
    dom=dict(x=_domain['x'],y=_domain['y'],z_ref=_domain['z_'+sid],area=np.full(30947,.25))
    g=surface_eval_r3_dev(pts,None,dom,C.DS,C.TAUS,support_rule='exact')['subdomains']['global']
    return [dict(d_mm=d,tau_mm=t,Ctau=g['by_d'][str(d)]['C_tau'][str(t)],Cvalid=g['by_d'][str(d)]['C_valid'],
                 n_nodes=g['n_nodes'],sup_n=g['by_d'][str(d)]['n_supported'],sup_rmse=g['by_d'][str(d)]['rmse']) for d in C.DS for t in C.TAUS]

def real_compute(r,store,permit):
    C.require(permit=='explicit_user_authorised_production','Real backend is disabled without explicit production permit')
    t0=time.monotonic();ch=init();p=prepare(r);ck=r['case_key']
    checkpoint=store.load_image(r)
    if checkpoint is None:
        store.event(ck,'render','started')
        img,diag=ch.render(p);sandbox()
        store.save_image(r,img,diag)
        store.event(ck,'render','finished')
    else:
        img,diag=checkpoint
        store.event(ck,'render','reused_checkpoint')
    store.event(ck,'decode','started')
    d=ch.decode_eval(p,img);sandbox()
    store.event(ck,'decode','finished')
    result=ch.result_row(p,img,diag,d,time.monotonic()-t0)
    result.update({k:C.rowdict(r)[k] for k in C.IDENTITY})
    sens=multi_eval(d['ev'],r['surface'])
    main=next(x for x in sens if x['d_mm']==6 and x['tau_mm']==.15)
    C.require(abs(main['Ctau']-result['glob_Ctau'])<=1e-10 and abs(main['Cvalid']-result['glob_Cvalid'])<=1e-10,'New global sensitivity evaluator differs from main result')
    C.metric_gate(pd.DataFrame([result]),pd.DataFrame([r]))
    m=r['method'];arrays={'image':img}
    arrays.update({m+'__raw__'+k:v for k,v in d['raw'].items()})
    arrays.update({m+'__ev__'+k:v for k,v in d['ev'].items()})
    meta=dict(case_key=ck,forward_key=r['forward_key'],method=m,result=result,sensitivity=sens,
              inputs=p['inputs'],input_key=p['input_key'],decode_meta=d['dmeta'],diag=diag,
              metrics=d['rec']['metrics'],full_identity=d['rec'].get('full_identity'),
              scientific=True,backend='frozen_DF_P4_1x1')
    return meta,arrays

def old_results():
    global _old_tables
    if _old_tables is None:
        base=C.PROJECT/'outputs/surface_revision'
        a=C.csv(base/'model_prep05/results.csv')
        a=a[((a.family=='point')&(a.method=='DF'))|((a.family=='line')&(a.method=='P4'))]
        b=C.csv(base/'model_prep06c/results_new.csv')
        t=pd.concat([a,b],ignore_index=True)
        C.require(not t.case_key.duplicated().any(),'Duplicate source result key')
        _old_tables=t.set_index('case_key')
    return _old_tables

def source_records(man,verify_hash=True):

    rows=[];old=old_results()
    for r in man.to_dict('records'):
        C.require(r['action']=='reuse','Source reader received new target')
        if verify_hash:C.require(C.sha(r['source_pack'])==r['source_pack_sha256'],'Source pack drift: '+r['case_key'])
        v=C.rowdict(old.loc[r['case_key']].to_dict()|{'case_key':r['case_key']})
        for k in C.IDENTITY:
            if k in v and v[k] is not None:C.require(v[k]==C.rowdict(r)[k],'Source identity mismatch: '+k)
        v.update({k:C.rowdict(r)[k] for k in C.IDENTITY})
        rows.append(v)
    t=pd.DataFrame(rows);C.metric_gate(t,man)
    return t

def source_sensitivity(man):
    t=C.csv(C.PRIOR/'sensitivity_cases.csv.gz')
    t=t[t.case_key.isin(man.case_key)].copy()
    C.require(len(t)==len(man)*8 and not t.duplicated(['case_key','d_mm','tau_mm']).any(),'Missing saved sensitivity')
    return t[['case_key','d_mm','tau_mm','Ctau','Cvalid','n_nodes','sup_n']]

def load_arrays(r,store=None):
    p=Path(r['source_pack']) if r['action']=='reuse' else store.path(r['case_key'])
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files if k!='_meta'}

def bind_only(man):

    blocked=[]
    def trace(frame,event,arg):
        if event=='call' and frame.f_code.co_name in ['render','decode','decode_eval','dot_decode','run_olines','synthesise_camera_image']:
            p=Path(frame.f_code.co_filename)
            if p.is_relative_to(C.PROJECT):
                blocked.append(str(p)+':'+frame.f_code.co_name)
                raise C.Stop('Numerical call forbidden during G0: '+blocked[-1])
    prior=sys.getprofile();sys.setprofile(trace)
    try:
        rows=[]
        for r in man[man.entry_subset].to_dict('records'):
            p=prepare(r)
            rows.append(dict(case_key=r['case_key'],forward_key=p['forward_key'],method=p['method'],raster=p['inputs']['raster_key']))
    finally:sys.setprofile(prior)
    C.require(len(rows)==24 and not blocked,'Real-entry input binding incomplete')
    return dict(input_bindings=rows,render_calls=0,decode_calls=0,forbidden_calls=blocked)
