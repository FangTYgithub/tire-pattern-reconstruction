# Check blur-result integrity and task scheduling with synthetic and stored cases.

from __future__ import annotations
import copy,json,sys,subprocess,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import core as C
import analysis as A

def _drill_job(r):
    if r.get('error'):raise RuntimeError('injected child crash')
    return dict(ok=True,case_key=r['case_key'])

def synthetic(man):

    r=man[C.IDENTITY].copy();r['state']='ok';r['glob_n_valid']=20;r['n_accepted']=20
    r['c']=20;r['w']=0;r['u']=0;r['glob_pt_rmse']=.1;r['glob_pt_p95']=.18
    r.loc[r.index[0],['state','glob_n_valid','n_accepted','c','glob_pt_rmse','glob_pt_p95']]=['no_output',0,0,0,np.nan,np.nan]
    ts=[]
    for d in C.DS:
        for tau in C.TAUS:
            s=r[['case_key']].copy();s['d_mm']=d;s['tau_mm']=tau;s['n_nodes']=30947
            s['sup_n']=24000 if d==5 else 28000;s['Cvalid']=s.sup_n/30947
            noise=r.sigma.to_numpy()/100
            s['Ctau']=np.minimum(s.Cvalid,(.38+tau+(d-5)*.04-.3*noise))
            s.loc[s.index[0],['sup_n','Cvalid','Ctau']]=0
            ts.append(s)
    t=pd.concat(ts,ignore_index=True);main=t[(t.d_mm==6)&(t.tau_mm==.15)].set_index('case_key')
    r['glob_Ctau']=r.case_key.map(main.Ctau);r['glob_Cvalid']=r.case_key.map(main.Cvalid)
    return r,t

def saved_meta(r):
    import backend as B
    record=B.source_records(pd.DataFrame([r]),False).iloc[0].to_dict()
    arrays=B.load_arrays(r)

    arrays={k:v for k,v in arrays.items() if k=='image' or k.startswith(r['method']+'__')}
    sens=B.source_sensitivity(pd.DataFrame([r])).drop(columns='case_key').to_dict('records')
    return dict(result=record,sensitivity=sens,scientific=False,backend='SAVED_DATA_DRILL'),arrays

# Test result integrity, pairing and bounded scheduling with controlled cases.
def run_checks(man,fid):
    checks={}
    def passed(name,fn):
        fn();checks[name]='PASS'
    def rejected(name,fn):
        try:fn()
        except (C.Stop,ValueError,pd.errors.MergeError) as e:
            checks[name]={'status':'PASS_REJECTED','reason':str(e)[:220]};return
        raise C.Stop('Counterexample unexpectedly passed: '+name)
    records,sens=synthetic(man)
    passed('full_7920_identity_and_metric_gate',lambda:C.metric_gate(records,man))
    passed('full_63360_threshold_gate',lambda:A.sensitivity_gate(sens,records))
    rejected('missing_case',lambda:C.metric_gate(records.iloc[1:],man))
    rejected('duplicate_case',lambda:C.metric_gate(pd.concat([records,records.iloc[:1]]),man))
    rejected('missing_metric_column',lambda:C.metric_gate(records.drop(columns='glob_Ctau'),man))
    def bad_result(column,value,index=1):
        q=records.copy();q.loc[q.index[index],column]=value;return q
    nonzero=int(np.where(records.sigma.to_numpy()>0)[0][0])
    rejected('wrong_repetition',lambda:C.metric_gate(bad_result('rep',0,nonzero),man))
    rejected('wrong_noise_seed',lambda:C.metric_gate(bad_result('seed',-1,nonzero),man))
    rejected('wrong_method',lambda:C.metric_gate(bad_result('method','DS'),man))
    rejected('wrong_forward_key',lambda:C.metric_gate(bad_result('forward_key','other'),man))
    rejected('valid_points_missing_rmse',lambda:C.metric_gate(bad_result('glob_pt_rmse',np.nan),man))
    rejected('no_output_positive_coverage',lambda:C.metric_gate(bad_result('glob_Ctau',.5,0),man))
    rejected('unconfirmed_denominator_mismatch',lambda:C.metric_gate(bad_result('u',1),man))
    rejected('missing_threshold_row',lambda:A.sensitivity_gate(sens.iloc[1:],records))
    rejected('wrong_repetition_in_manifest',lambda:C.check_manifest(man.assign(rep=0)))
    rejected('missing_manifest_case',lambda:C.check_manifest(man.iloc[1:]))
    def forbidden_write():
        with (C.PROJECT/'__mp06d_forbidden_write.txt').open('wb') as f:f.write(b'forbidden')
    rejected('scientific_project_write_guard',forbidden_write)
    C.require(not (C.PROJECT/'__mp06d_forbidden_write.txt').exists(),'Write guard allowed project mutation')

    q=records[(records.config.isin(['L_w0.5_s8','L_w1.5_s8']))&(records.env=='BLUR2')&(records.surface=='M10_curv')&(records.phase_block==1)&(records.sigma==40)].copy()
    q.loc[q.config=='L_w0.5_s8','glob_Ctau']=[.1,.2]
    q.loc[q.config=='L_w1.5_s8','glob_Ctau']=[.3,.4]
    e=A.paired(q,'L_w0.5_s8','L_w1.5_s8')
    C.require(len(e)==2 and np.allclose(e.delta_Ctau,[.2,.2]),'Two-repeat counterexample not fixed')
    checks['two_noise_repeats_are_two_pairs_not_four']=dict(status='PASS',pairs=2,deltas=e.delta_Ctau.tolist())
    rejected('duplicate_pair_slot',lambda:A.paired(pd.concat([q,q.iloc[:1]]),'L_w0.5_s8','L_w1.5_s8'))
    qbad=q.copy();qbad.loc[qbad.config=='L_w1.5_s8','seed']+=1
    rejected('paired_seed_mismatch',lambda:A.paired(qbad,'L_w0.5_s8','L_w1.5_s8'))

    import backend as B
    reuse=man[man.action=='reuse'];real=B.source_records(reuse,False);rt=B.source_sensitivity(reuse)
    passed('3224_saved_results_main_tau_consistent',lambda:A.sensitivity_gate(rt,real))
    with tempfile.TemporaryDirectory(prefix='_g0_',dir=C.OUT) as td:
        td=Path(td)

        selected=[]
        for batch,fam in [('MP05','point'),('MP05','line'),('MP06C','point'),('MP06C','line')]:
            r=C.rowdict(reuse[(reuse.source_batch==batch)&(reuse.family==fam)].iloc[0])
            meta,arrays=saved_meta(r);C.pack_gate(meta,arrays,r);selected.append((r,meta,arrays))
        checks['saved_pack_array_summary_checks']=dict(status='PASS',packs=4,methods=['DF','P4'],sources=['MP05','MP06C'])
        r,meta,arrays=selected[0];st=C.Store(td/'store')
        st.reserve(r,fid);st.commit(r,fid,meta,arrays)
        passed('resume_complete_without_recompute',lambda:C.require(st.resume(r,fid)['scientific'] is False,'Wrong drill metadata'))
        rejected('resume_old_method_or_freeze',lambda:st.resume(r,'other-freeze'))
        r2,meta2,arrays2=selected[1];st.reserve(r2,fid)
        try:st.commit(r2,fid,meta2,arrays2,interrupt_after_pack=True)
        except InterruptedError:pass
        C.require(st.resume(r2,fid) is not None,'Atomic file could not recover')
        checks['recover_interruption_between_file_and_index']='PASS'
        r3,meta3,arrays3=selected[2];st.reserve(r3,fid)
        rejected('incomplete_attempt_does_not_rerender',lambda:st.resume(r3,fid))
        r4,meta4,arrays4=selected[3];st.reserve(r4,fid);st.commit(r4,fid,meta4,arrays4)
        with st.path(r4['case_key']).open('ab') as f:f.write(b'corruption')
        rejected('corrupt_result_rejected',lambda:st.resume(r4,fid))
        rejected('duplicate_attempt_refused',lambda:st.reserve(r,fid))

        ar=dict(arrays);ar[r['method']+'__ev__accepted']=np.zeros_like(ar[r['method']+'__ev__accepted'])
        rejected('wrong_accepted_mask',lambda:C.pack_gate(meta,ar,r))

        f=td/'frozen_sample';f.write_text('a');h=C.sha(f);f.write_text('b')
        rejected('frozen_dependency_drift',lambda:C.verify_files({str(f):h}))
        result=A.publish(records,sens,man,td/'analysis',drill=True)
        checks['full_matrix_analysis_to_figures']=result|dict(status='PASS',scientific=False)

        byfam={r['family']:arrays for r,meta,arrays in selected}
        clouds=A.cloud_figures(man,td/'analysis/figures',lambda r:byfam[r['family']],drill=True)
        checks['cloud_figure_interface_saved_data']=dict(status='PASS',figures=2,scientific=False)

        from PIL import Image,ImageOps,ImageDraw
        ims=[]
        for p in result['figures']+clouds:
            with Image.open(p) as im:ims.append(ImageOps.contain(im.convert('RGB'),(1500,800)))
        board=Image.new('RGB',(1500,sum(i.height for i in ims)+60),'white');y=60
        ImageDraw.Draw(board).text((20,10),'G0 SYNTHETIC DATA / LAYOUT CHECK ONLY / NOT SCIENTIFIC RESULTS',fill='red')
        for im in ims:board.paste(im,((1500-im.width)//2,y));y+=im.height
        board.save(C.writable(C.OUT/'drill_preview.png'))
        checks['full_matrix_analysis_to_figures']['figures']=['drill_preview.png (synthetic, not scientific)']

    cp=subprocess.run([sys.executable,'-X','utf8','-B',str(C.HERE/'run.py'),'entry'],capture_output=True,text=True,encoding='utf-8')
    C.require(cp.returncode!=0 and ('not authorised' in cp.stderr or 'preparation.json' in cp.stderr or 'Current G0' in cp.stderr),'CLI production guard failed: '+cp.stderr[-400:])
    checks['unauthorised_cli_nonzero_exit']=dict(status='PASS',returncode=cp.returncode)
    cp=subprocess.run([sys.executable,'-X','utf8','-B',str(C.HERE/'run.py'),'invalid_stage'],capture_output=True,text=True,encoding='utf-8')
    C.require(cp.returncode!=0,'Entry parser crash masked as success')
    checks['entry_error_nonzero_exit']=dict(status='PASS',returncode=cp.returncode)
    cp=subprocess.run([sys.executable,'-X','utf8','-B',str(C.HERE/'run.py'),'inputs'],capture_output=True,text=True,encoding='utf-8')
    C.require(cp.returncode==0 and '"input_cases": 24' in cp.stdout,'Fresh-process frozen real-input binding failed: '+cp.stderr[-1800:])
    checks['fresh_process_24_inputs_with_enforced_freeze']=dict(status='PASS',render_count=0,decode_count=0)
    import run as R
    submitted=[];completed=[]
    n=R.bounded_dispatch([dict(case_key=str(i)) for i in range(7)],_drill_job,lambda r:submitted.append(r['case_key']),lambda r,o:completed.append(o),lambda r:None,2)
    C.require(n==7 and len(completed)==7 and all(x['ok'] for x in completed),'Multiprocess completion failed')
    checks['bounded_multiprocess_complete_and_joined']='PASS'
    submitted=[]
    rejected('worker_exception_stops_scheduling',lambda:R.bounded_dispatch([dict(case_key='0',error=True),dict(case_key='1')],_drill_job,lambda r:submitted.append(r['case_key']),lambda r,o:None,lambda r:None,1))
    C.require(submitted==['0'],'Work scheduled after known failure')
    checks['no_submission_after_first_failure']='PASS'
    C.require(not (C.OUT/'store.sqlite').exists(),'Scientific store created during G0')
    C.require(not (C.OUT/'data').exists(),'Scientific packs created during G0')
    checks['no_production_store_or_camera_data']='PASS'
    return checks
