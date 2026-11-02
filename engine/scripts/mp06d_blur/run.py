# Prepare, execute and analyze the complete blur experiment.

from __future__ import annotations
import os,sys,json,time,traceback,argparse,importlib.metadata
from pathlib import Path
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
import pandas as pd
import core as C

def freeze_files():
    prior=C.load(C.PRIOR/'checks.json')
    files={str((C.PROJECT/p).resolve()):h for p,h in prior['inputs'].items()}
    files.update({str((C.PROJECT/p).resolve()):h for p,h in prior['manifest']['prospective_input_dependencies'].items()})
    C.verify_files(files)


    paths=set()
    for folder in ['src','scripts']:
        paths.update((C.PROJECT/folder).rglob('*.py'))
    paths.update(C.PROJECT.glob('*.py'))
    paths.update(C.HERE.glob('*.py'))
    source_packs=set(C.csv(C.MANIFEST).source_pack.dropna().map(lambda p:str(Path(p).resolve())))
    paths.update(Path(p) for p in C.READS if p not in source_packs and Path(p).is_file())
    paths.update([C.MANIFEST,C.PRIOR/'checks.json',C.PRIOR/'sensitivity_cases.csv.gz',C.PRIOR/'sensitivity_choices.csv',C.DOMAIN])
    for r in ['outputs/line01r/night01r2/config/snapshot_line_encoding_dev_v1.json','outputs/line01r/night01r3/config/r3_config.json']:
        p=C.PROJECT/r
        if p.exists():paths.add(p)
    files.update({str(p.resolve()):C.sha(p) for p in paths})
    return files

def prepare():
    C.install_guard()
    import backend as B
    man=C.manifest();real=B.bind_only(man)

    sources=B.source_records(man[man.action=='reuse'],True)
    C.require(len(sources)==3224,'Wrong reuse gate count')
    files=freeze_files();fid=C.digest(files)
    freeze=dict(freeze_id=fid,files=files,manifest_sha256=C.EXPECTED_MANIFEST,policy=C.POLICY,
                packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','matplotlib']},python=sys.version,
                scope='static Python closure + runtime read configs; unknown runtime project reads STOP',created=time.time())
    C.save(C.OUT/'freeze.json',freeze)
    if not (C.OUT/'release.json').exists():
        C.save(C.OUT/'release.json',dict(freeze_id=fid,entry_authorised=False,full_authorised=False,
                 note='No imaging/decoding authorised. User must explicitly approve subsequent real entry.'))
    else:
        rel=C.load(C.OUT/'release.json')
        C.require(not rel.get('entry_authorised') and not rel.get('full_authorised'),'Cannot reprepare a released run')
        C.save(C.OUT/'release.json',rel|{'freeze_id':fid})
    import tests as T
    checks=T.run_checks(man,fid)
    C.verify_freeze()
    B.source_records(man[man.action=='reuse'],True)
    old=C.load(C.PRIOR/'checks.json')
    wording=dict(holdout_not_full_grid=True,wrong_at_sigma40_retained=True,preferred_score_not_best=True,
                 tau_observed_crossing_interval_stable=True,d_support_changes_dot_preference=True,
                 basis='Current corrected report and prior accepted result tables; no numerical rerun')

    ch=C.csv(C.PRIOR/'sensitivity_choices.csv')
    s=ch[(ch.family=='point')&(ch.tau_mm==.15)&(ch['floor']==.5)]
    C.require(set(s[(s.d_mm==6)&s.preferred.notna()].preferred)=={'D_r1_p8x6'},'Unexpected d6 recommendation')
    C.require(set(s[(s.d_mm==5)&s.preferred.notna()].preferred)=={'D_r1_p8x4'},'Unexpected d5 recommendation')
    record=dict(status='G0_READY_NO_IMAGING_AUTHORISATION',freeze_id=fid,target=7920,reuse=3224,new=4696,entry=24,remaining=4672,
                render_count=0,decode_count=0,real_backend_verified=False,real_input_preparation=real,
                source_records_passed=len(sources),freeze_files=len(files),checks=checks,wording=wording,
                prior_main_recalculation=old['reevaluation']['main_regression_maxdiff'],
                limitations=['No real image or decoder executed. G1 remains mandatory.',
                'Same-phase paired expansion, not new-phase validation.',
                'Any previously unseen runtime project dependency is a technical stop before use.'])
    C.save(C.OUT/'preparation.json',record)
    print(json.dumps({k:record[k] for k in ['status','target','reuse','new','entry','freeze_files','render_count','decode_count']},ensure_ascii=False))
    return record

def release(stage,freeze):
    p=C.OUT/'release.json';C.require(p.exists(),'No production release record')
    r=C.load(p)
    C.require(r.get('freeze_id')==freeze['freeze_id'],'Release bound to another freeze')
    g=C.load(C.OUT/'preparation.json')
    C.require(g.get('status')=='G0_READY_NO_IMAGING_AUTHORISATION' and g.get('freeze_id')==freeze['freeze_id'],'Current G0 not passed')
    C.require(r.get('entry_authorised') is True,'User has not authorised real-entry imaging')
    if stage in ['full','analysis']:
        C.require(r.get('full_authorised') is True,'User has not authorised remaining batch')
        e=C.load(C.OUT/'entry_check.json')
        C.require(e.get('ok') is True and e.get('freeze_id')==freeze['freeze_id'],'Real entry not complete')
        C.require(r.get('accepted_entry_sha256')==C.sha(C.OUT/'entry_check.json'),'Real entry lacks acceptance bound to current check')

def worker_init(freeze):
    C.install_guard();C.verify_files(freeze['files'])
    C.ALLOWED_PROJECT_READS={p for p in freeze['files'] if Path(p).is_relative_to(C.PROJECT)}

def worker(r,fid,root,permit):
    from backend import real_compute
    store=C.Store(root)
    try:
        meta,arrays=real_compute(r,store,permit)
        store.commit(r,fid,meta,arrays)
        return dict(ok=True,case_key=r['case_key'])
    except BaseException as e:
        store.fail(r['case_key'],e)
        return dict(ok=False,case_key=r['case_key'],error=repr(e),trace=traceback.format_exc()[-1800:])

def bounded_dispatch(rows,job,before,after,cancelled,workers,initializer=None,initargs=(),extra=()):

    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    pending={};it=iter(rows);done=0;failed=None
    pool=ProcessPoolExecutor(max_workers=workers,initializer=initializer,initargs=initargs)
    def submit():
        try:r=next(it)
        except StopIteration:return False
        before(r);pending[pool.submit(job,r,*extra)]=r;return True
    try:
        for _ in range(workers):submit()
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for f in ready:
                r=pending.pop(f)
                try:o=f.result()
                except BaseException as e:o=dict(ok=False,case_key=r['case_key'],error=repr(e))
                after(r,o)
                if not o['ok']:failed=o;break
                done+=1
            if failed:break
            while len(pending)<workers and submit():pass
        if failed:
            for f,r in pending.items():
                if f.cancel():cancelled(r)
            raise C.Stop('Worker failure, no further submissions: '+str(failed))
        return done
    finally:
        pool.shutdown(wait=True,cancel_futures=True)

# Run pending cases and record completed outputs.
def execute(stage,workers):
    C.install_guard();freeze=C.verify_freeze();release(stage,freeze);man=C.manifest();fid=freeze['freeze_id']
    import backend as B
    B.source_records(man[man.action=='reuse'],True)
    C.require(1<=workers<=8,'Workers outside 1..8')
    lock=C.writable(C.OUT/'run.lock')

    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        store=C.Store();rows=man[man.entry_subset if stage=='entry' else ((man.action=='new')&~man.entry_subset)]
        todo=[];reused=0
        for r in rows.to_dict('records'):
            r=C.rowdict(r)
            if store.resume(r,fid) is not None:reused+=1
            else:todo.append(r)

        stats={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in freeze['files']}
        limit=1 if stage=='entry' else workers
        def before(r):
            for p,st in stats.items():
                q=Path(p).stat();C.require((q.st_size,q.st_mtime_ns)==st,'Frozen input changed during execution: '+p)
            store.reserve(r,fid)
        counter=[0]
        def after(r,o):
            if not o['ok']:store.fail(r['case_key'],o.get('error'));return
            store.resume(r,fid);counter[0]+=1
            if counter[0]%100==0:print(f'{stage}: {counter[0]}/{len(todo)} completed',flush=True)
        done=bounded_dispatch(todo,worker,before,after,lambda r:store.fail(r['case_key'],'Cancelled after technical failure'),limit,
                              worker_init,(freeze,),(fid,C.OUT,'explicit_user_authorised_production'))
        C.verify_freeze()
        if stage=='entry':
            meta=[store.resume(C.rowdict(r),fid) for r in rows.to_dict('records')]
            R=pd.DataFrame([m['result'] for m in meta]);C.metric_gate(R,rows)
            for m in meta:
                r=m['result'];diag=m['diag'];C.require(diag.get('rays_per_pixel')==1,'Entry ray sampling mismatch')
                if r['sigma']==0:C.require(diag.get('noise_sha') is None,'sigma0 has noise draw')
            nr=R[R.sigma>0].groupby(['config','env','surface','phase_block','sigma']).diag_noise_sha
            C.require(nr.nunique().eq(2).all(),'Entry repetitions do not have distinct noise realisations')
            p=dict(ok=True,freeze_id=fid,cases=len(R),states=R.state.value_counts().to_dict(),source='real_backend',new_computed=done,resumed=reused,
                   packs_digest=C.digest({r['case_key']:C.sha(store.path(r['case_key'])) for r in rows.to_dict('records')}))
            C.save(C.OUT/'entry_check.json',p)
            print('24-case real entry complete; STOP awaiting separate acceptance and continuation authorisation.')
        else:print(f'Remaining batch complete: {done} computed, {reused} resumed. Run analysis after integrity check.')
    finally:
        lock.unlink(missing_ok=True)

def analyse():
    import backend as B
    import analysis as A
    C.install_guard();fz=C.verify_freeze();release('analysis',fz);m=C.manifest();store=C.Store()
    existing=B.source_records(m[m.action=='reuse']);sens=[B.source_sensitivity(m[m.action=='reuse'])];res=[]
    for r in m[m.action=='new'].to_dict('records'):
        meta=store.resume(C.rowdict(r),fz['freeze_id']);C.require(meta is not None,'New case missing: '+r['case_key'])
        C.require(meta.get('scientific') is True,'Drill material cannot enter production analysis')
        res.append(meta['result']);sens.append(pd.DataFrame(meta['sensitivity']).assign(case_key=r['case_key']))
    R=pd.concat([existing,pd.DataFrame(res)],ignore_index=True);T=pd.concat(sens,ignore_index=True)
    out=A.publish(R,T,m,C.OUT/'analysis')
    out['cloud_figures']=A.cloud_figures(m,C.OUT/'analysis/figures',lambda r:B.load_arrays(r,store))
    C.verify_freeze();print(json.dumps(out,ensure_ascii=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','check','inputs','entry','full','analysis'],nargs='?',default='prepare');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='check':
        f=C.verify_freeze();C.manifest();print('Freeze and manifest OK; no numerical work. '+f['freeze_id'])
    elif a.stage=='inputs':
        f=C.verify_freeze();worker_init(f)
        import backend as B
        r=B.bind_only(C.manifest());print(json.dumps(dict(ok=True,input_cases=len(r['input_bindings']),render_count=0,decode_count=0)))
    elif a.stage in ['entry','full']:execute(a.stage,a.workers)
    else:analyse()

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('INTERRUPTED; completed packs retained, incomplete attempts need inspection.',file=sys.stderr);sys.exit(130)
    except Exception as e:
        print(type(e).__name__+': '+str(e),file=sys.stderr);traceback.print_exc();sys.exit(2)
