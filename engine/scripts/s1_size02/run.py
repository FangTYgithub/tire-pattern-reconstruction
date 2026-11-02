# Run the additional 0.75 and 1.25 mm dot-radius cases with registered size-study conditions.

from __future__ import annotations
import os, sys, json, hashlib, time, argparse, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
HOME = Path(__file__).resolve().parents[2]
PROJECT = HOME/'research/project_Part1_v1.7'
SR = PROJECT/'scripts/surface_revision'
OUT = HOME/'outputs/s1_size02'
TASK = HOME/'protocols/documents/S1-SIZE-02_protocol.md'
OLD = PROJECT/'outputs/surface_revision/s12_resource01/results/views.csv'
DFOLD = PROJECT/'outputs/surface_revision/thesis_reinforce01/cases_pairs.csv'
SURFS = ['B_curv','M10_curv','M24_curv']
REGIMES = ['resource_cap_11695','fixed_peak_1']
METHODS = ['D0','DF']
READS=set(); _inited=False

def require(x,msg):
    if not bool(x): raise RuntimeError(msg)
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,Path): return str(x)
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def canon(x): return json.dumps(clean(x),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrsha(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csv(p): return pd.read_csv(p,float_precision='round_trip',low_memory=False)
def write(p,x):
    p=Path(p);require(p.resolve().is_relative_to(OUT),'write outside task');p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(canon(x),encoding='utf8');os.replace(t,p)
def savecsv(p,t):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');t.to_csv(q,index=False,encoding='utf-8-sig');os.replace(q,p)
def savepack(p,meta,a):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    with open(q,'wb') as f:np.savez_compressed(f,_meta=np.array(canon(meta)),**a)
    os.replace(q,p)
def readpack(p):
    with np.load(p,allow_pickle=False) as z:return json.loads(str(z['_meta'])),{k:z[k] for k in z.files if k!='_meta'}
def install_guard():
    def guard(event,args):
        if event=='open':
            raw,mode,flags=args
            if not isinstance(raw,(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(raw)).resolve()
            if not p.is_relative_to(PROJECT):return
            if (flags or 0)&(os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC):raise RuntimeError('Project read-only: '+str(p))
            if p.suffix=='.pyc':raise FileNotFoundError('source-only import')
            if p.is_file():READS.add(str(p))
        elif event in ('os.remove','os.rename','os.mkdir','os.rmdir','os.chmod','os.utime'):
            for raw in (args[:2] if event=='os.rename' else args[:1]):
                if isinstance(raw,(str,bytes,os.PathLike)):
                    p=Path(os.fsdecode(raw)).resolve()
                    if p.is_relative_to(PROJECT) and not(event=='os.mkdir' and p.is_dir()):raise RuntimeError('Project mutation denied: '+str(p))
    sys.addaudithook(guard)
def sandbox():
    for k,v in [('OUT_DIR','out'),('CASES_DIR','cases'),('LEDGER_DIR','ledger')]:os.environ['LINE01RB_'+k]=str(OUT/'scratch'/v)
    os.environ['LINE01RB_RENDER_CAP']='0'
def init():
    global _inited,K,S1,RP,DV,E,SC
    if _inited:return
    install_guard();sandbox()
    for p in ['s12_resource01','sections_adapter','s2_prepare','s345_ready','s3_code01','s3_code02']:
        sys.path.insert(0,str(SR/p))
    import s12_common as K
    import s1r_common as S1
    import rdy_platform as RP
    import s3c_dev as DV
    import s1r_eval as E
    import s2p_case as SC
    import m3_virtual_camera,ep_detect,ep_patterns,rdy_decode
    RP.bound();RP.dom_global()
    for sf in SURFS:RP.mesh(sf);RP.dom_local(sf)
    sandbox();_inited=True
def old():
    t=csv(OLD);t=t[(t.section=='S1')&(t.family=='point')&t.surface.isin(SURFS)&t.regime.isin(REGIMES)].copy()
    require(len(t)==864 and t.input_key.nunique()==720,'Original S1 views not complete');return t
def olddf():
    t=csv(DFOLD);t=t[(t.record_type=='A3_case')&(t.part=='S1')&(t.arm=='DF')].copy()
    require(len(t)==243 and not t.input_key.duplicated().any(),'DF reference not complete');return t
def event(kind,key,method='',status='started',detail=''):
    db=sqlite3.connect(OUT/'calls.sqlite',timeout=60)
    db.execute('CREATE TABLE IF NOT EXISTS calls(t REAL,kind TEXT,k TEXT,method TEXT,status TEXT,detail TEXT)')
    if status=='started':
        n=db.execute("SELECT count(*) FROM calls WHERE kind=? AND status='started'",(kind,)).fetchone()[0]
        cap={'render':588,'decode':1176,'regression':12}[kind]
        require(n<cap,'Call cap reached: '+kind)
    db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?)',(time.time(),kind,key,method,status,detail));db.commit();db.close()
def calls():
    if not (OUT/'calls.sqlite').exists():return {}
    with sqlite3.connect(OUT/'calls.sqlite') as db:return {f'{a}:{b}':n for a,b,n in db.execute('SELECT kind,status,count(*) FROM calls GROUP BY kind,status')}
def context(r):
    p=OUT/'rasters'/f"{r['raster_key']}.npz"
    if p.exists():
        with np.load(p) as z:R={k:z[k] for k in z.files}
    else:R=S1.load_raster(r['raster_key'])
    b=R['binary'].astype(np.float32);pk=float(r['peak_target'])
    return dict(binary=b,centres=R['centres'],pattern=b*np.float32(pk),peak=pk,size_mm=float(r['size_mm']),meta_full=None,row={'radius_mm':float(r['size_mm'])})
def numerical(r,img,method):
    ctx=context(r);sf=r['surface'];sid=r['platform_surface']
    if method=='D0':dec=RP.bound().call('decode','D0',img,ctx);af=None;nf=0
    else:
        D=DV.dot_decode(img,ctx['binary'],1,[r['size_mm']],'DF','binding');dec=DV.dot_dec(D,'single');af=D['cd']['a_floor'];nf=D['cd']['n_floor']
    m,a,info,node=E.evaluate_case(sid,'point',dec,ctx,RP.mesh(sf),RP.dom_local(sf),RP.dom_global(),False)
    raw,rawmeta=SC.raw_arrays('point',dec)

    from surface_eval_r3 import surface_eval_r3_dev
    dom=RP.dom_global();D=dict(x=dom['x'],y=dom['y'],z_ref=dom['z_'+sid],area=np.full(30947,.25))
    ev=surface_eval_r3_dev(a['pts'][a['valid']],None,D,[6.],[.15],return_fields=True,support_rule='exact')
    g=ev['subdomains']['global'];bd=g['by_d']['6.0'];f=ev['fields']
    require(abs(bd['C_tau']['0.15']-m['glob_Ctau'])<1e-12,'Node field Ctau differs')
    for k,v in raw.items():a['raw__'+k]=v
    for k,v in f.items():a['glob_node__'+k]=v
    a['glob_node__supported']=(f['simplex']>=0)&(f['support_measure']<=6.)
    a['glob_node__accurate']=a['glob_node__supported']&(np.abs(f['err'])<=.15)
    m.update(a_floor=af,n_floor=nf,det_otsu_threshold=rawmeta['thresh'])
    return m,a,dict(info=info,raw_meta=rawmeta)
def gate(m,a):
    require(m['glob_n_nodes']==30947,'Domain changed')
    for k in ['glob_Ctau','glob_Cvalid','glob_cov_interp','glob_recovery']:
        require(m[k] is not None and 0<=m[k]<=1,'Undefined coverage/recovery '+k)
    require(m['id_correct']+m['id_wrong']+m['id_unconf']==m['n_accepted'],'Identity denominator')
    require(m['glob_id_correct']+m['glob_id_wrong']+m['glob_id_unconf']==m['glob_n_valid'],'Global identity denominator')
    require(m['n_accepted']==int(a['accepted'].sum()) and m['n_valid']==int(a['valid'].sum()),'Point masks')
    if not m['n_valid']:
        require(m['glob_Ctau']==m['glob_Cvalid']==m['glob_recovery']==0,'No-output coverage must be zero')
    if not m['glob_n_valid']:require(m['glob_pt_rmse'] is None,'Undefined point RMSE must be null')
def dependency_files(extra=()):
    files=set(READS)|set(map(str,extra))
    for mod in list(sys.modules.values()):
        q=getattr(mod,'__file__',None)
        if q and Path(q).suffix=='.py' and Path(q).resolve().is_relative_to(PROJECT):files.add(str(Path(q).resolve()))
    files|={str(TASK),str(Path(__file__).resolve()),str(OUT/'manifest.csv'),str(OUT/'resource_table.csv')}
    return {p:sha(p) for p in sorted(files) if Path(p).is_file()}
def freezecheck():
    f=load(OUT/'freeze.json');bad=[p for p,h in f['files'].items() if not Path(p).is_file() or sha(p)!=h]
    require(not bad,'Freeze mismatch: '+str(bad[:4]));return f['bundle']
# Build the additional radius cases and verify their registered size-study inputs.
def prepare():
    require(not (OUT/'manifest.csv').exists(),'Manifest exists; do not overwrite registered plan')
    init();a=old();b=olddf();ras=[];checks=[];rows=[]
    for ph in range(3):
        label=f'sn_phase{ph}';ref=a[(a.size_mm==1)&(a.phase_index==ph)].iloc[0]
        base=S1.load_raster(ref.raster_key)['centres']
        for r in [.5,.75,1.,1.25,1.5]:
            R=S1.build_raster('point',r,label);require(np.array_equal(R['centres'],base),'Centre set changed')
            if r in [.5,1.,1.5]:
                oldkey=a[(a.size_mm==r)&(a.phase_index==ph)].raster_key.unique();require(len(oldkey)==1 and R['key']==oldkey[0],'Raster regression failed')
                S1.load_raster(oldkey[0])
            else:
                p=OUT/'rasters'/f"{R['key']}.npz";p.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(p,binary=R['binary'].astype(np.uint8),centres=R['centres'])
            checks.append(dict(radius=r,phase=label,centre_sha=arrsha(base),raster=R['key'],n=len(base),old_raster_exact=r in [.5,1.,1.5]))
            for regime in REGIMES:
                res=K.resource(R['binary'],regime);require(not K.resource_check(res,regime),'Resource check failed')
                ras.append(dict(size_mm=r,diameter_mm=2*r,phase_label=label,raster_key=R['key'],n_features_design=len(base),**res))
                ras[-1]['regime']=regime
                if r not in [.75,1.25]:continue
                for si,sf in enumerate(SURFS):
                    sid=dict(B_curv='B_curv',M10_curv='A_curv',M24_curv='C_curv')[sf]
                    for sig in [0,20,40,60,80,100]:
                        for rep in range(1 if sig==0 else 3):
                            seed=None if sig==0 else 9205000+10*ph+rep
                            ik=K.input_key(R['key'],res['peak_f32'],sid,float(sig),seed)
                            entrysig=[0,40,80][(si+int(r==1.25))%3]
                            rows.append(dict(input_key=ik,surface=sf,platform_surface=sid,family='point',section='S1',layout='S1_R8x4',size_mm=r,diameter_mm=2*r,regime=regime,phase_label=label,phase_index=ph,sigma=sig,rep=rep,noise_seed=seed,raster_key=R['key'],n_features_design=len(base),methods='D0;DF',entry_subset=ph==0 and rep==0 and sig==entrysig,**res))
    man=pd.DataFrame(rows);require(len(man)==576 and man.input_key.nunique()==576,'New image count');require(man.entry_subset.sum()==12,'Entry count')
    require(not set(man.input_key)&set(a.input_key),'Unexpected reusable image')
    savecsv(OUT/'manifest.csv',man);savecsv(OUT/'resource_table.csv',pd.DataFrame(ras))

    regression=[]
    for sf in SURFS:
        for sig in [40,80]:
            radius=.5 if sig==40 else 1.5
            u=a[(a.surface==sf)&(a.regime==REGIMES[0])&(a.size_mm==radius)&(a.sigma==sig)&(a.phase_index==0)&(a.rep==0)].iloc[0].to_dict()
            for method in METHODS:regression.append(dict(input_key=u['input_key'],method=method))
    examples=[]
    for reg in REGIMES:
        for sig in [0,40,80]:
            v=a[(a.surface=='M24_curv')&(a.size_mm==1)&(a.regime==reg)&(a.sigma==sig)].copy()
            v['dist']=(v.glob_Ctau-v.glob_Ctau.median()).abs();u=v.sort_values(['dist','phase_index','rep','input_key']).iloc[0]
            examples.append(dict(surface=u.surface,regime=reg,sigma=sig,phase_label=u.phase_label,rep=int(u.rep)))

    sources=[OLD,DFOLD,TASK]
    for _,u in a.drop_duplicates('input_key').iterrows():
        for fld in ['image_record','eval_record']:
            p=PROJECT/u[fld];sources.extend([p,p.with_suffix('.npz')])
    sources += list((PROJECT/'outputs/surface_revision/thesis_reinforce01/shards').glob('a3_df_*.npz'))
    for ik in b[b.reuse_class=='reuse_DF'].input_key:
        p=PROJECT/f'outputs/surface_revision/s3_code04/cache/C1/{ik}.npz';sources.append(p)
    source_hash={str(p):sha(p) for p in set(sources)}
    write(OUT/'source_hashes.json',source_hash)
    write(OUT/'checks/preparation.json',dict(raster_checks=checks,regression=regression,examples=examples,old_D0_views=864,old_unique_images=720,old_DF_key=243,new_images=576,new_method_results=1152,meshes=RP.surfaces(),domain=RP.contract(),imports=dependency_files()))
    write(OUT/'protocol.json',dict(task='S1-SIZE-02',time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),task_sha=sha(TASK),new_radii=[.75,1.25],surfaces=SURFS,sigmas=[0,20,40,60,80,100],regimes=REGIMES,phase_rule='original sn_phase0..2, original max-radius edge margin',noise_seed='9205000 + 10*phase_index + rep; null at sigma0',methods=METHODS,domain='original frozen Omega0, 30947 nodes',d=6,tau=.15,scope='no Gaussian, no local-domain analysis, no 5850',statistics='within-phase repetition means then three equal phase means; phase min/max, all phase means and defined denominators; no confidence interval',pairing='surface, regime, sigma, phase, rep; method/image identities explicit',representatives=examples,regression=regression,caps=dict(render=588,production_decode=1176,regression_decode=12),main_comparisons=['adjacent radii','new radii minus r1','fixed peak minus resource cap','DF minus D0 at resource cap sigma40/60/80'],missing_old_DF='No extension; full five-radius DF restricted to complete 243 old cases',status='prepared, not rendered'))
    print('PREPARED: 576 images, 12 entry, 864 reusable D0 views, 243 reusable DF records',flush=True)
def regress():
    init();p=load(OUT/'checks/preparation.json');a=old().drop_duplicates('input_key').set_index('input_key');b=olddf().set_index('input_key');report=[]
    done=load(OUT/'checks/regression.json') if (OUT/'checks/regression.json').exists() else []
    completed={(x['input_key'],x['method']) for x in done};report.extend(done)
    for item in p['regression']:
        ik,method=item['input_key'],item['method']
        if (ik,method) in completed:continue
        u=a.loc[ik].to_dict();u['input_key']=ik
        with np.load((PROJECT/u['image_record']).with_suffix('.npz')) as z:img=z['image']
        event('regression',ik,method)
        try:
            m,ar,extra=numerical(u,img,method);gate(m,ar)
            ref=load(PROJECT/u['eval_record'])['metrics'] if method=='D0' else b.loc[ik].to_dict()
            diffs={}
            for k in ['n_accepted','n_valid','glob_n_valid','glob_Ctau','glob_Cvalid','glob_recovery','glob_pt_rmse','glob_pt_p95','glob_surf_rmse','id_correct','id_wrong','id_unconf']:
                x,y=m[k],ref[k]
                if x is None or pd.isna(y):require(x is None and pd.isna(y),'Null regression '+k);continue
                d=abs(float(x)-float(y));require(d<=1e-12*max(1,abs(float(y))),'Metric regression '+k+': '+str(d));diffs[k]=d
            checked=[]
            if method=='D0':
                with np.load((PROJECT/u['eval_record']).with_suffix('.npz')) as z:
                    for k in z.files:
                        if k in ar:
                            require(np.array_equal(ar[k].astype(z[k].dtype),z[k],equal_nan=True),'D0 array regression '+k);checked.append(k)
            else:
                if b.loc[ik]['reuse_class']=='reuse_DF':
                    q=PROJECT/f'outputs/surface_revision/s3_code04/cache/C1/{ik}.npz'
                    with np.load(q) as z:
                        for k,src in [('pts','DF_pts'),('accepted','DF_acc'),('zr','DF_zr')]:
                            require(np.array_equal(ar[k].astype(z[src].dtype),z[src],equal_nan=True),'DF array regression '+k);checked.append(k)
                else:
                    found=False
                    for q in (PROJECT/'outputs/surface_revision/thesis_reinforce01/shards').glob('a3_df_*.npz'):
                        with np.load(q) as z:
                            if ik+'|metrics' not in z.files:continue
                            for k in ['pts','accepted','valid','zr','in_global','point_correct','point_wrong','point_unconf','feat_truth_dot','feat_confirmable','exp_recovered']:
                                src=ik+'|'+k;require(np.array_equal(ar[k].astype(z[src].dtype),z[src],equal_nan=True),'DF shard regression '+k);checked.append(k)
                            found=True;break
                    require(found,'Missing DF source')
            event('regression',ik,method,'finished');report.append(dict(**item,max_metric_diff=max(diffs.values()),arrays=checked,pass_exact=True));write(OUT/'checks/regression.json',report)
        except Exception as e:event('regression',ik,method,'failed',str(e));raise
    write(OUT/'checks/regression_dependencies.json',dependency_files());print('REGRESSION:',len(report),'passed',flush=True)
def freeze():
    init();require(len(load(OUT/'checks/regression.json'))==12,'Regression incomplete')
    files=load(OUT/'checks/preparation.json')['imports']|load(OUT/'checks/regression_dependencies.json')|dependency_files()
    sources=load(OUT/'source_hashes.json');require(all(sha(p)==h for p,h in sources.items()),'Source drift before freeze')
    for p,h in files.items():require(sha(p)==h,'Preparation dependency drift: '+p)
    files[str(OUT/'protocol.json')]=sha(OUT/'protocol.json');files[str(OUT/'source_hashes.json')]=sha(OUT/'source_hashes.json')
    write(OUT/'freeze.json',dict(bundle=digest(files),files=files,time=time.strftime('%Y-%m-%dT%H:%M:%S%z')));print('FROZEN',len(files),'dependencies',flush=True)
def run(mode):
    init();bundle=freezecheck();m=csv(OUT/'manifest.csv');m=m[m.entry_subset] if mode=='entry' else m
    if mode=='full':require(load(OUT/'checks/entry.json')['passed'],'Entry gate not passed')
    start=time.monotonic();n=0
    for r in m.to_dict('records'):
        ik=r['input_key'];p=OUT/'cases'/f'{ik}.npz'
        if p.exists():
            meta,a=readpack(p);require(meta['bundle']==bundle and meta['row']==clean(r),'Stale checkpoint')
        else:meta=dict(bundle=bundle,row=clean(r),complete=False,methods={});a={}
        if meta['complete']:
            require(set(meta['methods'])==set(METHODS) and 'image' in a,'Incomplete saved pack');n+=1;continue
        if 'image' not in a:
            ctx=context(r);event('render',ik)
            try:
                im=RP.bound().call('render',r['platform_surface'],ctx['pattern'],ctx['peak'],r['sigma'],int(r['noise_seed']) if r['sigma'] else 0,1.)
                a['image']=im['image'];meta['imaging']=im['meta'];require(a['image'].dtype==np.uint8 and a['image'].shape==(512,512),'Image contract')
                savepack(p,meta,a);event('render',ik,status='finished')
            except Exception as e:event('render',ik,status='failed',detail=str(e));raise
        for method in METHODS:
            if method in meta['methods']:continue
            event('decode',ik,method)
            try:
                t=time.monotonic();v,ar,info=numerical(r,a['image'],method);gate(v,ar)
                meta['methods'][method]=dict(metrics=clean(v),diagnostics=clean(info),seconds=time.monotonic()-t,decode_key=digest([ik,method,bundle]))
                a.update({method+'__'+k:x for k,x in ar.items()});savepack(p,meta,a);event('decode',ik,method,'finished')
            except Exception as e:event('decode',ik,method,'failed',str(e));raise
        meta['complete']=True;savepack(p,meta,a);n+=1
        if n%12==0 or n==len(m):
            write(OUT/'progress.json',dict(stage=mode,completed_in_stage=n,total_in_stage=len(m),elapsed_s=time.monotonic()-start,calls=calls()))
            print(mode,n,'/',len(m),'elapsed',round(time.monotonic()-start,1),flush=True)
    freezecheck()
    if mode=='entry':write(OUT/'checks/entry.json',dict(passed=True,images=12,methods=24,calls=calls(),elapsed_s=time.monotonic()-start,scope='numerical and persistence checks; visual inspection by operator before full'))
    else:write(OUT/'checks/run_complete.json',dict(passed=True,images=len(m),calls=calls(),elapsed_s=time.monotonic()-start))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','regress','freeze','entry','full']);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        {'prepare':prepare,'regress':regress,'freeze':freeze,'entry':lambda:run('entry'),'full':lambda:run('full')}[args.command]()
    except Exception as e:
        write(OUT/'technical_stop.json',dict(command=args.command,error=repr(e),calls=calls(),time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        raise
