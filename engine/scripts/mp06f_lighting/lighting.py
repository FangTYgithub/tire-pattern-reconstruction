# Prepare and execute uniform-attenuation and illumination-gradient cases.

from __future__ import annotations
import os,sys,time,json,argparse,itertools,traceback,sqlite3
from pathlib import Path
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
HOME=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HOME/'scripts/mp06d_blur'))
import numpy as np
import pandas as pd
import core as C
OUT=HOME/'outputs/mp06f_lighting'; C.OUT=OUT
import backend as B
import analysis as A
import run as ENGINE
BASE=C.PROJECT/'outputs/surface_revision'
OLD=HOME/'outputs/mp06d_blur'
PLAN=HOME/'outputs/mp06e_holdout/illumination_manifest_9504_PLAN_ONLY.csv'
PLAN_SHA='b2a87270b4a66bc2ec500bacfd40ed88f65e779ce1df13fdfb147ddff0a18160'
ENVS=['NOMINAL','UNIFORM','G05','GRAD_X','GRAD_Y','GRAD_Y_REV']
LABELS=['g = 1','g = 0,75','g = 0,50','Gradient x','Gradient y','Gradient −y']
SOURCES=[BASE/'model_prep05/results.csv',BASE/'model_prep06c/results_new.csv',OLD/'analysis/results.csv']
SENS_SOURCES=[C.PRIOR/'sensitivity_cases.csv.gz',OLD/'analysis/sensitivity.csv.gz']

class Store(C.Store):
    def __init__(self):super().__init__(OUT)
    def reserve(self,r,fid):
        ck=r['case_key'];b=C.canon(C.binding(r,fid))
        C.require(r['action']=='new','Reused case must never render')
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            C.require(con.execute('SELECT count(*) FROM cases').fetchone()[0]<6554,'Lighting cap reached')
            C.require(con.execute('SELECT 1 FROM cases WHERE ck=?',(ck,)).fetchone() is None,'Previously attempted case')
            con.execute('INSERT INTO cases VALUES(?,?,?,?,?)',(ck,b,'reserved',None,None))
        self.event(ck,'case','reserved')

def prepare_input(r):
    ch=B.init();p=ch.prep(C.rowdict(r));B.sandbox()
    expected='r_forward' if r['env']=='GRAD_Y_REV' else 'mp3_forward'
    C.require(p['route']==expected==r['forward_route'],'Lighting route mismatch')
    return p

def manifest_gate(m):
    keys={f'{cfg}|{sf}|b{b}|s{s}|r{rep}|{env}' for cfg,env,sf,b,s in itertools.product(C.DOTS+C.LINES,ENVS,C.SURFS,range(4),C.SIGMAS) for rep in ([0] if s==0 else [1,2])}
    C.require(len(m)==9504 and set(m.case_key)==keys and not m.forward_key.duplicated().any(),'Lighting matrix not closed')
    C.require(m.action.value_counts().to_dict()=={'new':6554,'reuse':2950},'Reuse/new count mismatch')
    C.require(m.execution_authorised.eq(True).all() and int(m.entry_subset.sum())==24,'Authorisation/entry mismatch')
    for r in m.to_dict('records'):
        r=C.rowdict(r);fam='point' if r['config'] in C.DOTS else 'line'
        seed=None if r['sigma']==0 else (9870000 if fam=='point' else 9880000)+100*C.SURFS.index(r['surface'])+10*r['phase_block']+r['rep']
        C.require(r['rep'] in ([0] if r['sigma']==0 else [1,2]) and seed==r['seed'],'Wrong noise identity')
        C.require(r['family']==fam and r['method']==('DF' if fam=='point' else 'P4') and r['psf_true_px']==1 and r['rays_per_pixel']==1,'Frozen method/sampling mismatch')
        C.require(0<r['peak']<=1 and 0<r['integral']<=11695.01,'Resource violation')
        field=json.loads(r['field_json']);e=r['env']
        if e in ENVS[:3]:want={'kind':'const','value':{'NOMINAL':1.,'UNIFORM':.75,'G05':.5}[e]}
        else:want=dict(kind={'GRAD_X':'gradient_x','GRAD_Y':'gradient_y','GRAD_Y_REV':'gradient_y_reversed'}[e],amp=.25,mean=.75,x_half_mm=46.08,y_half_mm=46.08)
        C.require(field==want,'Unexpected illumination field')
        C.require(not r['entry_subset'] or r['action']=='new','Reuse entered new subset')
    for fam,g in m.groupby('family'):
        cols=['phase_u','phase_v'] if fam=='point' else ['phase_frac']
        C.require(g.groupby('phase_block')[cols].nunique().eq(1).all().all(),'Physical phases inconsistent')

def source_data(m,hashes=True):
    tables=[C.csv(p) for p in SOURCES];tables[0]=tables[0][((tables[0].family=='point')&(tables[0].method=='DF'))|((tables[0].family=='line')&(tables[0].method=='P4'))]
    old=pd.concat(tables,ignore_index=True).drop_duplicates('case_key').set_index('case_key')
    rr=[]
    for r in m.to_dict('records'):
        r=C.rowdict(r)
        if hashes:C.require(C.sha(r['source_pack'])==r['source_pack_sha256'],'Changed reused source')
        q=C.rowdict(old.loc[r['case_key']]);C.require(q['forward_key']==r['forward_key'] and q['method']==r['method'],'Source identity mismatch')
        for k in C.IDENTITY:
            if k in q and q[k] is not None:C.require(q[k]==r[k],'Source identity mismatch '+k)
        q.update({k:r[k] for k in C.IDENTITY});rr.append(q)
    R=pd.DataFrame(rr);C.metric_gate(R,m)
    cols=['case_key','d_mm','tau_mm','Ctau','Cvalid','n_nodes','sup_n']
    T=pd.concat([C.csv(p)[cols] for p in SENS_SOURCES],ignore_index=True).drop_duplicates(['case_key','d_mm','tau_mm'])
    T=T[T.case_key.isin(m.case_key)].copy();A.sensitivity_gate(T,R)
    return R,T

def prepare():
    C.require(not (OUT/'freeze.json').exists(),'Already registered')
    C.install_guard();C.require(C.sha(PLAN)==PLAN_SHA,'Planned manifest changed')
    m=C.csv(PLAN);m['part']='MP06F_LIGHTING';m['execution_authorised']=True;m['entry_subset']=False
    entries=[('G05','L_w0.75_s8','M10_curv',0),('GRAD_Y_REV','D_r1_p8x3','B_curv',0),
      ('G05','D_r1_p8x6','M24_curv',60),('G05','L_w1.25_s8','M24_curv',40),
      ('UNIFORM','D_r1_p8x4','M10_curv',80),('UNIFORM','L_w0.75_s8','B_curv',20),
      ('GRAD_X','D_r1_p8x2.5','B_curv',40),('GRAD_X','L_w1.25_s8','M10_curv',40),
      ('GRAD_Y','D_r1_p8x3','M24_curv',20),('GRAD_Y','L_w0.75_s8','M24_curv',100),
      ('GRAD_Y_REV','D_r1_p8x6','M10_curv',80),('GRAD_Y_REV','L_w1.25_s8','B_curv',60),
      ('G05','D_r1.5_p8x6','B_curv',100)]
    for e,c,sf,s in entries:m.loc[(m.env==e)&(m.config==c)&(m.surface==sf)&(m.sigma==s)&(m.phase_block==1),'entry_subset']=True
    manifest_gate(m)
    for r in m[m.entry_subset].to_dict('records'):prepare_input(C.rowdict(r))
    R,T=source_data(m[m.action=='reuse'])
    C.save_csv(OUT/'manifest.csv',m)
    protocol=dict(date='2026-10-22',authorisation='User requests completion of necessary lighting experiment after necessity assessment; scientific findings are not a release gate',
      target=9504,reuse=2950,new=6554,entry_included=24,remaining=6530,policy=C.POLICY,
      purpose='Select feasible and near-optimal pattern size/spacing for registered illumination and noise conditions',
      environments=ENVS,PSF=1,noise=C.SIGMAS,surfaces=C.SURFS,candidates=C.DOTS+C.LINES,
      contrasts='same physical phase and noise seed: uniform vs nominal; each gradient vs uniform .75; reverse-y vs y',
      scientific_stop='No extra imaging for low coverage, no output, ties or adverse results; retain and report',
      technical_stop='Exception, input/hash drift, incomplete data, illegal metrics or duplicate pairing: stop submissions and preserve completed data',
      reporting='Use existing phase-mean criterion, not an added every-phase criterion; fixed alternatives and 2pp near-optimal band; show full admissible set',
      figures='selection maps, full curves, paired effects and fixed M24 block1 rep1 point clouds; no per-image PNGs',
      closure='This closed grid completes lighting module; no automatic reflection, intermediate gradients, interactions or holdout expansion')
    C.save(OUT/'protocol.json',protocol)
    old=C.load(OLD/'freeze.json');C.verify_files(old['files']);files=dict(old['files'])
    packs=set(m.source_pack.dropna().map(lambda p:str(Path(p).resolve())))
    for p in C.READS:
        if p not in packs and Path(p).is_file():files[p]=C.sha(p)
    for p in list(Path(__file__).parent.glob('*.py'))+SOURCES+SENS_SOURCES+[PLAN,OUT/'manifest.csv',OUT/'protocol.json']:
        files[str(p.resolve())]=C.sha(p)
    f=dict(files=files,freeze_id=C.digest(files),manifest_sha256=C.sha(OUT/'manifest.csv'),policy=C.POLICY,created=time.time())
    C.save(OUT/'freeze.json',f)
    C.save(OUT/'progress.json',dict(stage='registered',target=9504,reuse=2950,new=6554,completed_new=0))
    print(C.canon(dict(registered=True,entry=24,new=6554,reused=len(R),reused_sensitivity=len(T),freeze_id=f['freeze_id'])))

def verify():
    C.install_guard();f=C.load(OUT/'freeze.json');C.require(C.digest(f['files'])==f['freeze_id'] and f['policy']==C.POLICY,'Freeze/policy mismatch')
    C.verify_files(f['files']);C.require(C.sha(OUT/'manifest.csv')==f['manifest_sha256'],'Manifest drift')
    m=C.csv(OUT/'manifest.csv');manifest_gate(m);return f,m

def init_worker(f):
    C.install_guard();C.verify_files(f['files']);C.ALLOWED_PROJECT_READS={p for p in f['files'] if Path(p).is_relative_to(C.PROJECT)}

def work(r,fid):
    st=Store()
    try:
        t0=time.monotonic();ch=B.init();p=prepare_input(r);ck=r['case_key'];cached=st.load_image(r)
        if cached is None:
            st.event(ck,'render','started');img,diag=ch.render(p);B.sandbox();st.save_image(r,img,diag);st.event(ck,'render','finished')
        else:img,diag=cached;st.event(ck,'render','reused_checkpoint')
        st.event(ck,'decode','started');d=ch.decode_eval(p,img);B.sandbox();st.event(ck,'decode','finished')
        result=ch.result_row(p,img,diag,d,time.monotonic()-t0);result.update({k:r[k] for k in C.IDENTITY})
        sens=B.multi_eval(d['ev'],r['surface']);main=next(a for a in sens if a['d_mm']==6 and a['tau_mm']==.15)
        C.require(abs(main['Ctau']-result['glob_Ctau'])<1e-10 and abs(main['Cvalid']-result['glob_Cvalid'])<1e-10,'Main evaluation mismatch')
        method=r['method'];arrays={'image':img};arrays.update({method+'__raw__'+k:v for k,v in d['raw'].items()});arrays.update({method+'__ev__'+k:v for k,v in d['ev'].items()})
        meta=dict(case_key=ck,forward_key=r['forward_key'],method=method,result=result,sensitivity=sens,inputs=p['inputs'],input_key=p['input_key'],
          decode_meta=d['dmeta'],diag=diag,metrics=d['rec']['metrics'],full_identity=d['rec'].get('full_identity'),scientific=True,backend='frozen_DF_P4_lighting')
        st.commit(r,fid,meta,arrays);return dict(ok=True,case_key=ck)
    except BaseException as e:
        st.fail(r['case_key'],e);return dict(ok=False,case_key=r['case_key'],error=repr(e),trace=traceback.format_exc()[-2000:])

def execute(stage):
    f,m=verify();fid=f['freeze_id'];st=Store()
    if stage=='full':
        a=C.load(OUT/'entry_audit.json');C.require(a['ok'] and a['freeze_id']==fid,'Real entry not accepted')
    rows=m[m.entry_subset if stage=='entry' else ((m.action=='new')&~m.entry_subset)]
    todo=[C.rowdict(r) for r in rows.to_dict('records') if st.resume(C.rowdict(r),fid) is None]
    if not todo:print('Already complete; no render');return
    lock=C.writable(OUT/'run.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    stats={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in f['files']};count=[0];start=time.time()
    def before(r):
        for p,s in stats.items():
            v=Path(p).stat();C.require((v.st_size,v.st_mtime_ns)==s,'Frozen input changed: '+p)
        st.reserve(r,fid)
    def after(r,res):
        if not res['ok']:return
        st.resume(r,fid);count[0]+=1
        if count[0]%100==0:
            print(f'{stage} {count[0]}/{len(todo)}; {time.time()-start:.0f}s',flush=True)
            C.save(OUT/'progress.json',dict(stage=stage,completed_new=(24 if stage=='full' else 0)+count[0],new=6554,target=9504,elapsed=time.time()-start))
    try:
        ENGINE.bounded_dispatch(todo,work,before,after,lambda r:st.fail(r['case_key'],'cancelled after technical stop'),1 if stage=='entry' else 8,init_worker,(f,),(fid,))
        verify();C.save(OUT/f'{stage}_execution.json',dict(freeze_id=fid,computed=len(todo),resumed=len(rows)-len(todo),seconds=time.time()-start))
    finally:lock.unlink(missing_ok=True)

def collect(m,f,reuse=True):
    st=Store();rr=[];tt=[]
    if reuse and (m.action=='reuse').any():
        r,t=source_data(m[m.action=='reuse']);rr.append(r);tt.append(t)
    rows=[];ts=[]
    for r in m[m.action=='new'].to_dict('records'):
        meta=st.resume(C.rowdict(r),f['freeze_id']);C.require(meta is not None,'Missing new pack');rows.append(meta['result']);ts.extend([dict(case_key=r['case_key'],**v) for v in meta['sensitivity']])
    if rows:rr.append(pd.DataFrame(rows));tt.append(pd.DataFrame(ts))
    R=pd.concat(rr,ignore_index=True);T=pd.concat(tt,ignore_index=True);C.metric_gate(R,m);A.sensitivity_gate(T,R);return R,T

def audit_entry():
    f,m=verify();sub=m[m.entry_subset];R,T=collect(sub,f,False);st=Store();checks=[]
    sys.path.insert(0,str(OLD));from audit_execution import independent_surface
    for fam in ['point','line']:
        r=C.rowdict(sub[(sub.family==fam)&(sub.env=='G05')&(sub.sigma>0)].sort_values('sigma').iloc[0]);a=B.load_arrays(r,st);p=r['method']+'__ev__'
        val=independent_surface(a[p+'pts'][a[p+'valid']],r['surface']);q=T[T.case_key==r['case_key']]
        diff=max(max(abs(v.Ctau-val[v.d_mm,v.tau_mm][0]),abs(v.Cvalid-val[v.d_mm,v.tau_mm][1])) for v in q.itertuples())
        C.require(diff<1e-10,'Independent surface mismatch');checks.append(dict(case_key=r['case_key'],maxdiff=diff))
    g=R[R.sigma>0].groupby(['config','surface','env','phase_block','sigma'])
    C.require(len(g)==11 and g.rep.apply(lambda x:set(x)=={1,2}).all() and g.diag_noise_sha.nunique().eq(2).all(),'Entry noise pairs failed')
    for r in sub.to_dict('records'):
        meta=st.resume(C.rowdict(r),f['freeze_id']);d=meta['diag']
        C.require(d['rays_per_pixel']==1 and d['psf_true_px']==1,'Entry forward settings')
        C.require((d.get('noise_sha') is None)==(r['sigma']==0),'Zero-noise draw mismatch')
        if r['env'] in ['G05','UNIFORM']:C.require(abs(d['received_light_ratio']-{'G05':.5,'UNIFORM':.75}[r['env']])<1e-10,'Uniform gain mismatch')
        else:C.require(.5<=d['gain_min']<=d['gain_max']<=1,'Gradient out of bounds')
    C.save(OUT/'entry_audit.json',dict(ok=True,freeze_id=f['freeze_id'],cases=len(R),pairs=len(g),states=R.state.value_counts().to_dict(),independent_surface=checks))
    print(C.canon(dict(entry_passed=True,cases=len(R),states=R.state.value_counts().to_dict(),independent_surface=checks)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','entry','audit-entry','full']);args=p.parse_args()
    try:
        if args.stage=='prepare':prepare()
        elif args.stage=='audit-entry':audit_entry()
        else:execute(args.stage)
    except Exception:traceback.print_exc();sys.exit(2)
