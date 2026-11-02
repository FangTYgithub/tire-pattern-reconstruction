# Register and run a finite new-phase verification of blur recommendations.

from __future__ import annotations
import os,sys,time,json,argparse,itertools,traceback
from pathlib import Path
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
HOME=Path(__file__).resolve().parents[2]
LIB=HOME/'scripts/mp06d_blur'
sys.path.insert(0,str(LIB))
import numpy as np
import pandas as pd
import core as C

OUT=HOME/'outputs/mp06e_holdout'
C.OUT=OUT
import backend as B
import run as ENGINE
BASE=C.PROJECT/'outputs/surface_revision'
OLD=HOME/'outputs/mp06d_blur'
SCENARIOS=[
 dict(id='H1',family='line',env='BLUR2_5',sigma=20,d=6.,reference='L_w1_s8',recommended='L_w1.25_s8',configs=['L_w1_s8','L_w1.25_s8','L_w1.5_s8'],prediction='recommended remains admissible and within 2 pp of tested alternatives; no unique-best claim'),
 dict(id='H2',family='line',env='BLUR3',sigma=0,d=6.,reference='L_w1.5_s8',recommended='L_w1.25_s8',configs=['L_w1.25_s8','L_w1.5_s8'],prediction='both registered wide-line alternatives remain admissible; no superiority claim'),
 dict(id='H3',family='point',env='BLUR2_5',sigma=40,d=6.,reference='D_r1_p8x4',recommended='D_r1_p8x6',configs=['D_r1_p8x4','D_r1_p8x6'],prediction='8x6 remains admissible near coverage floor; paired improvement over 8x4'),
 dict(id='H4',family='point',env='NOMINAL',sigma=60,d=6.,reference='D_r1_p8x4',recommended='D_r1_p8x6',configs=['D_r1_p8x4','D_r1_p8x6'],prediction='8x6 remains admissible near noise boundary; paired improvement over 8x4'),
 dict(id='H5',family='point',env='BLUR1_5',sigma=40,d=5.,reference='D_r1_p8x6',recommended='D_r1_p8x4',configs=['D_r1_p8x4','D_r1_p8x6'],prediction='stricter support d5 favors 8x4; evaluate d6 on the same images without a second render'),
 dict(id='H6',family='line',env='NOMINAL',sigma=40,d=6.,reference='L_w0.75_s8',recommended='L_w0.5_s8',configs=['L_w0.5_s8','L_w0.75_s8'],prediction='0.5 remains admissible near noise boundary; paired improvement over 0.75'),
]
SEEDS={'point':21910000,'line':21920000}

def setup():
 C.install_guard();B.init()
 import r_manifest as RM
 import r_inputs as RI
 import mp4_formal as MF
 import tr3_b2 as REG
 return RM,RI,MF,REG

def current_files():
 old=C.load(OLD/'freeze.json');C.verify_files(old['files'])
 files=dict(old['files'])
 for p in C.READS:
  if Path(p).is_file():files[p]=C.sha(p)
 for p in Path(__file__).parent.glob('*.py'):files[str(p.resolve())]=C.sha(p)
 for p in [OUT/'manifest.csv',OUT/'protocol.json',OLD/'analysis/choices.csv',OLD/'analysis/surfaces.csv']:
  files[str(p)]=C.sha(p)
 return files

def registry(RM,REG):
 reg=REG.registry();ps=reg['positions'];ls=reg['phases']
 ps=ps[ps.lattice.astype(str).str.startswith('R')|ps.lattice.astype(str).isin(['unknown','nan','generic','None'])]
 ls=ls[(ls.pitch.isna()|pd.to_numeric(ls.pitch,errors='coerce').sub(8).abs().lt(1e-6))&(ls.orientation.isna()|pd.to_numeric(ls.orientation,errors='coerce').abs().lt(1e-6))]
 points=[(float(a),float(b)) for a,b in zip(ps.u,ps.v)]
 lines=list(pd.to_numeric(ls.phase,errors='coerce').dropna().astype(float))
 seeds=set(reg['seeds']);rasters=set(reg['raster_keys']);sources=[]

 files=[BASE/'model_prep05/formal_manifest.csv',BASE/'model_prep06c/holdout_manifest.csv',C.MANIFEST]

 files+=list((HOME/'outputs').glob('*/*manifest*.csv'))
 for p in sorted(set(files)):
  if p.is_relative_to(OUT):continue
  t=C.csv(p);sources.append(dict(path=str(p),sha256=C.sha(p),rows=len(t)))
  if {'phase_u','phase_v'}.issubset(t):points+=list(t[['phase_u','phase_v']].dropna().itertuples(index=False,name=None))
  if 'phase_frac' in t:lines+=list(pd.to_numeric(t.phase_frac,errors='coerce').dropna())
  if 'seed' in t:seeds|=set(int(v) for v in pd.to_numeric(t.seed,errors='coerce').dropna())
  if 'raster_key' in t:rasters|=set(t.raster_key.dropna())
 return np.unique(np.asarray(points,float),axis=0),np.unique(lines),seeds,rasters,sources

def choose_phases(RM,RI,MF,REG,m5):
 pts,lns,seeds,rasters,sources=registry(RM,REG)
 templates=RI.template_rows(m5);out={};audit={}
 for fam in ['point','line']:
  configs=sorted({c for sc in SCENARIOS if sc['family']==fam for c in sc['configs']})
  acc=[];reject={};minimum=.05 if fam=='point' else 1e-9
  ks=np.arange(1,20001)
  if fam=='line':
   vals=(.5+ks*MF.SQ2M1)%1-.5
   distances=np.min(np.abs((vals[:,None]-lns[None,:]+.5)%1-.5),axis=1)
   order=list(ks[np.argsort(-distances,kind='stable')])
  else:order=list(ks)
  while order:
   k=int(order.pop(0))
   ph=({'u':float((.5+k/MF.RHO)%1),'v':float((.5+k/MF.RHO**2)%1)} if fam=='point' else {'phi':float((.5+k*MF.SQ2M1)%1-.5)})
   if fam=='point':
    used=np.vstack([pts,np.array([[a['u'],a['v']] for a in acc])]) if acc else pts
    d=np.sqrt(np.sum(((np.array([ph['u'],ph['v']])-used+.5)%1-.5)**2,axis=1));near=float(d.min())
   else:
    used=np.r_[lns,[a['phi'] for a in acc]];near=float(np.min(np.abs((ph['phi']-used+.5)%1-.5)))
   if near<minimum:reject['distance']=reject.get('distance',0)+1;continue
   why=[]
   for cfg in configs:
    built=MF.build(templates[cfg]['row'],fam,ph)
    if built['raster_key'] in rasters:why.append('seen_raster')
    if not built['resource_ok']:why.append('resource')
    if fam=='line' and built['registered_without_lit']:why.append('registered_without_lit')
   if why:
    for w in set(why):reject[w]=reject.get(w,0)+1
    MF._RAST.clear()
    continue
   acc.append(dict(block=len(acc),k=k,**ph,min_distance=near))
   if len(acc)==4:break
   if fam=='line':
    distances=np.minimum(distances,np.abs((vals-ph['phi']+.5)%1-.5))
    remaining=set(order);order=[int(k) for k in ks[np.argsort(-distances,kind='stable')] if int(k) in remaining]
  C.require(len(acc)==4,f'Not enough novel {fam} rasters; stop before imaging')
  out[fam]=acc;audit[fam]=dict(min_distance=minimum,rejected=reject,phases=acc,
      rule=('legacy 0.05-cell sequence' if fam=='point' else 'geometry-only maximin on 20000 sequence locations; every tested candidate raster novel'),
      preimaging_revision=('old 0.015-cycle threshold exhausted before any imaging; no outcomes used' if fam=='line' else None))
 newseeds={base+100*s+10*b+r for base in SEEDS.values() for s,b,r in itertools.product(range(3),range(4),[1,2])}
 C.require(not newseeds&seeds,'Holdout noise seeds previously used')
 return out,dict(phases=audit,known_positions=len(pts),known_line_phases=len(lns),known_seeds=len(seeds),sources=sources,
                qualification='same position domain, new phases and new noise; not out-of-domain generalization',
                legacy_scope=REG.registry.__doc__,unlisted_technical_material_not_claimed_unseen=True)

def manifest_gate(m):
 expected={f'{cfg}|{sf}|bE{b}|s{sc["sigma"]}|r{r}|{sc["env"]}' for sc in SCENARIOS for cfg,sf,b in itertools.product(sc['configs'],C.SURFS,range(4)) for r in ([0] if sc['sigma']==0 else [1,2])}
 C.require(len(m)==288 and set(m.case_key)==expected and not m.case_key.duplicated().any() and not m.forward_key.duplicated().any(),'Holdout matrix incomplete/duplicated')
 for r in m.to_dict('records'):
  sc=next(s for s in SCENARIOS if s['id']==r['scenario']);b=int(r['phase_block'][1:])
  seed=None if r['sigma']==0 else SEEDS[r['family']]+100*C.SURFS.index(r['surface'])+10*b+r['rep']
  C.require(C.clean(r['seed'])==seed and r['rep'] in ([0] if r['sigma']==0 else [1,2]),'Seed/repetition mismatch')
  C.require(r['config'] in sc['configs'] and r['env']==sc['env'] and r['sigma']==sc['sigma'],'Scenario identity mismatch')
  C.require(r['method']==('DF' if r['family']=='point' else 'P4') and r['rays_per_pixel']==1,'Method mismatch')
  C.require(0<r['peak']<=1 and 0<r['integral']<=11695.01,'Resource mismatch')
 C.require(int(m.entry_subset.sum())==13,'Wrong technical entry count')

def prepare():
 C.require(not (OUT/'freeze.json').exists(),'Already frozen; do not overwrite registration')
 RM,RI,MF,REG=setup();m5=C.csv(BASE/'model_prep05/formal_manifest.csv')
 phases,audit=choose_phases(RM,RI,MF,REG,m5);planner=RM.Planner(m5);rows=[]
 template=C.rowdict(C.csv(C.MANIFEST).iloc[0])
 for sc in SCENARIOS:
  for cfg,sf,b in itertools.product(sc['configs'],C.SURFS,range(4)):
   for rep in ([0] if sc['sigma']==0 else [1,2]):
    r=planner.row('MP06E_HOLDOUT',cfg,sf,phases[sc['family']][b],sc['sigma'],rep,sc['env'],seed_base=SEEDS,block_label=f'E{b}')
    r.update(scenario=sc['id'],main_d=sc['d'],action='new',source_pack='',source_pack_sha256='',entry_subset=False,
      method_bundle_sha256=template['method_bundle_sha256'],global_domain_sha256=template['global_domain_sha256'],evaluation_source_sha256=template['evaluation_source_sha256'])
    rows.append(r)
 m=pd.DataFrame(rows)

 for sid,cfg,sf in [('H1','L_w1.25_s8','M24_curv'),('H3','D_r1_p8x6','M24_curv'),('H4','D_r1_p8x6','B_curv'),('H5','D_r1_p8x4','M10_curv'),('H6','L_w0.5_s8','M10_curv'),('H1','L_w1_s8','B_curv'),('H2','L_w1.25_s8','B_curv')]:
  m.loc[(m.scenario==sid)&(m.config==cfg)&(m.surface==sf)&(m.phase_block=='E0'),'entry_subset']=True
 manifest_gate(m)
 for r in m.to_dict('records'):B.prepare(C.rowdict(r))
 C.save_csv(OUT/'manifest.csv',m)
 protocol=dict(created=time.time(),scientific_render_count=0,new_cap=288,entry_included=13,scenarios=SCENARIOS,phase_registry=audit,seed_base=SEEDS,
  policy=C.POLICY,primary_tau=.15,primary_floor=.5,primary_d_by_scenario={s['id']:s['d'] for s in SCENARIOS},
  confirm='fixed recommended configuration: all three surface phase-mean Ctau >= .5 plus every-image RMSE<=.3,w=0,u_share<=.05,defined metrics and exposure>0',
  direction='positive mean paired difference on each of three surfaces; additionally report all 12 surface-phase differences, no significance claim',
  near_optimal='recommended score within .02 of highest score among admissible tested alternatives; two percentage points is an engineering band, not equivalence CI',
  failure='retain scientific failure; do not tune thresholds, replace phases, reselect a validation winner, or infer full candidate-pool absence from subset',
  technical_stop='first exception, file/key/hash mismatch, missing/duplicate pair, illegal metric; join workers; preserve completed cases',
  illumination_authorised=False)
 C.save(OUT/'protocol.json',protocol)
 files=current_files();freeze=dict(files=files,freeze_id=C.digest(files),manifest_sha256=C.sha(OUT/'manifest.csv'),created=time.time())
 C.save(OUT/'freeze.json',freeze)
 print(json.dumps(dict(status='registered_no_images_yet',cases=len(m),phase_registry=audit['phases'],freeze_id=freeze['freeze_id']),ensure_ascii=False))

def verify():
 C.install_guard();f=C.load(OUT/'freeze.json');C.require(C.digest(f['files'])==f['freeze_id'],'Freeze digest')
 C.verify_files(f['files']);C.require(C.sha(OUT/'manifest.csv')==f['manifest_sha256'],'Manifest drift')
 m=C.csv(OUT/'manifest.csv');manifest_gate(m);return f,m

def init_worker(f):
 C.install_guard();C.verify_files(f['files'])
 C.ALLOWED_PROJECT_READS={p for p in f['files'] if Path(p).is_relative_to(C.PROJECT)}

def work(r,fid):
 store=C.Store(OUT)
 try:
  meta,arrays=B.real_compute(r,store,'explicit_user_authorised_production');store.commit(r,fid,meta,arrays)
  return dict(ok=True,case_key=r['case_key'])
 except BaseException as e:
  store.fail(r['case_key'],e);return dict(ok=False,case_key=r['case_key'],error=repr(e),trace=traceback.format_exc()[-1800:])

def execute(stage):
 f,m=verify();store=C.Store(OUT);fid=f['freeze_id']
 if stage=='full':
  a=C.load(OUT/'entry_audit.json');C.require(a['ok'] and a['freeze_id']==fid,'Entry gate not passed')
 rows=m[m.entry_subset if stage=='entry' else ~m.entry_subset];todo=[]
 for r in rows.to_dict('records'):
  r=C.rowdict(r)
  if store.resume(r,fid) is None:todo.append(r)
 if not todo:print('Already complete; no render');return
 lock=C.writable(OUT/'run.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 start=time.time();counter=[0]
 try:
  def before(r):
   C.require(r['case_key'] in set(m.case_key),'Outside registered cap');store.reserve(r,fid)
  def after(r,res):
   if not res['ok']:return
   store.resume(r,fid);counter[0]+=1
   if counter[0]%24==0:print(f'{stage} {counter[0]}/{len(todo)}',flush=True)
  ENGINE.bounded_dispatch(todo,work,before,after,lambda r:store.fail(r['case_key'],'cancelled after technical stop'),1 if stage=='entry' else 8,init_worker,(f,),(fid,))
  verify();C.save(OUT/f'{stage}_execution.json',dict(freeze_id=fid,completed=len(todo),resumed=len(rows)-len(todo),seconds=time.time()-start))
 finally:lock.unlink(missing_ok=True)

def collect(m,f):
 store=C.Store(OUT);res=[];sens=[]
 for r in m.to_dict('records'):
  meta=store.resume(C.rowdict(r),f['freeze_id']);C.require(meta is not None,'Missing saved case')
  res.append(meta['result']);sens.extend([dict(case_key=r['case_key'],**a) for a in meta['sensitivity']])
 R=pd.DataFrame(res);T=pd.DataFrame(sens);C.metric_gate(R,m)
 import analysis as A
 A.sensitivity_gate(T,R)
 return R,T

def audit_entry():
 f,m=verify();m=m[m.entry_subset];r,t=collect(m,f)
 sys.path.insert(0,str(OLD));from audit_execution import independent_surface
 st=C.Store(OUT);checks=[]
 for fam in ['point','line']:
  rr=C.rowdict(m[m.family==fam].sort_values('case_key').iloc[0]);a=B.load_arrays(rr,st);p=rr['method']+'__ev__'
  v=independent_surface(a[p+'pts'][a[p+'valid']],rr['surface']);q=t[t.case_key==rr['case_key']]
  err=max(max(abs(x.Ctau-v[x.d_mm,x.tau_mm][0]),abs(x.Cvalid-v[x.d_mm,x.tau_mm][1])) for x in q.itertuples())
  C.require(err<1e-10,'Independent coverage mismatch');checks.append(dict(case=rr['case_key'],coverage_error=err))
 noisy=r[r.sigma>0].groupby(['config','env','surface','phase_block','sigma'])
 C.require(noisy.rep.apply(lambda a:set(a)=={1,2}).all() and noisy.diag_noise_sha.nunique().eq(2).all(),'Entry noise pair failure')
 C.require(r[r.sigma==0].diag_noise_sha.isna().all(),'Zero noise drew noise')
 C.save(OUT/'entry_audit.json',dict(ok=True,freeze_id=f['freeze_id'],cases=len(r),independent_checks=checks,states=r.state.value_counts().to_dict()))
 print('Real entry independent check passed',len(r))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','entry','audit-entry','full']);a=p.parse_args()
 try:
  if a.stage=='prepare':prepare()
  elif a.stage=='audit-entry':audit_entry()
  else:execute(a.stage)
 except Exception:
  traceback.print_exc();sys.exit(2)
