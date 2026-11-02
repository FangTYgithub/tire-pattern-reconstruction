# Define the blur matrix, selection criteria and content-bound result storage.

from __future__ import annotations
import os, sys, json, hashlib, itertools, sqlite3, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HOME = HERE.parents[1]
PROJECT = HOME / 'research/project_Part1_v1.7'
PRIOR = HOME / 'outputs/mp06c_zero_close_2026-10-21'
OUT = HOME / 'outputs/mp06d_blur'
MANIFEST = PRIOR / 'blur_manifest_7920.csv'
EXPECTED_MANIFEST = '8bfe884dcbe1b4e048c4e957b191cd8c835b5d480fd979401713dedffdbb6532'
DOMAIN = PROJECT / 'outputs/surface_revision/s1_real01/domain/global_domain_step0p50.npz'
DOTS = ['D_r1_p8x'+x for x in ['2.5','3','4','6','8']] + ['D_r1.5_p8x4','D_r1.5_p8x6']
LINES = ['L_w'+x+'_s8' for x in ['0.5','0.75','1','1.25','1.5']]
ENVS = dict(NOMINAL=1., BLUR1_5=1.5, BLUR2=2., BLUR2_5=2.5, BLUR3=3.)
SURFS = ['B_curv','M10_curv','M24_curv']
SIGMAS = [0,20,40,60,80,100]
DS, TAUS = [5.,6.], [.1,.15,.2,.3]  # Support-distance and height-tolerance grids [mm]; primary pair: 6 and 0.15.
# Selection thresholds; floors and tie_band are fractions, rmse_max is in millimetres.
POLICY = dict(domain='global_30947', ds=DS, taus=TAUS, main=[6.,.15], floors=[.5,.7],
              rmse_max=.3, wrong_max=0, unconfirmed_max=.05, tie_band=.02,
              aggregate='repetitions within phase -> four equal phases -> worst surface',
              tie='reference if near best; else smaller maximum image point RMSE, then config ID')
IDENTITY = ['case_key','forward_key','family','config','surface','phase_block','sigma','rep','seed',
            'phase_u','phase_v','phase_frac','env','method','method_bundle_sha256','psf_true_px']
READS = set()
ALLOWED_PROJECT_READS = None
_guard_installed = False

class Stop(RuntimeError):
    pass

def require(ok, msg):
    if not bool(ok):
        raise Stop(msg)

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x

def canon(x): return json.dumps(clean(x), sort_keys=True, ensure_ascii=False, separators=(',',':'), allow_nan=False)
def digest(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csv(p): return pd.read_csv(p, float_precision='round_trip', low_memory=False)
def rowdict(r): return clean(dict(r))
def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def arrays_sha(a): return digest({k:array_sha(v) for k,v in sorted(a.items())})

def writable(p):
    p=Path(p).resolve()
    require(p.is_relative_to(OUT.resolve()), 'Write outside isolated output: '+str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def atomic(p, writer):
    p=writable(p); tmp=p.with_name(p.name+'.tmp'+str(os.getpid()))
    writer(tmp)
    os.replace(tmp,p)
    return p
def save(p,x): return atomic(p, lambda t:t.write_text(canon(x),encoding='utf-8'))
def save_csv(p,t): return atomic(p,lambda q:t.to_csv(q,index=False,encoding='utf-8-sig'))

def install_guard():
    global _guard_installed
    if _guard_installed: return
    def audit(event,args):
        if event=='open':
            raw,mode,flags=args
            if not isinstance(raw,(str,bytes,os.PathLike)): return
            p=Path(os.fsdecode(raw)).resolve()
            if not p.is_relative_to(PROJECT): return
            writing=bool((flags or 0)&(os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC))
            if writing: raise Stop('Scientific project is read-only: '+str(p))


            if p.suffix=='.pyc':raise FileNotFoundError('MP06D source-only scientific import: '+str(p))
            READS.add(str(p))
            if ALLOWED_PROJECT_READS is not None and str(p) not in ALLOWED_PROJECT_READS:
                raise Stop('Unfrozen project input: '+str(p))
        elif event in ('os.remove','os.rename','os.mkdir','os.rmdir','os.chmod','os.utime'):
            paths=args[:2] if event=='os.rename' else args[:1]
            for raw in paths:
                if isinstance(raw,(str,bytes,os.PathLike)):
                    p=Path(os.fsdecode(raw)).resolve()
                    if p.is_relative_to(PROJECT) and not (event=='os.mkdir' and p.is_dir()):
                        raise Stop('Scientific project mutation denied: '+str(p))
    sys.addaudithook(audit); _guard_installed=True

def check_manifest(m, full=True):
    cols=set(IDENTITY+['action','source_pack','source_pack_sha256','entry_subset','integral','peak','rays_per_pixel','field_json'])
    require(cols.issubset(m.columns), 'Manifest missing columns: '+str(cols-set(m.columns)))
    require(not m.case_key.duplicated().any() and not m.forward_key.duplicated().any(), 'Duplicate case/forward keys')
    for r in m.to_dict('records'):
        r=rowdict(r); s=r['sigma']; rep=r['rep']; sf=r['surface']; cfg=r['config']; b=r['phase_block']
        require(cfg in DOTS+LINES and sf in SURFS and s in SIGMAS and b in range(4), 'Unknown target')
        require(r['family']==('point' if cfg in DOTS else 'line'), 'Family mismatch')
        require(r['method']==('DF' if cfg in DOTS else 'P4'), 'Wrong decoder')
        require(r['env'] in ENVS and r['psf_true_px']==ENVS[r['env']], 'PSF/env mismatch')
        require(rep in ([0] if s==0 else [1,2]), 'Wrong repetition label')
        seed=None if s==0 else (9870000 if cfg in DOTS else 9880000)+100*SURFS.index(sf)+10*b+rep
        require(r['seed']==seed, 'Wrong noise seed / zero-noise seed')
        require(r['case_key']==f'{cfg}|{sf}|b{b}|s{int(s)}|r{rep}|{r["env"]}', 'Case key does not match fields')
        require(r['action'] in ['new','reuse'], 'Unknown action')
        require(r['rays_per_pixel']==1 and 0<r['peak']<=1 and 0<r['integral']<=11695.01, 'Resource/sample mismatch')
        require(json.loads(r['field_json'])=={'kind':'const','value':1.0}, 'Illumination outside blur scope')
        require(not r['entry_subset'] or r['action']=='new', 'Reused case in new entry set')
        if r['action']=='reuse':
            require(bool(r['source_pack']) and len(r['source_pack_sha256'] or '')==64, 'Missing reuse source')
    if full:
        expected={f'{cfg}|{sf}|b{b}|s{s}|r{rep}|{e}' for cfg,e,sf,b,s in itertools.product(DOTS+LINES,ENVS,SURFS,range(4),SIGMAS) for rep in ([0] if s==0 else [1,2])}
        require(set(m.case_key)==expected, 'Target matrix missing/extra cases')
        require(m.action.value_counts().to_dict()=={'new':4696,'reuse':3224}, 'Wrong new/reuse counts')
        require(int(m.entry_subset.sum())==24, 'Entry subset count !=24')

    for fam,g in m.groupby('family'):
        fields=['phase_u','phase_v'] if fam=='point' else ['phase_frac']
        require(g.groupby('phase_block')[fields].nunique().eq(1).all().all(), 'Phase label aliases physical locations')

def manifest():
    require(sha(MANIFEST)==EXPECTED_MANIFEST, 'Canonical manifest changed')
    m=csv(MANIFEST); check_manifest(m)
    return m

def binding(r,freeze_id):
    return dict(row={k:rowdict(r)[k] for k in IDENTITY}, forward_key=r['forward_key'],
                execution_freeze=freeze_id, evaluation=POLICY, domain_sha=r['global_domain_sha256'],
                evaluation_source=r['evaluation_source_sha256'])

def metric_gate(records, expected):

    cols=set(IDENTITY+['state','glob_Ctau','glob_Cvalid','glob_pt_rmse','glob_pt_p95','glob_n_valid','n_accepted','c','w','u'])
    require(cols.issubset(records.columns), 'Missing result columns: '+str(cols-set(records.columns)))
    require(not records.case_key.duplicated().any(), 'Duplicate result case')
    require(set(records.case_key)==set(expected.case_key), 'Incomplete/mixed result key set')
    exp=expected.set_index('case_key')
    for r in records.to_dict('records'):
        r=rowdict(r); target=rowdict(exp.loc[r['case_key']].to_dict()|{'case_key':r['case_key']})
        require(all(r[k]==target[k] for k in IDENTITY), 'Record identity/rep/seed/method mismatch: '+r['case_key'])
        ct,cv=r['glob_Ctau'],r['glob_Cvalid']
        require(ct is not None and cv is not None and 0<=ct<=cv+1e-12 and cv<=1, 'Invalid surface coverage')
        for k in ['glob_n_valid','n_accepted','c','w','u']:
            require(isinstance(r[k],(int,float)) and r[k]>=0 and int(r[k])==r[k], 'Invalid count '+k)
        require(r['c']+r['w']+r['u']==r['n_accepted'], 'Identity denominator != accepted')
        require(r['glob_n_valid']<=r['n_accepted'], 'Global valid exceeds accepted')
        n=r['glob_n_valid']; rm,pp=r['glob_pt_rmse'],r['glob_pt_p95']
        require((n==0 and rm is None and pp is None) or (n>0 and rm is not None and pp is not None and rm>=0 and pp>=0), 'Point error undefined/defined against wrong denominator')
        if n<3: require(cv==0 and ct==0, 'Unsupported surface has positive coverage')
        require(r['state'] in ['ok','no_output'], 'Technical/unknown state is not scientific completion')
        require((r['n_accepted']==0)==(r['state']=='no_output'), 'No-output state inconsistent')
    return dict(records=len(records), complete=True)

# Check saved pixel/cloud/identity arrays against summary, without decoding.
def pack_gate(meta,arrays,r):

    metric_gate(pd.DataFrame([meta['result']]),pd.DataFrame([r]))
    m=r['method'];prefix=m+'__ev__';result=meta['result']
    require('image' in arrays and arrays['image'].shape==(512,512) and arrays['image'].dtype==np.uint8,'Invalid saved image')
    require(any(k.startswith(m+'__raw__') for k in arrays),'Missing raw decoder arrays')
    ev={k[len(prefix):]:v for k,v in arrays.items() if k.startswith(prefix)}
    require({'pts','accepted','valid','in_global','zr'}.issubset(ev),'Missing evaluation arrays')
    pts=ev['pts'];n=len(pts)
    require(pts.shape==(n,3),'Invalid point shape')
    for k in ['accepted','valid','in_global']:
        require(ev[k].shape==(n,) and ev[k].dtype==bool,'Invalid point mask '+k)
    require(ev['zr'].shape==(n,),'Reference height shape')
    require((~ev['valid']|ev['accepted']).all(),'Valid points are not accepted')
    require(int(ev['accepted'].sum())==result['n_accepted'],'Accepted count does not match arrays')
    valid=ev['valid']&ev['in_global'];require(int(valid.sum())==result['glob_n_valid'],'Global point denominator mismatch')
    if valid.any():
        err=pts[valid,2].astype(float)-ev['zr'][valid].astype(float)
        require(np.isfinite(err).all(),'Nonfinite valid point errors')
        rm=float(np.sqrt(np.mean(err**2)));p95=float(np.percentile(np.abs(err),95))
        require(abs(rm-result['glob_pt_rmse'])<=2e-6 and abs(p95-result['glob_pt_p95'])<=2e-6,'Saved point RMSE/P95 inconsistent with cloud')
    tag='full_point_' if r['family']=='line' else 'point_'
    masks=[]
    for key,name in [('c','correct'),('w','wrong'),('u','unconf')]:
        require(tag+name in ev,'Missing full identity mask')
        require(ev[tag+name].shape==(n,) and ev[tag+name].dtype==bool,'Invalid identity mask')
        require(int(ev[tag+name].sum())==result[key],'Identity count does not match arrays: '+key)
        masks.append(ev[tag+name].astype(np.int8))
    require(np.array_equal(sum(masks),ev['accepted'].astype(np.int8)),'Identity masks do not partition accepted points')

# Store case arrays and status records with checked content identities.
class Store:

    def __init__(self,root=OUT):
        self.root=writable(Path(root)/'store.sqlite').parent
        self.db=self.root/'store.sqlite'
        with self.connect() as con:
            con.execute('CREATE TABLE IF NOT EXISTS cases (ck TEXT PRIMARY KEY,binding TEXT,status TEXT,sha TEXT,error TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY,ck TEXT,stage TEXT,status TEXT,t REAL)')


    @__import__('contextlib').contextmanager
    def connect(self):
        con=sqlite3.connect(self.db,timeout=60)
        try:
            with con:yield con
        finally:con.close()
    def path(self,ck): return self.root/'data'/(hashlib.sha256(ck.encode()).hexdigest()[:24]+'.npz')
    def image_path(self,ck): return self.path(ck).with_suffix('.image.npz')
    def save_image(self,r,img,diag):
        p=self.image_path(r['case_key'])
        require(not p.exists(),'Existing image checkpoint must not be overwritten')
        require(img.shape==(512,512) and img.dtype==np.uint8,'Invalid image checkpoint')
        meta=dict(case_key=r['case_key'],forward_key=r['forward_key'],image_sha=arrays_sha({'image':img}),diag=diag)
        def writer(t):
            with t.open('wb') as f:np.savez_compressed(f,_meta=np.array(canon(meta)),image=img)
        atomic(p,writer)
    def load_image(self,r):
        p=self.image_path(r['case_key'])
        if not p.exists():return None
        with np.load(p,allow_pickle=False) as z:img=z['image'];meta=json.loads(str(z['_meta']))
        require(meta['case_key']==r['case_key'] and meta['forward_key']==r['forward_key'],'Stale image checkpoint')
        require(img.shape==(512,512) and img.dtype==np.uint8 and meta['image_sha']==arrays_sha({'image':img}),'Corrupt image checkpoint')
        return img,meta['diag']
    def event(self,ck,stage,status):
        with self.connect() as c: c.execute('INSERT INTO events(ck,stage,status,t) VALUES(?,?,?,?)',(ck,stage,status,time.time()))
    def reserve(self,r,fid):
        ck=r['case_key']; b=canon(binding(r,fid))
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            require(c.execute('SELECT count(*) FROM cases').fetchone()[0]<4696,'Attempt cap reached')
            require(c.execute('SELECT 1 FROM cases WHERE ck=?',(ck,)).fetchone() is None,'Case already attempted; review before retry')
            c.execute('INSERT INTO cases VALUES(?,?,?,?,?)',(ck,b,'reserved',None,None))
        self.event(ck,'case','reserved')
    def fail(self,ck,error):
        with self.connect() as c: c.execute('UPDATE cases SET status=?,error=? WHERE ck=?',('technical_stop',str(error)[:600],ck))
        self.event(ck,'case','technical_stop')
    def commit(self,r,fid,meta,arrays,interrupt_after_pack=False):
        ck=r['case_key']; meta=dict(meta,binding=binding(r,fid),arrays_sha256=arrays_sha(arrays))
        with self.connect() as c:
            state=c.execute('SELECT binding,status FROM cases WHERE ck=?',(ck,)).fetchone()
        require(state==(canon(binding(r,fid)),'reserved'),'Commit without matching reservation')
        require(not self.path(ck).exists(),'Existing pack must be resumed, never overwritten')
        pack_gate(meta,arrays,r)
        def writer(t):
            with t.open('wb') as f: np.savez_compressed(f,_meta=np.array(canon(meta)),**arrays)
        p=atomic(self.path(ck),writer); h=sha(p)
        if interrupt_after_pack: raise InterruptedError('isolated interruption after atomic pack')
        with self.connect() as c: c.execute('UPDATE cases SET status=?,sha=? WHERE ck=?',('complete',h,ck))
        self.event(ck,'case','complete')

        self.image_path(ck).unlink(missing_ok=True)
        return meta
    def resume(self,r,fid):
        ck=r['case_key'];p=self.path(ck)
        with self.connect() as c: rec=c.execute('SELECT binding,status,sha FROM cases WHERE ck=?',(ck,)).fetchone()
        if rec is None:
            require(not p.exists(),'Unindexed result pack; review required');return None
        b,status,h=rec
        require(b==canon(binding(r,fid)),'Stale method/row/evaluation/freeze binding')
        require(p.exists(),'Incomplete attempt without committed pack; no automatic rerender')
        if h: require(sha(p)==h,'Pack checksum mismatch')
        try:
            with np.load(p,allow_pickle=False) as z:
                meta=json.loads(str(z['_meta']));arrays={k:z[k] for k in z.files if k!='_meta'}
        except Exception as e: raise Stop('Unreadable pack: '+repr(e)) from e
        require(meta['binding']==binding(r,fid) and meta['arrays_sha256']==arrays_sha(arrays),'Pack binding/array checksum mismatch')
        pack_gate(meta,arrays,r)
        if status=='reserved':
            with self.connect() as c:c.execute('UPDATE cases SET status=?,sha=? WHERE ck=?',('complete',sha(p),ck))
            self.event(ck,'case','recovered_atomic_pack')
        else:require(status=='complete','Stopped attempt is not a completed case')
        return meta

def verify_files(files):
    bad=[p for p,h in files.items() if not Path(p).is_file() or sha(p)!=h]
    require(not bad,'Frozen input drift: '+str(bad[:10]))

def verify_freeze():
    import importlib.metadata
    f=load(OUT/'freeze.json')
    require(digest(f['files'])==f['freeze_id'],'Freeze record digest mismatch')
    verify_files(f['files'])
    require(f['manifest_sha256']==EXPECTED_MANIFEST,'Wrong frozen manifest')
    require(f['policy']==POLICY,'Evaluation policy changed')
    require(f['python']==sys.version and all(importlib.metadata.version(k)==v for k,v in f['packages'].items()),'Numerical runtime version changed')
    return f
