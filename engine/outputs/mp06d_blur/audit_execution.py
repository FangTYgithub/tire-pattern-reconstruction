# Check completed blur cases and independently recompute selected surface metrics.

import sys,json,sqlite3,hashlib,itertools
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/mp06d_blur'))
import core as C

def radius_by_clipped_cells(v):


    v=v-v[0];rho=0.
    for i in range(3):
        poly=list(v.copy())
        for j in range(3):
            if i==j:continue
            n=2*(v[j]-v[i]);b=float(v[j]@v[j]-v[i]@v[i]);out=[]
            for a,z in zip(poly,poly[1:]+poly[:1]):
                fa=float(n@a-b);fz=float(n@z-b)
                ina=fa<=1e-12;inz=fz<=1e-12
                if ina:out.append(a)
                if ina!=inz:out.append(a+(z-a)*(fa/(fa-fz)))
            poly=out
        if poly:rho=max(rho,float(np.linalg.norm(np.asarray(poly)-v[i],axis=1).max()))
    return rho

def independent_surface(points,surface):
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator
    z=np.load(C.DOMAIN,allow_pickle=False);xy=np.column_stack([z['x'],z['y']]);zr=z[{'B_curv':'z_B_curv','M10_curv':'z_A_curv','M24_curv':'z_C_curv'}[surface]]
    p=np.asarray(points,dtype=float);p=p[np.isfinite(p).all(axis=1)]
    _,idx=np.unique(np.rint(p[:,:2]/1e-6).astype(np.int64),axis=0,return_index=True);p=p[np.sort(idx)]
    if len(p)<3:return {(d,t):(0.,0.) for d,t in itertools.product(C.DS,C.TAUS)}
    tri=Delaunay(p[:,:2]);simp=tri.find_simplex(xy)
    radii=np.array([radius_by_clipped_cells(v) for v in p[tri.simplices,:2]])
    err=LinearNDInterpolator(tri,p[:,2])(xy)-zr
    return {(d,t):(float(((simp>=0)&(radii[np.maximum(simp,0)]<=d)&np.isfinite(err)&(abs(err)<=t)).sum()/len(xy)),
                   float(((simp>=0)&(radii[np.maximum(simp,0)]<=d)&np.isfinite(err)).sum()/len(xy))) for d,t in itertools.product(C.DS,C.TAUS)}

def main():
    C.install_guard();f=C.verify_freeze();m=C.manifest();sub=m[m.entry_subset];store=C.Store();records=[];packs={};max_rmse=0.
    C.require(abs(radius_by_clipped_cells(np.array([[0.,0.],[2.,0.],[1.,.1]]))-.505)<1e-10,'Independent radius reference failed')
    for r in sub.to_dict('records'):
        meta=store.resume(C.rowdict(r),f['freeze_id']);result=meta['result'];records.append(result)
        with np.load(store.path(r['case_key']),allow_pickle=False) as z:a={k:z[k] for k in z.files if k!='_meta'}
        ev={k.split('__ev__',1)[1]:v for k,v in a.items() if '__ev__' in k};mask=ev['valid']&ev['in_global']
        if mask.any():
            err=ev['pts'][mask,2].astype(float)-ev['zr'][mask].astype(float)
            rm=float(np.linalg.norm(err)/np.sqrt(mask.sum()));max_rmse=max(max_rmse,abs(rm-result['glob_pt_rmse']))
        if result['n_accepted']==0:C.require(result['glob_Ctau']==0 and result['glob_pt_rmse'] is None,'Illegal no-output')
        packs[r['case_key']]=(r,meta,a,ev)
    t=pd.DataFrame(records);C.metric_gate(t,sub)
    key=['config','surface','phase_block','sigma','env']
    p=t[t.sigma>0].groupby(key).agg(n=('rep','size'),reps=('rep',lambda q:set(q)=={1,2}),seeds=('seed','nunique'),noise=('diag_noise_sha','nunique'),image=('image_sha','nunique'))
    C.require(p.n.eq(2).all() and p.reps.all() and p.seeds.eq(2).all() and p.noise.eq(2).all() and p.image.eq(2).all(),'Independent pairing check failed')
    sampled=[]
    for fam in ['point','line']:
        for s0 in [True,False]:
            q=t[(t.family==fam)&(t.sigma.eq(0)==s0)].sort_values('case_key').iloc[0]
            r,meta,a,ev=packs[q.case_key];values=independent_surface(ev['pts'][ev['valid']],r['surface'])
            dif=max(max(abs(v['Ctau']-values[v['d_mm'],v['tau_mm']][0]),abs(v['Cvalid']-values[v['d_mm'],v['tau_mm']][1])) for v in meta['sensitivity'])
            C.require(dif<1e-10,'Independent node coverage differs '+q.case_key)
            sampled.append(dict(case_key=q.case_key,max_coverage_difference=dif))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[s['case_key'] for s in sampled]+[t.sort_values(['sigma','case_key']).iloc[-1].case_key]
    fig=plt.figure(figsize=(13,3.5*len(selected)))
    for i,ck in enumerate(selected):
        r,meta,a,ev=packs[ck];v=ev['valid']&ev['in_global'];pts=ev['pts'][v];err=pts[:,2]-ev['zr'][v]
        ax=fig.add_subplot(len(selected),3,3*i+1);ax.imshow(a['image'],cmap='gray',vmin=0,vmax=255);ax.set_axis_off();ax.set_title(ck.replace('|',' / '),fontsize=7)
        ax=fig.add_subplot(len(selected),3,3*i+2,projection='3d')
        if len(pts):ax.scatter(*pts.T,c=pts[:,2],vmin=-16,vmax=4,s=1,cmap='viridis')
        else:ax.text2D(.05,.5,'Keine gültigen Punkte',transform=ax.transAxes)
        ax.set(xlim=(-46,46),ylim=(-46,46),zlim=(-16,4),xlabel='x / mm',ylabel='y / mm',zlabel='z / mm');ax.view_init(25,-65)
        ax=fig.add_subplot(len(selected),3,3*i+3)
        if len(pts):ax.scatter(pts[:,0],pts[:,1],c=err,vmin=-.3,vmax=.3,cmap='coolwarm',s=2)
        else:ax.text(.1,.5,'Keine gültigen Punkte',transform=ax.transAxes)
        ax.set(xlim=(-46,46),ylim=(-46,46),xlabel='x / mm',ylabel='y / mm');ax.set_aspect('equal');ax.set_title(f"n={len(pts)}, Cτ={meta['result']['glob_Ctau']:.3f}")
    fig.suptitle('24-Fälle-Eingangsprüfung: Kamerabild, Punktwolke, Höhenfehler (±0,3 mm)');fig.tight_layout(rect=(0,0,1,.98));fig.savefig(C.OUT/'entry_preview.png',dpi=120);plt.close(fig)
    result=dict(ok=True,freeze_id=f['freeze_id'],entry_check_sha256=C.sha(C.OUT/'entry_check.json'),cases=24,
        states=t.state.value_counts().to_dict(),point_error_maxdiff=max_rmse,paired_groups=len(p),independent_node_checks=sampled,
        checks=['24 packs reopened and array hashes verified','legal scientific no-output retained','two seeds/noise fields/images per noisy pair',
        'global point errors independently recomputed','4 clouds ×8 thresholds: clipped Voronoi-cell support + scipy interpolation'],
        cases_summary=t[['case_key','n_accepted','glob_n_valid','glob_Ctau','glob_Cvalid','glob_pt_rmse','c','w','u']].to_dict('records'))
    C.save(C.OUT/'entry_audit.json',result);C.verify_freeze();print(json.dumps({k:v for k,v in result.items() if k!='cases_summary'},ensure_ascii=False))
# Check completed cases, paired effects and selection rows against saved outputs.
def audit_final():
    C.install_guard();fz=C.verify_freeze();man=C.manifest();folder=C.OUT/'analysis'
    results=C.csv(folder/'results.csv');sens=C.csv(folder/'sensitivity.csv.gz');surf=C.csv(folder/'surfaces.csv');choices=C.csv(folder/'choices.csv');effects=C.csv(folder/'paired_effects.csv.gz')
    expected={f'{cfg}|{sf}|b{b}|s{s}|r{rep}|{env}' for cfg,env,sf,b,s in itertools.product(C.DOTS+C.LINES,C.ENVS,C.SURFS,range(4),C.SIGMAS) for rep in ([0] if s==0 else [1,2])}
    C.require(len(results)==7920 and len(set(results.case_key))==7920 and set(results.case_key)==expected,'Independent full matrix check failed')
    C.require(len(sens)==63360 and len(effects)==6600 and len(choices)==960,'Independent analysis cardinality mismatch')
    idx=results.set_index('case_key');delta=[]
    keys=['family','surface','env','sigma','phase_block','phase_u','phase_v','phase_frac','rep','seed']
    for row in effects.to_dict('records'):
        a=C.rowdict(idx.loc[row['case_key_left']]);b=C.rowdict(idx.loc[row['case_key_right']])
        C.require(all(a[k]==b[k] for k in keys),'Independent physical pairing mismatch')
        delta.append(abs((b['glob_Ctau']-a['glob_Ctau'])-row['delta_Ctau']))
    C.require(max(delta)<1e-12 and not effects.duplicated(['left_config','right_config','case_key_left','case_key_right']).any(),'Duplicated or wrong paired differences')

    checked=0;maxmean=0.
    result_groups={k:g for k,g in results.groupby(['config','env','sigma'])}
    sensitivity_groups={k:g.set_index('case_key') for k,g in sens.groupby(['d_mm','tau_mm'])}
    surface_values=surf.set_index(['config','env','sigma','surface','d_mm','tau_mm']).Ctau.to_dict()
    for row in choices.to_dict('records'):
        cfgs=C.DOTS if row['family']=='point' else C.LINES;qualified=[];scores={};rm={}
        for cfg in cfgs:
            q=result_groups[cfg,row['env'],row['sigma']]
            z=sensitivity_groups[row['d_mm'],row['tau_mm']].loc[q.case_key]
            means=[];safe=True
            for sf in C.SURFS:
                qq=q[q.surface==sf];means.append(float(np.mean([z.loc[qq[qq.phase_block==b].case_key,'Ctau'].mean() for b in range(4)])))
                safe &= bool(qq.glob_pt_rmse.notna().all() and qq.glob_pt_rmse.le(.3).all() and qq.w.eq(0).all() and ((qq.c+qq.w)>0).all() and (qq.u/(qq.c+qq.w+qq.u)).le(.05).all())
                sk=(cfg,row['env'],row['sigma'],sf,row['d_mm'],row['tau_mm'])
                C.require(sk in surface_values,'Missing surface aggregate')
                maxmean=max(maxmean,abs(means[-1]-float(surface_values[sk])))
            scores[cfg]=min(means);rm[cfg]=q.glob_pt_rmse.max()
            if safe and min(means)>=row['floor']:qualified.append(cfg)
        got=[] if pd.isna(row['qualified']) else row['qualified'].split(';')
        C.require(sorted(qualified)==got,'Independent admissible set differs')
        if qualified:
            best=max(scores[c] for c in qualified);near=sorted(c for c in qualified if scores[c]>=best-.02-1e-12)
            ref='D_r1_p8x6' if row['family']=='point' else 'L_w0.5_s8';pref=ref if ref in near else min(near,key=lambda c:(rm[c],c))
            C.require(near==row['near_optimal'].split(';') and pref==row['preferred'] and abs(scores[pref]-row['pref_score'])<1e-12,'Independent preference differs')
        else:C.require(pd.isna(row['preferred']),'Inadmissible candidate selected')
        checked+=1
    C.require(maxmean<1e-12,'Surface aggregation differs')

    unchanged=0
    for r in man[man.action=='reuse'].to_dict('records'):
        C.require(C.sha(r['source_pack'])==r['source_pack_sha256'],'Reused source changed');unchanged+=1
    with sqlite3.connect(C.OUT/'store.sqlite') as db:
        ledger=db.execute('SELECT ck,status,sha FROM cases').fetchall();events=db.execute('SELECT stage,status,count(*) FROM events GROUP BY stage,status').fetchall()
    C.require(len(ledger)==4696 and all(st=='complete' for _,st,_ in ledger),'Incomplete new ledger')
    store=C.Store()
    for ck,st,h in ledger:C.require(C.sha(store.path(ck))==h,'New pack changed')

    samples=[]
    for fam in ['point','line']:


        for sigma in ([20,80] if fam=='point' else [0,80]):
            available=man[(man.action=='new')&(man.family==fam)&(man.sigma==sigma)&(man.surface=='M10_curv')].sort_values('case_key')
            C.require(len(available)>0,'No new cases in the registered audit stratum')
            rr=available.iloc[-1]
            meta=store.resume(C.rowdict(rr),fz['freeze_id'])
            with np.load(store.path(rr.case_key)) as a:pts=a[rr.method+'__ev__pts'][a[rr.method+'__ev__valid']]
            calc=independent_surface(pts,rr.surface)
            diff=max(max(abs(v['Ctau']-calc[v['d_mm'],v['tau_mm']][0]),abs(v['Cvalid']-calc[v['d_mm'],v['tau_mm']][1])) for v in meta['sensitivity'])
            C.require(diff<1e-10,'Final independent cloud coverage differs')
            samples.append(dict(case_key=rr.case_key,maxdiff=diff))
    out=dict(ok=True,images=7920,new=4696,reused=unchanged,states=results.state.value_counts().to_dict(),pair_rows=6600,pair_maxdiff=max(delta),
        choices_independently_checked=checked,aggregate_maxdiff=maxmean,sensitivity_rows=len(sens),new_cloud_samples=samples,
        event_counts=events,all_new_packs_hash_unchanged=True,all_frozen_files_unchanged=True,freeze_id=fz['freeze_id'])
    C.verify_freeze();C.save(C.OUT/'final_audit.json',out);print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='final':audit_final()
    else:main()
