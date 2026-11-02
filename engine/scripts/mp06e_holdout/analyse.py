# Evaluate candidate recommendations on the selected new-phase blur cases.

from __future__ import annotations
import sys,json,itertools
from pathlib import Path
import holdout as H
import numpy as np
import pandas as pd
C=H.C;OUT=H.OUT
import analysis as A

# Evaluate candidate feasibility and near-optimal membership on held-out blur cases.
def analyze():
 f,m=H.verify();R,T=H.collect(m,f);long,S,choices=A.tables(R,T,m)
 key=['family','surface','env','sigma','phase_block','phase_u','phase_v','phase_frac','rep','seed']
 effects=[];summaries=[];surfaces=[]
 for sc in H.SCENARIOS:
  ss=S[(S.env==sc['env'])&(S.sigma==sc['sigma'])&(S.family==sc['family'])&(S.d_mm==sc['d'])&(S.tau_mm==.15)].copy()
  ss['scenario']=sc['id'];surfaces.append(ss)
  score={cfg:float(g.Ctau.min()) for cfg,g in ss.groupby('config')}
  qual={cfg:bool(len(g)==3 and g.safe.all() and g.Ctau.ge(.5).all()) for cfg,g in ss.groupby('config')}
  good=[cfg for cfg in sc['configs'] if qual[cfg]]
  best=max(score[cfg] for cfg in good) if good else None
  rec=sc['recommended'];ref=sc['reference']
  comparisons=[cfg for cfg in sc['configs'] if cfg!=ref]
  per_comparison={}
  for cfg in comparisons:
   a=R[(R.config==ref)&(R.env==sc['env'])&(R.sigma==sc['sigma'])]
   b=R[(R.config==cfg)&(R.env==sc['env'])&(R.sigma==sc['sigma'])]
   q=a[key+['case_key','diag_noise_sha']].merge(b[key+['case_key','diag_noise_sha']],on=key,suffixes=('_reference','_candidate'),validate='one_to_one')
   C.require(len(q)==len(a)==len(b),'Holdout pairing incomplete')
   C.require((q.diag_noise_sha_reference.fillna('')==q.diag_noise_sha_candidate.fillna('')).all(),'Not common noise field across candidate pair')
   lookup=T[(T.d_mm==sc['d'])&(T.tau_mm==.15)].set_index('case_key').Ctau
   q['delta']=q.case_key_candidate.map(lookup)-q.case_key_reference.map(lookup)
   q['reference_ct']=q.case_key_reference.map(lookup);q['candidate_ct']=q.case_key_candidate.map(lookup)
   block=q.groupby(['surface','phase_block']).agg(delta=('delta','mean'),reference_ct=('reference_ct','mean'),candidate_ct=('candidate_ct','mean')).reset_index()
   means=block.groupby('surface').delta.mean()
   per_comparison[cfg]=dict(mean=float(means.mean()),min_surface=float(means.min()),min_phase=float(block.delta.min()),max_phase=float(block.delta.max()),positive_blocks=int((block.delta>0).sum()),n_blocks=len(block))
   block['scenario']=sc['id'];block['reference']=ref;block['candidate']=cfg;effects.append(block)
  for cfg in sc['configs']:
   g=ss[ss.config==cfg];delta=per_comparison.get(cfg,{})
   old=C.csv(H.OLD/'analysis/surfaces.csv')
   old=old[(old.config==cfg)&(old.env==sc['env'])&(old.sigma==sc['sigma'])&(old.d_mm==sc['d'])&(old.tau_mm==.15)]
   summaries.append(dict(scenario=sc['id'],family=sc['family'],env=sc['env'],sigma=sc['sigma'],d=sc['d'],config=cfg,
     preregistered_recommendation=cfg==rec,qualified=qual[cfg],score=score[cfg],development_score=float(old.Ctau.min()),
     near_best_tested=(best is not None and qual[cfg] and score[cfg]>=best-.02-1e-12),
     phase_min=float(g.phase_min.min()),phase_max=float(g.phase_max.max()),wrong=int(g.wrong.sum()),unconfirmed=int(g.unconfirmed.sum()),
     rmse_max=float(g.rmse_max.max()),u_max=float(g.u_max.max()),n_images=int(g.n_images.sum()),
     positive_surface_direction=(delta.get('min_surface',0)>0 if cfg!=ref else None),**delta))
 V=pd.DataFrame(summaries);E=pd.concat(effects,ignore_index=True);SS=pd.concat(surfaces,ignore_index=True)

 store=C.Store(OUT);maxerr=0.;hashes={}
 for r in m.to_dict('records'):
  meta=store.resume(C.rowdict(r),f['freeze_id']);path=store.path(r['case_key']);hashes[r['case_key']]=C.sha(path)
  arr=H.B.load_arrays(r,store);p=r['method']+'__ev__';v=arr[p+'valid']&arr[p+'in_global']
  if v.any():
   e=arr[p+'pts'][v,2].astype(float)-arr[p+'zr'][v].astype(float)
   maxerr=max(maxerr,abs(float(np.sqrt(np.dot(e,e)/len(e)))-meta['result']['glob_pt_rmse']))
 C.require(maxerr<2e-6,'Independent point RMSE mismatch')
 sys.path.insert(0,str(H.OLD));from audit_execution import independent_surface
 checks=[]
 for fam,sig in [('point',40),('point',60),('line',0),('line',20)]:
  r=C.rowdict(m[(m.family==fam)&(m.sigma==sig)&(m.surface=='M24_curv')].sort_values('case_key').iloc[0])
  arr=H.B.load_arrays(r,store);p=r['method']+'__ev__';v=independent_surface(arr[p+'pts'][arr[p+'valid']],r['surface'])
  q=T[T.case_key==r['case_key']];err=max(max(abs(t.Ctau-v[t.d_mm,t.tau_mm][0]),abs(t.Cvalid-v[t.d_mm,t.tau_mm][1])) for t in q.itertuples())
  C.require(err<1e-10,'Independent surface evaluation mismatch');checks.append(dict(case_key=r['case_key'],maxdiff=err))

 ix=R.set_index('case_key');verified=0
 for row in V.to_dict('records'):
  q=m[(m.scenario==row['scenario'])&(m.config==row['config'])];lookup=T[(T.d_mm==row['d'])&(T.tau_mm==.15)].set_index('case_key')
  means=[];safe=True
  for sf in C.SURFS:
   qs=q[q.surface==sf];means.append(np.mean([lookup.loc[qs[qs.phase_block==b].case_key,'Ctau'].mean() for b in sorted(qs.phase_block.unique())]))
   for ck in qs.case_key:
    z=C.rowdict(ix.loc[ck]);safe &= z['glob_pt_rmse'] is not None and z['glob_pt_rmse']<=.3 and z['w']==0 and z['c']+z['w']>0 and z['u']/(z['c']+z['w']+z['u'])<=.05
  C.require(abs(min(means)-row['score'])<1e-12 and bool(safe and min(means)>=.5)==row['qualified'],'Independent qualification mismatch');verified+=1
 noisy=R[R.sigma>0].groupby(['config','env','surface','phase_block','sigma'])
 C.require(noisy.rep.apply(lambda q:set(q)=={1,2}).all() and noisy.diag_noise_sha.nunique().eq(2).all(),'Repeated or missing noise repetitions')
 for name,tab in [('results.csv',R),('sensitivity.csv.gz',long),('validation.csv',V),('surface_validation.csv',SS),('paired_blocks.csv',E),('sensitivity_choices.csv',choices)]:
  if name.endswith('.gz'):C.atomic(OUT/name,lambda p:tab.to_csv(p,index=False,compression='gzip'))
  else:C.save_csv(OUT/name,tab)
 C.save(OUT/'audit.json',dict(ok=True,freeze_id=f['freeze_id'],cases=len(R),states=R.state.value_counts().to_dict(),noise_pairs=len(noisy),point_rmse_maxdiff=maxerr,
    independent_coverage=checks,independent_qualifications=verified,pack_hashes=hashes,render_cap=288,rerun_previous_data=0))
 figures(V,E,m,store);H.verify()
 print(V[['scenario','config','qualified','near_best_tested','score','development_score','phase_min','mean']].to_string(index=False))

def figures(V,E,m,store):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(14,8));palette=['#376d9e','#d58026','#629a64']
 for ax,sc in zip(axes.flat,H.SCENARIOS):
  q=V[V.scenario==sc['id']]
  for i,r in enumerate(q.itertuples()):
   ax.bar(i-.16,r.development_score,.3,color='#b9c4cf');ax.bar(i+.16,r.score,.3,color=palette[i]);ax.text(i+.16,r.score+.025,f'{r.score:.3f}',ha='center',fontsize=8)
  ax.axhline(.5,color='black',ls='--',lw=1);ax.set_ylim(0,1);ax.set_xticks(range(len(q)),[A.label(c) for c in q.config],rotation=15,fontsize=8)
  ax.set_title(f'{sc["id"]}: PSF {C.ENVS[sc["env"]]:g}, Rausch-σ {sc["sigma"]}, d {sc["d"]:g}',fontsize=10)
  ax.set_ylabel('Cτ: schlechtester Oberflächenmittelwert');ax.grid(axis='y',alpha=.2)
 fig.suptitle('Neue Phasen: Entwicklung (grau) und Bestätigung (farbig)')
 fig.text(.5,.02,'τ = 0,15 mm. 50 % allein reicht nicht: Fehler- und Identitätsbedingungen gelten zusätzlich. Keine neue Kandidatenwahl.',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(OUT/'Validierung.png',dpi=160);plt.close(fig)

 selected=[('H1','L_w1_s8'),('H1','L_w1.25_s8'),('H3','D_r1_p8x4'),('H3','D_r1_p8x6')]
 fig=plt.figure(figsize=(13,12))
 for i,(sid,cfg) in enumerate(selected):
  r=C.rowdict(m[(m.scenario==sid)&(m.config==cfg)&(m.surface=='M24_curv')&(m.phase_block=='E0')&(m.rep==1)].iloc[0]);arr=H.B.load_arrays(r,store);p=r['method']+'__ev__';v=arr[p+'valid']&arr[p+'in_global'];pts=arr[p+'pts'][v];err=pts[:,2]-arr[p+'zr'][v]
  ax=fig.add_subplot(4,3,3*i+1);ax.imshow(arr['image'],cmap='gray',vmin=0,vmax=255);ax.axis('off');ax.set_title(f'{sid}: {A.label(cfg)} mm',fontsize=10)
  ax=fig.add_subplot(4,3,3*i+2,projection='3d')
  if len(pts):ax.scatter(*pts.T,c=pts[:,2],cmap='viridis',vmin=-16,vmax=4,s=1)
  else:ax.text2D(.1,.5,'Keine gültigen Punkte',transform=ax.transAxes)
  ax.set(xlim=(-46,46),ylim=(-46,46),zlim=(-16,4),xlabel='x / mm',ylabel='y / mm');ax.view_init(25,-65);ax.set_title(f'n={len(pts)}; z / mm',fontsize=9);ax.tick_params(labelsize=7)
  ax=fig.add_subplot(4,3,3*i+3)
  if len(pts):ax.scatter(pts[:,0],pts[:,1],c=err,cmap='coolwarm',vmin=-.3,vmax=.3,s=2)
  ax.set(xlim=(-46,46),ylim=(-46,46),xlabel='x / mm',ylabel='y / mm');ax.set_aspect('equal');ax.set_title('Höhenfehler: blau −0,3 / rot +0,3 mm',fontsize=8)
 fig.suptitle('Feste Fallauswahl: M24, neue Phase E0, Wiederholung 1')
 fig.tight_layout(rect=[0,.02,1,.96],w_pad=3,h_pad=2);fig.savefig(OUT/'Punktwolken.png',dpi=150);plt.close(fig)

if __name__=='__main__':analyze()
