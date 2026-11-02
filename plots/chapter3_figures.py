# Draw thesis curves and point-cloud comparisons from saved numerical results.

from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from plot_io import configure
ROOT, OUT = configure('chapter3_figures')
P=ROOT/'research/project_Part1_v1.7'
S=P/'outputs/surface_revision'
WORK=OUT
FIG=OUT/'figures'
FIG.mkdir(parents=True,exist_ok=True)
SOURCE={}
def read(rel):
    p=S/rel; SOURCE[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p,low_memory=False)
SM={'B_curv':'EF','M10_curv':'M10','M24_curv':'M24'}
SURFS=['B_curv','M10_curv','M24_curv']; SIG=[0,20,40,60,80,100]
AV=read('s3_fe01/results/A_s12_views.csv'); AV=AV[AV.rule=='DF'].copy()
B=read('s3_fe01/results/B_size02_cases.csv'); B=B[B.rule=='DF'].copy()
SIZE=pd.concat([AV[AV.section=='S1'],B],ignore_index=True)
LAY=AV[AV.section=='S2'].copy()
assert len(SIZE)==1440 and SIZE.input_key.nunique()==1296
assert len(LAY)==2640 and LAY.input_key.nunique()==2640
SZ=SIZE.groupby(['regime','surface','size_mm','sigma','phase_label']).mean(numeric_only=True).reset_index()
LC=LAY.groupby(['layout','surface','sigma','phase_label']).mean(numeric_only=True).reset_index()
C=read('s3_fe01/results/C_s4_cases.csv'); C=C[C.rule=='DF'].copy()
M2=read('s4_real02/manifest.csv'); M2=M2[M2.family=='point']

cols=['image_key','strategy','block','surface','phase','sigma','layout','view_id']
V2=M2[cols].merge(C[C.group=='six_noise'][['image_key','glob_Ctau']],on='image_key',validate='many_to_one')
assert V2.glob_Ctau.notna().all()
R7=read('s4_real07/results.csv')
V7=R7[(R7.family=='point')&(R7.method=='D0')&(R7.record_type=='view')&(R7.domain=='glob')].copy()
assert not V7.duplicated(['surface','sigma','block','phase','arm']).any()
metric=C.set_index('image_key').glob_Ctau
V7['DF_Ctau']=V7.image_key.map(metric);V7['DF_S0']=V7.s0_image_key.map(metric)
assert V7[['DF_Ctau','DF_S0']].notna().all().all()
V7['delta']=V7.DF_Ctau-V7.DF_S0
NB=V7.groupby(['surface','sigma','block','arm']).delta.mean().unstack('arm')
NT=NB.groupby(['surface','sigma']).median()*100
NT['S2*−S1']=(NB['S2*']-NB['S1']).groupby(['surface','sigma']).median()*100
NEW=NT.reset_index()
COM=read('s5_align05/case_results.csv'); COM=COM[COM.front_end.isin(['DF','P4'])].copy()
assert len(COM)==3960
CP=COM.groupby(['surface','combo','sigma','phase_label']).mean(numeric_only=True).reset_index()
EF=read('s5_align06/paired_effects.csv');EF.sigma=pd.to_numeric(EF.sigma,errors='coerce')
EF=EF[(EF.front_end.isin(['DF','P4']))&(EF.metric=='glob_Ctau')&(EF.record_type=='interaction')].copy()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'savefig.dpi':175})
def save(fig,n):
    fig.savefig(FIG/f'Abb_3_{n:02d}.png',bbox_inches='tight',facecolor='white');plt.close(fig)
def lab(c):return str(c).replace('D_','').replace('_p',' / R').replace('r','r = ',1).replace('x','×').replace('.',',')
def plot_curves(df,group,configs,metrics,n,title):
    fig,axes=plt.subplots(len(metrics),3,figsize=(12,2.65*len(metrics)+1.0),squeeze=False,sharex=True)
    colors=plt.get_cmap('tab10')
    for j,sf in enumerate(SURFS):
      for k,c in enumerate(configs):
        sub=df[(df.surface==sf)&(df[group]==c)]
        for i,(col,ylabel,scale) in enumerate(metrics):
          g=sub.groupby('sigma')[col].agg(['mean','min','max']).reindex(SIG)*scale
          ax=axes[i,j];ax.plot(SIG,g['mean'],'o-',ms=3,lw=1.4,color=colors(k),label=(f'r = {c:g} mm' if group=='size_mm' else lab(c)))
          ax.fill_between(SIG,g['min'],g['max'],color=colors(k),alpha=.09,lw=0)
          ax.grid(alpha=.18);ax.set_xticks(SIG)
          if scale==100:ax.set_ylim(0,103)
          elif 'rmse' in col:ax.set_yscale('log')
          if j==0:ax.set_ylabel(ylabel)
      axes[0,j].set_title(SM[sf]);axes[-1,j].set_xlabel('Rauschstandardabweichung σ / Graustufen')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=min(5,len(configs)),fontsize=8)
    fig.suptitle(title+'\nBänder: Spannweite der Phasenmittel; DF',fontsize=12)
    fig.tight_layout(rect=(0,.11 if len(configs)>5 else .07,1,.91));save(fig,n)

def s4plot():
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
    for ax,sf in zip(axes,SURFS):
      for j,st in enumerate(['S0','S1','S2','S3','S4']):
        q=V2[(V2.surface==sf)&(V2.strategy==st)]
        if st=='S0': z=q.groupby('sigma').glob_Ctau.agg(['median','min','max'])
        else:
          b=q.groupby(['sigma','block']).glob_Ctau.median().reset_index()
          z=b.groupby('sigma').glob_Ctau.agg(['median','min','max'])
        z=z.reindex(SIG)*100
        ax.plot(SIG,z['median'],'o-' if st!='S4' else 's--',ms=3,lw=1.4,label=st)
        ax.fill_between(SIG,z['min'],z['max'],alpha=.07)
      ax.set(title=SM[sf],xlabel='σ / Graustufen',xticks=SIG,ylim=(0,100));ax.grid(alpha=.2)
    axes[0].set_ylabel('Cτ / %');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5)
    fig.suptitle('Positionsvariation; DF; j = 0,20; Median über Layoutblöcke')
    fig.tight_layout(rect=(0,.08,1,.91));save(fig,19)

def interaction():
    sets=['dots_parallel','dots_perp','lines','lines_6.3_8']
    assert set(sets).issubset(set(EF['set']))
    fig,axes=plt.subplots(len(sets),3,figsize=(12,2.5*len(sets)+.8),squeeze=False,sharex=True)
    labels={'dots_parallel':'Punkte: 6 → 10 mm\nr = 1 / 1,5 mm','dots_perp':'Punkte: 4 → 6 mm\nr = 1 / 1,5 mm','lines':'Linien: 4,45 → 8 mm\nb = 0,5 / 1,5 mm','lines_6.3_8':'Linien: 6,3 → 8 mm\nb = 0,5 / 1,5 mm; Zusatzvergleich'}
    for i,se in enumerate(sets):
      for j,sf in enumerate(SURFS):
        ax=axes[i,j]
        for eff,col,labx in [('E_small','#2d70a6','kleines Merkmal'),('E_large','#c1483c','großes Merkmal'),('I','#202020','Interaktion')]:
          an='posthoc' if se=='lines_6.3_8' else 'registered'
          fe='DF' if se.startswith('dots_') else 'P4'
          d=EF[(EF['set']==se)&(EF.surface==sf)&(EF.effect==eff)&EF.sigma.isin(SIG)&EF.analysis.str.startswith(an)&(EF.front_end==fe)].sort_values('sigma')
          assert d.sigma.tolist()==SIG, (se,sf,eff)
          ax.plot(d.sigma,100*d['mean'],'o-',c=col,label=labx,ms=3)
          ax.fill_between(d.sigma,100*d.res_lo_5,100*d.res_hi_95,color=col,alpha=.1)
        ax.axhline(0,color='#888',lw=.7);ax.grid(alpha=.17);ax.set_xticks(SIG)
        if i==0:ax.set_title(SM[sf])
        if j==0:ax.set_ylabel(labels.get(se,se)+'\nΔCτ / Prozentpunkte')
        if i==len(sets)-1:ax.set_xlabel('σ / Graustufen')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3)
    fig.suptitle('Größe–Abstand-Interaktion; Punkte DF, Linien P4\nBänder: deskriptive 5–95-%-Blockintervalle')
    fig.tight_layout(rect=(0,.06,1,.93));save(fig,26)

def size_spacing_plot():

    configs=sorted(COM[COM.front_end=='DF'].combo.unique())
    rows=[('glob_Ctau','Cτ global / %',100),('loc_Ctau','Cτ lokal / %',100),
          ('glob_pt_rmse','Punkt-RMSE / mm',1),('sup_surf_rmse','gestützte Fläche: RMSE / mm',1)]
    fig,axes=plt.subplots(4,3,figsize=(12,12.0),squeeze=False,sharex=True)
    colors=plt.get_cmap('tab10')
    for j,sf in enumerate(SURFS):
      for k,c in enumerate(configs):
        raw=COM[(COM.front_end=='DF')&(COM.surface==sf)&(COM.combo==c)]
        phase=CP[(CP.surface==sf)&(CP.combo==c)]
        for i,(col,label,scale) in enumerate(rows):
          ax=axes[i,j]
          if scale==100:
            g=phase.groupby('sigma')[col].agg(['mean','min','max']).reindex(SIG)*100
            y=g['mean'];ax.fill_between(SIG,g['min'],g['max'],color=colors(k),alpha=.09,lw=0)
          else:
            g=raw.groupby('sigma')[col].agg(['median','count','size']).reindex(SIG)
            y=g['median']
          ax.plot(SIG,y,'o-',ms=3,lw=1.3,color=colors(k),label=lab(c))
          if scale==1:
            partial=(g['count']>0)&(g['count']<g['size'])
            ax.plot(np.array(SIG)[partial],y[partial],'o',ms=4,mfc='white',mec=colors(k))
            ax.set_yscale('log')
          else:ax.set_ylim(0,103)
          ax.grid(alpha=.18);ax.set_xticks(SIG)
          if j==0:ax.set_ylabel(label)
      axes[0,j].set_title(SM[sf]);axes[-1,j].set_xlabel('Rauschstandardabweichung σ / Graustufen')
    fig.suptitle('Größe und Abstand: neun Punktkonfigurationen; DF\nCτ: Phasenmittel und Spannweite; RMSE: Median definierter Fälle',fontsize=12)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,fontsize=8)
    fig.text(.5,.012,'Hohler Marker: RMSE nicht in allen Fällen definiert; fehlender Punkt: kein definierter Fall.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,.94));save(fig,24)

def curve6():
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for ax,reg,title in zip(axes,['resource_cap_11695','fixed_peak_1'],['Gemeinsame Obergrenzen','Spitzenwert A = 1']):
      q=SZ[SZ.regime==reg].groupby(['size_mm','sigma']).glob_Ctau.mean().unstack().reindex(index=[.5,.75,1,1.25,1.5],columns=SIG)*100
      for s in [20,40,60,80,100]:ax.plot(q.index,q[s],'o-',label=f'σ = {s}')
      ax.set(title=title,xlabel='Punktradius r / mm',xticks=q.index,ylim=(0,100));ax.grid(alpha=.18)
    axes[0].set_ylabel('Cτ / %; Mittel der drei Flächen');fig.legend(*axes[0].get_legend_handles_labels(),ncol=5,loc='lower center')
    fig.suptitle('Der zulässige Grauwertaufwand verschiebt den günstigen Radius; DF')
    fig.tight_layout(rect=(0,.09,1,.91));save(fig,6)

def loadpts(row,mode):
    if mode=='fe':
      stage='size02' if row.size_mm in [.75,1.25] else 's12'
      p=S/f's3_fe01/{stage}/cache/{row.input_key}.npz';pref=''
    elif mode=='s4':p=S/f's3_fe01/s4/cache/{row.image_key}.npz';pref=''
    else:p=P/row.eval_record;pref=row.image_key+'__'
    SOURCE[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p,allow_pickle=False) as z:
      if pref+'pts' not in z:pref=''
      pts=z[pref+'pts'];v=z[pref+'valid'].astype(bool)&z[pref+'in_global'].astype(bool)
      return pts[v],z[pref+'zr'][v]

def cloud(rows,mode,n,title,labels):
    dom=S/'s1_real01/domain/global_domain_step0p50.npz'
    sf=rows[0].surface;zk={'B_curv':'z_B_curv','M10_curv':'z_A_curv','M24_curv':'z_C_curv'}[sf]
    with np.load(dom) as z:
      xyz=np.c_[z['x'],z['y'],z[zk]];xyz=xyz[np.isfinite(xyz).all(axis=1)]
    ptslist=[loadpts(r,mode) for r in rows]
    allz=np.concatenate([xyz[:,2]]+[p[:,2] for p,_ in ptslist]);lo,hi=float(allz.min()),float(allz.max())
    fig=plt.figure(figsize=(12,7.7));norm=Normalize(lo,hi);en=Normalize(-.5,.5)
    for i,((pts,zr),label) in enumerate(zip(ptslist,labels)):
      ax=fig.add_subplot(2,len(rows),i+1,projection='3d')
      thin=xyz[::35];ax.scatter(*thin.T,c='#90ab95',s=.6,alpha=.18)
      ax.scatter(*pts.T,c=pts[:,2],cmap='viridis',norm=norm,s=9,depthshade=False)
      ax.set(xlim=(-45,45),ylim=(-45,45),zlim=(lo,hi),xlabel='x / mm',ylabel='y / mm',zlabel='z / mm',title=label)
      ax.set_box_aspect((90,90,40));ax.view_init(25,-58)
      ax=fig.add_subplot(2,len(rows),len(rows)+i+1)
      ax.scatter(pts[:,0],pts[:,1],c=pts[:,2]-zr,cmap='RdBu_r',norm=en,s=12)
      ax.set(xlim=(-45,45),ylim=(-45,45),aspect='equal',xlabel='x / mm',ylabel='y / mm',title=f'n = {len(pts)}; RMSE = {np.sqrt(np.mean((pts[:,2]-zr)**2)):.3f} mm')
    fig.suptitle(title+'\nOben: Höhen und Referenz (z überhöht); unten: Höhenfehler, DF',fontsize=12)
    fig.subplots_adjust(top=.86,bottom=.08,left=.05,right=.90,wspace=.30,hspace=.26)
    fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),cax=fig.add_axes([.93,.56,.012,.26]),label='z / mm')
    fig.colorbar(ScalarMappable(norm=en,cmap='RdBu_r'),cax=fig.add_axes([.93,.13,.012,.26]),label='Δz / mm',extend='both')
    tail='; '.join(f'{label}: {int((np.abs(p[:,2]-zr)>.5).sum())} / {len(p)}' for (p,zr),label in zip(ptslist,labels))
    fig.text(.5,.02,'|Δz| > 0,5 mm (Farbsättigung, Punkte beibehalten): '+tail,ha='center',fontsize=8)
    save(fig,n)

def idealplot():
    t=read('thesis_check01/_parts/B_results.csv')
    q=t[(t.record_type=='summary')&(t.section=='S2')&(t.family=='point')&(t.domain=='glob')&(t.sigma==0)]
    if q.empty:
      print('ideal types',t.record_type.value_counts().to_dict())
      q=t[(t.section=='S2')&(t.family=='point')&(t.domain=='glob')&(t.sigma==0)&t.ref_C_tau_mean.notna()]
    assert not q.empty
    cfgs=['R6x4','R7.35x4','R8x4','R10x4','R12x4','R8x2.5','R8x3','R8x6','R8x8','T8x4']
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),sharey=True)
    for ax,sf in zip(axes,SURFS):
      d=q[q.surface==sf].groupby('layout').ref_C_tau_mean.mean().reindex(cfgs)*100
      x=np.arange(len(cfgs));ax.bar(x,d,color='#c4d2c1',label='ideale Abtastreferenz')
      for s,col in [(0,'#111111'),(40,'#bd4040')]:
        a=LC[(LC.surface==sf)&(LC.sigma==s)].groupby('layout').glob_Ctau.mean().reindex(cfgs)*100
        ax.plot(x,a,'o',c=col,label=f'DF, σ = {s}')
      ax.set(title=SM[sf],xticks=x,xticklabels=[c.replace('x','×') for c in cfgs],ylim=(0,105));ax.tick_params(axis='x',labelrotation=65);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Cτ / %');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3)
    fig.suptitle('Punktlayouts: ideale Abtastung und tatsächliche Rekonstruktion; d = 6 mm, τ = 0,15 mm')
    fig.tight_layout(rect=(0,.08,1,.92));save(fig,11)

def run():
    cov=[('glob_Ctau','Cτ / %',100),('glob_recovery','korrekt wiederhergestellt / %',100)]
    err=[('glob_pt_rmse','Punkt-RMSE / mm',1),('glob_surf_rmse','Flächen-RMSE / mm',1)]
    rad=[.5,.75,1,1.25,1.5]
    plot_curves(SZ[SZ.regime=='resource_cap_11695'],'size_mm',rad,cov,1,'Punktradius bei gemeinsamem Grauwertbudget und Spitzenwertlimit')
    plot_curves(SZ[SZ.regime=='resource_cap_11695'],'size_mm',rad,err,2,'Höhenfehler bei gemeinsamem Grauwertbudget und Spitzenwertlimit')
    plot_curves(SZ[SZ.regime=='fixed_peak_1'],'size_mm',rad,cov,4,'Punktradius bei festem Spitzenwert A = 1')
    curve6()
    met=[cov[0],*err,cov[1]]
    plot_curves(LC,'layout',['R6x4','R7.35x4','R8x4','R10x4','R12x4'],met,7,'Variation des Abstandes entlang der Epipolarrichtung')
    plot_curves(LC,'layout',['R8x2.5','R8x3','R8x4','R8x6','R8x8','T8x4'],met,8,'Querabstand und versetzte Anordnung')
    s4plot();interaction();idealplot()
    cfgs=sorted(COM[COM.front_end=='DF'].combo.unique())
    size_spacing_plot()
    plot_curves(CP,'combo',['D_r0.75_p8x4','D_r1_p8x4','D_r1_p8x6','D_r1.5_p8x4','D_r1.5_p8x6'],[('glob_recovery','korrekt wiederhergestellt / %',100),('glob_Ctau','Cτ / %',100)],27,'Merkmalwiederherstellung und präzise Flächenabdeckung')

    q=SIZE[(SIZE.surface=='M24_curv')&(SIZE.regime=='resource_cap_11695')&(SIZE.sigma==40)]
    ref=q[q.size_mm==1].sort_values(['phase_label','rep']).iloc[0]
    rows=[q[(q.size_mm==r)&(q.phase_label==ref.phase_label)&(q.rep==ref.rep)].iloc[0] for r in [.75,1,1.25]]
    cloud(rows,'fe',3,f'M24; σ = 40; {ref.phase_label}, Realisierung {int(ref.rep)}',[f'r = {r:g} mm' for r in [.75,1,1.25]])
    q=COM[(COM.front_end=='DF')&(COM.combo=='D_r0.75_p8x4')&(COM.surface=='M10_curv')&(COM.phase_label=='ph0')]
    rows=[q[(q.sigma==s)&(q.rep==(0 if s==0 else 1))].iloc[0] for s in [0,40,100]]
    cloud(rows,'combo',28,'M10; r = 0,75 mm; R8×4; Phase 0',[f'σ = {s}' for s in [0,40,100]])
    v=V7[(V7.surface=='M10_curv')&(V7.sigma==0)&(V7.arm=='S1')].sort_values(['delta','block','phase']).iloc[0]
    rows=[C[C.image_key==key].iloc[0] for key in [v.s0_image_key,v.image_key]]
    cloud(rows,'s4',21,f'M10; σ = 0; {v.block}, {v.phase}',['S0: regelmäßiges Raster','S1: freie Verschiebung'])
    NEW.to_csv(WORK/'s4_df_registered_views.csv',index=False)
    info={'sources':SOURCE,'new_figures':len(list(FIG.glob('Abb_3_*.png'))),'s4_view_rows':len(V7),'s4_shared_view_recovery':len(V7)-V7.image_key.nunique(),'size_rows':len(SIZE),'layout_rows':len(LAY),'combo_rows':len(COM),'case3':[{k:r[k] for k in ['size_mm','phase_label','rep','glob_Ctau','glob_n_valid','glob_pt_rmse']} for r in rows] if False else {'phase':ref.phase_label,'rep':int(ref.rep)},'s4_case21':{'block':v.block,'phase':v.phase,'Ctau0':v.DF_S0,'Ctau1':v.DF_Ctau}}
    (WORK/'data_checks.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(NEW.to_string(index=False));print('figures generated',info['new_figures'])

if __name__=='__main__':run()
