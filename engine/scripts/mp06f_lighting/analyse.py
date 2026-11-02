# Analyze illumination effects and generate candidate-selection figures.

from __future__ import annotations
import sys,json,itertools,sqlite3,time
from pathlib import Path
import lighting as L
import numpy as np
import pandas as pd
C=L.C;A=L.A;OUT=L.OUT
ENV_PAIRS=[('NOMINAL','UNIFORM'),('NOMINAL','G05'),('UNIFORM','GRAD_X'),('UNIFORM','GRAD_Y'),('UNIFORM','GRAD_Y_REV'),('GRAD_Y','GRAD_Y_REV')]

def environment_pairs(R):
    keys=['family','config','surface','sigma','phase_block','phase_u','phase_v','phase_frac','rep','seed']
    cols=keys+['case_key','glob_Ctau','glob_Cvalid','diag_noise_sha','diag_received_light_ratio','diag_signal_peak_noise_free_grey']
    pairs=[]
    for e0,e1 in ENV_PAIRS:
        a=R[R.env==e0];b=R[R.env==e1]
        p=a[cols].merge(b[cols],on=keys,validate='one_to_one',suffixes=('_reference','_condition'),how='outer',indicator=True)
        C.require(len(p)==1584 and p._merge.eq('both').all(),'Incomplete environment pairing')
        C.require(p.diag_noise_sha_reference.fillna('').eq(p.diag_noise_sha_condition.fillna('')).all(),'Different noise fields in environmental pair')
        p['reference_env']=e0;p['condition_env']=e1
        for k in ['Ctau','Cvalid']:p['delta_'+k]=p['glob_'+k+'_condition']-p['glob_'+k+'_reference']
        pairs.append(p.drop(columns='_merge'))
    return pd.concat(pairs,ignore_index=True)

def independent_decisions(R,T,S,choices):
    groups={k:g for k,g in R.groupby(['config','env','sigma'])}
    sens={k:g.set_index('case_key').Ctau for k,g in T.groupby(['d_mm','tau_mm'])}
    sv=S.set_index(['config','env','sigma','surface','d_mm','tau_mm']).Ctau.to_dict();diff=0.
    for r in choices.to_dict('records'):
        cfgs=C.DOTS if r['family']=='point' else C.LINES;good=[];scores={};errors={}
        for cfg in cfgs:
            q=groups[cfg,r['env'],r['sigma']];ss=sens[r['d_mm'],r['tau_mm']];means=[];safe=True
            for sf in C.SURFS:
                g=q[q.surface==sf];v=float(np.mean([ss.loc[g[g.phase_block==b].case_key].mean() for b in range(4)]));means.append(v)
                diff=max(diff,abs(v-sv[cfg,r['env'],r['sigma'],sf,r['d_mm'],r['tau_mm']]))
                safe &= bool(g.glob_pt_rmse.notna().all() and g.glob_pt_rmse.le(.3).all() and g.w.eq(0).all() and (g.c+g.w).gt(0).all() and (g.u/(g.c+g.w+g.u)).le(.05).all())
            scores[cfg]=min(means);errors[cfg]=q.glob_pt_rmse.max()
            if safe and min(means)>=r['floor']:good.append(cfg)
        C.require(sorted(good)==sorted(r['qualified'].split(';') if r['qualified'] else []),'Independent qualification differs')
        if good:
            best=max(scores[c] for c in good);near=sorted(c for c in good if scores[c]>=best-.02-1e-12)
            ref='D_r1_p8x6' if r['family']=='point' else 'L_w0.5_s8';pref=ref if ref in near else min(near,key=lambda c:(errors[c],c))
            C.require(near==r['near_optimal'].split(';') and pref==r['preferred'] and abs(scores[pref]-r['pref_score'])<1e-12,'Independent recommendation differs')
        else:C.require(pd.isna(r['preferred']),'Unqualified candidate selected')
    C.require(diff<1e-12,'Phase aggregation differs');return dict(choices_checked=len(choices),aggregate_maxdiff=diff)

def run():
    start=time.time();f,m=L.verify();R,T=L.collect(m,f);long,S,choices=A.tables(R,T,m)
    E=environment_pairs(R);P=A.paired_table(R)
    C.require(len(R)==9504 and len(T)==76032 and len(choices)==1152 and len(E)==9504 and len(P)==7920,'Final table cardinality')

    idx=R.set_index('case_key');diff=0.
    for q in E.itertuples():
        a=idx.loc[q.case_key_reference];b=idx.loc[q.case_key_condition]
        for k in ['family','config','surface','sigma','phase_block','rep','seed','phase_u','phase_v','phase_frac']:C.require(C.clean(a[k])==C.clean(b[k]),'Independent environment pair mismatch')
        diff=max(diff,abs(float(b.glob_Ctau-a.glob_Ctau)-q.delta_Ctau))
    for q in P.itertuples():
        a=idx.loc[q.case_key_left];b=idx.loc[q.case_key_right]
        for k in ['family','surface','env','sigma','phase_block','rep','seed','phase_u','phase_v','phase_frac']:C.require(C.clean(a[k])==C.clean(b[k]),'Independent candidate pair mismatch')
        C.require(a.diag_noise_sha==b.diag_noise_sha or (pd.isna(a.diag_noise_sha) and pd.isna(b.diag_noise_sha)),'Candidate noise mismatch')
        diff=max(diff,abs(float(b.glob_Ctau-a.glob_Ctau)-q.delta_Ctau))
    C.require(diff<1e-12,'Independent paired effect differs')
    check=independent_decisions(R,T,S,choices)
    Eblock=E.groupby(['family','config','surface','sigma','phase_block','reference_env','condition_env']).agg(delta_Ctau=('delta_Ctau','mean'),delta_Cvalid=('delta_Cvalid','mean')).reset_index()
    Esummary=Eblock.groupby(['family','config','sigma','reference_env','condition_env']).agg(delta=('delta_Ctau','mean'),block_min=('delta_Ctau','min'),block_max=('delta_Ctau','max'),delta_Cvalid=('delta_Cvalid','mean'),blocks=('phase_block','size')).reset_index()
    noise=R[R.sigma>0].groupby(['config','env','surface','phase_block','sigma'])
    C.require(noise.rep.apply(lambda x:set(x)=={1,2}).all() and noise.diag_noise_sha.nunique().eq(2).all(),'Noise repetition error')
    store=L.Store()
    with store.connect() as con:
        ledger=con.execute('SELECT ck,status,sha FROM cases').fetchall();events=con.execute('SELECT stage,status,count(*) FROM events GROUP BY stage,status').fetchall()
    C.require(len(ledger)==6554 and all(st=='complete' for _,st,_ in ledger),'Incomplete rendering ledger')
    C.require(sum(n for st,status,n in events if st=='render' and status=='started')==6554,'Unexpected render attempts')
    for ck,st,h in ledger:C.require(C.sha(store.path(ck))==h,'New pack changed')
    sys.path.insert(0,str(L.OLD));from audit_execution import independent_surface
    cloud=[]
    for fam,env,sigma in [('point','G05',20),('point','GRAD_Y_REV',60),('line','GRAD_Y',20),('line','GRAD_X',80)]:
        r=C.rowdict(m[(m.action=='new')&(m.family==fam)&(m.env==env)&(m.sigma==sigma)&(m.surface=='M24_curv')].sort_values('case_key').iloc[-1]);arr=L.B.load_arrays(r,store);p=r['method']+'__ev__'
        val=independent_surface(arr[p+'pts'][arr[p+'valid']],r['surface']);q=T[T.case_key==r['case_key']]
        d=max(max(abs(z.Ctau-val[z.d_mm,z.tau_mm][0]),abs(z.Cvalid-val[z.d_mm,z.tau_mm][1])) for z in q.itertuples())
        C.require(d<1e-10,'Independent node coverage differs');cloud.append(dict(case_key=r['case_key'],maxdiff=d))
    for name,t in [('results.csv',R),('sensitivity.csv.gz',T),('surfaces.csv',S),('choices.csv',choices),('environment_pairs.csv.gz',E),('candidate_pairs.csv.gz',P),('environment_effects.csv',Esummary)]:
        if name.endswith('.gz'):C.atomic(OUT/name,lambda p,t=t:t.to_csv(p,index=False,compression='gzip'))
        else:C.save_csv(OUT/name,t)
    audit=dict(ok=True,freeze_id=f['freeze_id'],images=len(R),new=6554,reuse=2950,sensitivity_rows=len(T),noise_pairs=len(noise),states=R.state.value_counts().to_dict(),
      pair_rows=len(E)+len(P),pair_maxdiff=diff,independent_cloud_checks=cloud,event_counts=events,unchanged_source_packs=2950,seconds=time.time()-start,**check)
    if (OUT/'validation_fix.json').exists():
        fix=C.load(OUT/'validation_fix.json');C.verify_files(fix['files'])
        audit['validation_correction']=dict(record_sha256=C.sha(OUT/'validation_fix.json'),new_imaging_due_to_repair=0,one_uncommitted_case_redecoded_from_saved_image=True,scientific_numbers_or_policy_changed=False)
    L.verify();C.save(OUT/'audit.json',audit)
    print(C.canon(audit));print(choices[(choices.d_mm==6)&(choices.tau_mm==.15)&(choices.floor==.5)][['family','env','sigma','qualified','preferred','pref_score']].to_string(index=False))

# Generate illumination curves and candidate-selection comparisons from saved tables.
def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,Normalize
    from matplotlib.cm import ScalarMappable
    R=C.csv(OUT/'results.csv');S=C.csv(OUT/'surfaces.csv');choices=C.csv(OUT/'choices.csv');m=C.csv(OUT/'manifest.csv');st=L.Store()
    root=C.writable(OUT/'figures/anchor').parent
    fig,axes=plt.subplots(2,2,figsize=(17,11))
    for i,fam in enumerate(['point','line']):
        for j,d in enumerate([5.,6.]):
            ax=axes[i,j];q=choices[(choices.family==fam)&(choices.d_mm==d)&(choices.tau_mm==.15)&(choices.floor==.5)]
            mat=np.zeros((6,6));txt=np.empty((6,6),dtype=object)
            for r in q.to_dict('records'):
                y=L.ENVS.index(r['env']);x=C.SIGMAS.index(int(r['sigma']));ok=pd.notna(r['preferred']);mat[y,x]=ok;txt[y,x]=A.label(r['preferred'])+f'\n{r["pref_score"]*100:.1f} %' if ok else 'kein zulässiger\nKandidat'
            ax.imshow(mat,vmin=0,vmax=1,cmap=ListedColormap(['#e8b4ac','#b3d9b4']),aspect='auto')
            for y,x in itertools.product(range(6),range(6)):ax.text(x,y,txt[y,x],ha='center',va='center',fontsize=7.5)
            ax.set_xticks(range(6),C.SIGMAS);ax.set_yticks(range(6),L.LABELS);ax.set_xlabel('Rausch-σ / Graustufen');ax.set_title(('Punkte' if fam=='point' else 'Linien')+f'; d = {d:g} mm')
    fig.suptitle('Beleuchtung: zulässige Auswahl bei τ = 0,15 mm und Cτ ≥ 50 %',fontsize=15)
    fig.text(.5,.025,'Zahl: schlechtester Oberflächenmittelwert des ausgewählten Kandidaten. Zusätzlich: Punkt-RMSE ≤ 0,30 mm, keine bestätigte Fehlidentität, u ≤ 5 %.\nPSF = 1 px; gleiche Kandidaten, vier Phasen je Oberfläche. Ø / Abstand bzw. Breite / Linienabstand in mm.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.95]);fig.savefig(root/'Auswahl.png',dpi=155);plt.close(fig)
    main=S[(S.d_mm==6)&(S.tau_mm==.15)]
    for fam,cfgs in [('point',C.DOTS),('line',C.LINES)]:
        fig,axes=plt.subplots(2,6,figsize=(21,8),sharex=True)
        for j,e in enumerate(L.ENVS):
            for cfg in cfgs:
                g=main[(main.config==cfg)&(main.env==e)].groupby('sigma').agg(ct=('Ctau','mean'),lo=('Ctau','min'),hi=('Ctau','max'),cv=('Cvalid','mean')).reindex(C.SIGMAS)
                axes[0,j].plot(C.SIGMAS,g.ct,'o-',ms=3,label=A.label(cfg));axes[0,j].fill_between(C.SIGMAS,g.lo,g.hi,alpha=.08)
                axes[1,j].plot(C.SIGMAS,g.cv,'o-',ms=3)
            axes[0,j].set_title(L.LABELS[j]);axes[0,j].axhline(.5,ls='--',c='gray',lw=.7)
            axes[1,j].set_xlabel('Rausch-σ');axes[1,j].set_xticks(C.SIGMAS)
            for ax in axes[:,j]:ax.set_ylim(0,1);ax.grid(alpha=.2)
        axes[0,0].set_ylabel('Genauigkeitsabdeckung Cτ');axes[1,0].set_ylabel('Stützabdeckung Cvalid')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=9,bbox_to_anchor=(.5,.005))
        fig.suptitle(('Punktmuster' if fam=='point' else 'Linienmuster')+': vollständige Beleuchtungs- und Rauschreihe')
        fig.text(.5,.12,'τ = 0,15 mm; d = 6 mm. Linien: Mittel der drei Oberflächen; Bänder: Oberflächenbereich. σ0 als eigene Bedingung.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.17,1,.95]);fig.savefig(root/f'Kurven_{fam}.png',dpi=145);plt.close(fig)

    for fam,cfg in [('point','D_r1_p8x6'),('line','L_w0.5_s8')]:
        fig=plt.figure(figsize=(13,12))
        for i,env in enumerate(['NOMINAL','G05','GRAD_Y','GRAD_Y_REV']):
            r=C.rowdict(m[(m.config==cfg)&(m.env==env)&(m.surface=='M24_curv')&(m.phase_block==1)&(m.sigma==40)&(m.rep==1)].iloc[0]);a=L.B.load_arrays(r,st);p=r['method']+'__ev__';v=a[p+'valid']&a[p+'in_global'];pts=a[p+'pts'][v];err=pts[:,2]-a[p+'zr'][v];res=R[R.case_key==r['case_key']].iloc[0]
            ax=fig.add_subplot(4,3,3*i+1);ax.imshow(a['image'],cmap='gray',vmin=0,vmax=255);ax.axis('off');ax.set_title(L.LABELS[L.ENVS.index(env)]+f'; Cτ = {res.glob_Ctau:.3f}',fontsize=10)
            ax=fig.add_subplot(4,3,3*i+2,projection='3d')
            if len(pts):ax.scatter(*pts.T,c=pts[:,2],s=1,cmap='viridis',vmin=-16,vmax=4)
            else:ax.text2D(.1,.5,'Keine gültigen Punkte',transform=ax.transAxes)
            ax.set(xlim=(-46,46),ylim=(-46,46),zlim=(-16,4),xlabel='x / mm',ylabel='y / mm');ax.view_init(25,-65);ax.set_title(f'n = {len(pts)}; z / mm',fontsize=9);ax.tick_params(labelsize=7)
            ax=fig.add_subplot(4,3,3*i+3)
            if len(pts):ax.scatter(pts[:,0],pts[:,1],c=err,cmap='coolwarm',vmin=-.3,vmax=.3,s=3 if fam=='point' else 1)
            else:ax.text(.1,.5,'Keine gültigen Punkte',transform=ax.transAxes)
            ax.set(xlim=(-46,46),ylim=(-46,46),xlabel='x / mm',ylabel='y / mm');ax.set_aspect('equal');ax.set_title('Höhenfehler / mm',fontsize=9)
        fig.suptitle(f'{A.label(cfg)} mm — M24, Phase 1, Wiederholung 1, Rausch-σ 40')
        fig.tight_layout(rect=[0,.1,1,.95],w_pad=3,h_pad=2)
        for rect,cmap,lim,lab in [([.15,.045,.25,.012],'viridis',(-16,4),'Höhe z / mm'),([.6,.045,.25,.012],'coolwarm',(-.3,.3),'Höhenfehler / mm')]:
            ax=fig.add_axes(rect);fig.colorbar(ScalarMappable(norm=Normalize(*lim),cmap=cmap),cax=ax,orientation='horizontal',extend='both' if cmap=='coolwarm' else 'neither').set_label(lab)
        fig.savefig(root/f'Punktwolken_{fam}.png',dpi=145);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    for ax,cfg,fam in zip(axes,['D_r1_p8x6','L_w0.5_s8'],['Punkte','Linien']):
        q=main[(main.config==cfg)&(main.sigma==40)].groupby('env')[['Ctau','Cvalid']].mean().reindex(L.ENVS)
        x=np.arange(6);a=q.Ctau.to_numpy();b=q.Cvalid.to_numpy()-a;c=1-q.Cvalid.to_numpy()
        ax.bar(x,a,color='#52916a',label='genau');ax.bar(x,b,bottom=a,color='#e4b155',label='gestützt, Fehler > τ');ax.bar(x,c,bottom=a+b,color='#bdc2c7',label='ohne Stützung')
        ax.set_xticks(x,L.LABELS,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_title(fam+': '+A.label(cfg)+' mm');ax.set_ylabel('Anteil der 30 947 Auswerteknoten')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3);fig.suptitle('Wo geht Abdeckung verloren? Rausch-σ 40, τ = 0,15 mm, d = 6 mm')
    fig.tight_layout(rect=[0,.1,1,.92]);fig.savefig(root/'Verlustanteile.png',dpi=160);plt.close(fig)
    print('Six German scientific figures saved.')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='figures':figures()
    else:run()
