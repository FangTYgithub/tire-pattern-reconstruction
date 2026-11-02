# Summarize blur results, paired differences and parameter-selection tables.

from __future__ import annotations
import numpy as np
import pandas as pd
import core as C

def label(cfg):
    if cfg.startswith('D_r'):
        r,p=cfg[3:].split('_p');return f'Ø{2*float(r):g} / '+p.replace('x','×')
    w,s=cfg[3:].split('_s');return f'w{float(w):g} / s{s}'

def sensitivity_gate(t,records):
    need={'case_key','d_mm','tau_mm','Ctau','Cvalid','n_nodes','sup_n'}
    C.require(need.issubset(t.columns),'Missing sensitivity fields')
    C.require(not t.duplicated(['case_key','d_mm','tau_mm']).any(),'Duplicate threshold row')
    want={(ck,d,tau) for ck in records.case_key for d in C.DS for tau in C.TAUS}
    C.require(set(t[['case_key','d_mm','tau_mm']].itertuples(index=False,name=None))==want,'Missing sensitivity combinations')
    C.require(t.n_nodes.eq(30947).all() and t.sup_n.ge(0).all() and t.sup_n.le(30947).all(),'Wrong fixed global denominator')
    C.require(t[['Ctau','Cvalid']].notna().all().all() and t.Ctau.ge(0).all() and (t.Ctau<=t.Cvalid+1e-12).all() and t.Cvalid.le(1).all(),'Invalid coverage sensitivity')
    C.require(np.allclose(t.Cvalid,t.sup_n/30947,rtol=0,atol=1e-12),'Support numerator inconsistent')

    p=t.pivot(index=['case_key','d_mm'],columns='tau_mm',values='Ctau')
    C.require((np.diff(p.to_numpy(),axis=1)>=-1e-12).all(),'Non-monotone tau coverage')
    p=t.pivot(index=['case_key','tau_mm'],columns='d_mm',values='Ctau')
    C.require((p[6]>=p[5]-1e-12).all(),'Non-monotone support coverage')
    main=t[(t.d_mm==6)&(t.tau_mm==.15)].merge(records[['case_key','glob_Ctau','glob_Cvalid']],on='case_key',validate='one_to_one')
    C.require(np.allclose(main.Ctau,main.glob_Ctau,atol=1e-10,rtol=0) and np.allclose(main.Cvalid,main.glob_Cvalid,atol=1e-10,rtol=0),'Main threshold metric drift')

def tables(records,sensitivity,expected):
    C.metric_gate(records,expected);sensitivity_gate(sensitivity,records)
    t=sensitivity.merge(records,on='case_key',validate='many_to_one')
    t['unsupported_fraction']=1-t.Cvalid
    t['supported_over_tolerance_fraction']=t.Cvalid-t.Ctau
    t['accurate_fraction']=t.Ctau
    t['u_share']=t.u/(t.c+t.w+t.u).replace(0,np.nan)
    t['w_share']=t.w/(t.c+t.w).replace(0,np.nan)
    t['exposure']=t.c+t.w
    keys=['family','config','env','sigma','surface','d_mm','tau_mm']
    phase=t.groupby(keys+['phase_block'],dropna=False).agg(Ctau=('Ctau','mean'),Cvalid=('Cvalid','mean'),point_rmse=('glob_pt_rmse','mean')).reset_index()
    surf=phase.groupby(keys).agg(Ctau=('Ctau','mean'),Cvalid=('Cvalid','mean'),point_rmse=('point_rmse','mean'),phase_min=('Ctau','min'),phase_max=('Ctau','max'),n_phases=('phase_block','nunique')).reset_index()
    risk=t.groupby(keys).agg(n_images=('case_key','size'),n_no_output=('state',lambda x:int(x.eq('no_output').sum())),rmse_n=('glob_pt_rmse','count'),rmse_max=('glob_pt_rmse','max'),wrong=('w','sum'),correct=('c','sum'),unconfirmed=('u','sum'),u_max=('u_share','max'),u_n=('u_share','count'),min_exposure=('exposure','min')).reset_index()
    surf=surf.merge(risk,on=keys,validate='one_to_one')
    surf['safe']=(surf.n_phases==4)&(surf.rmse_n==surf.n_images)&(surf.rmse_max<=.3)&(surf.wrong==0)&(surf.u_n==surf.n_images)&(surf.u_max<=.05)&(surf.min_exposure>0)
    rows=[]
    for (fam,e,s,d,tau),g in surf.groupby(['family','env','sigma','d_mm','tau_mm']):
        for floor in [.5,.7]:
            candidates=[]
            for cfg,q in g.groupby('config'):
                candidates.append(dict(config=cfg,qualified=len(q)==3 and q.safe.all() and q.Ctau.ge(floor).all(),score=q.Ctau.min(),rmse=q.rmse_max.max()))
            ca=pd.DataFrame(candidates);good=ca[ca.qualified];pref=None;best=None;near=[];score=None
            if len(good):
                best=float(good.score.max());n=good[good.score>=best-.02-1e-12];near=sorted(n.config)
                ref='D_r1_p8x6' if fam=='point' else 'L_w0.5_s8'
                pref=ref if ref in near else n.sort_values(['rmse','config']).iloc[0].config
                score=float(ca[ca.config==pref].score.iloc[0])
            rows.append(dict(family=fam,env=e,sigma=s,d_mm=d,tau_mm=tau,floor=floor,n_configs=len(ca),qualified=';'.join(sorted(good.config)),near_optimal=';'.join(near),preferred=pref,best_score=best,pref_score=score,status='qualified' if pref else 'no_qualified_candidate'))
    return t,surf,pd.DataFrame(rows)

def paired(records,left,right):

    keys=['family','surface','env','sigma','phase_block','phase_u','phase_v','phase_frac','rep','seed']
    a=records[records.config==left];b=records[records.config==right]
    C.require(len(a)>0 and len(a)==len(b),'Unequal paired sample sets')
    q=a[keys+['case_key','glob_Ctau','glob_Cvalid']].merge(b[keys+['case_key','glob_Ctau','glob_Cvalid']],on=keys,suffixes=('_left','_right'),validate='one_to_one',how='outer',indicator=True)
    C.require(q._merge.eq('both').all(),'Unmatched physical phase/rep/seed')
    q['delta_Ctau']=q.glob_Ctau_right-q.glob_Ctau_left
    q['delta_Cvalid']=q.glob_Cvalid_right-q.glob_Cvalid_left
    q['left_config']=left;q['right_config']=right
    return q.drop(columns='_merge')

def paired_table(records):
    pairs=[('L_w0.5_s8',x) for x in C.LINES[1:]]+[('D_r1_p8x6',x) for x in C.DOTS if x!='D_r1_p8x6']
    return pd.concat([paired(records,a,b) for a,b in pairs],ignore_index=True)

def figures(surf,choices,records,root,drill=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    root=C.writable(root/'anchor').parent
    suffix=' — ISOLIERTER TEST, KEINE FORSCHUNGSERGEBNISSE' if drill else ''
    paths=[]
    main=surf[(surf.d_mm==6)&(surf.tau_mm==.15)]
    for fam,cfgs in [('point',C.DOTS),('line',C.LINES)]:
        fig,axes=plt.subplots(2,5,figsize=(18,7),sharex=True)
        for j,e in enumerate(C.ENVS):
            for cfg in cfgs:
                g=main[(main.family==fam)&(main.env==e)&(main.config==cfg)]
                g=g.groupby('sigma').agg(ct=('Ctau','mean'),lo=('Ctau','min'),hi=('Ctau','max'),rmse=('point_rmse','mean')).reindex(C.SIGMAS)
                axes[0,j].plot(C.SIGMAS,g.ct,'o-',ms=3,label=label(cfg)+' mm')
                axes[0,j].fill_between(C.SIGMAS,g.lo,g.hi,alpha=.07)
                axes[1,j].plot(C.SIGMAS,g.rmse,'o-',ms=3)
            axes[0,j].set_title(f'PSF σ = {C.ENVS[e]:g} px');axes[0,j].set_ylim(0,1)
            axes[1,j].set_xlabel('Rausch-σ / Graustufen');axes[1,j].set_ylim(bottom=0)
            axes[0,j].grid(alpha=.2);axes[1,j].grid(alpha=.2)
        axes[0,0].set_ylabel('Cτ, global; Mittel über 3 Oberflächen')
        axes[1,0].set_ylabel('Punkt-RMSE / mm\n(nur gültige Punkte)')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.005),ncol=4,fontsize=8)
        fig.suptitle(('Punktmuster' if fam=='point' else 'Linienmuster')+': vollständiges Unschärfemodul'+suffix)
        fig.text(.5,.115,'Bänder: Bereich der drei Oberflächen, kein Konfidenzintervall. σ0 separat; Verbindungslinien sind keine Interpolation.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.18,1,.94]);p=root/f'Kurven_{fam}.png';fig.savefig(p,dpi=150);plt.close(fig);paths.append(str(p))
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for i,fam in enumerate(['point','line']):
        for j,d in enumerate([5.,6.]):
            ax=axes[i,j];p=choices[(choices.family==fam)&(choices.d_mm==d)&(choices.tau_mm==.15)&(choices.floor==.5)]
            mat=np.full((5,6),np.nan);txt=np.full((5,6),'nicht gemessen',dtype=object)
            for r in p.to_dict('records'):
                y=list(C.ENVS).index(r['env']);x=C.SIGMAS.index(int(r['sigma']));ok=pd.notna(r['preferred'])
                mat[y,x]=1 if ok else 0
                txt[y,x]=(label(r['preferred'])+f'\n{r["pref_score"]:.3f}') if ok else 'kein zulässiger\nKandidat'
            cmap=ListedColormap(['#e8b4ac','#b3d9b4']);cmap.set_bad('#dddddd');ax.imshow(mat,vmin=0,vmax=1,cmap=cmap,aspect='auto')
            for y in range(5):
                for x in range(6):ax.text(x,y,txt[y,x],ha='center',va='center',fontsize=7)
            ax.set_xticks(range(6),C.SIGMAS);ax.set_yticks(range(5),list(C.ENVS.values()));ax.set_xlabel('Rausch-σ');ax.set_ylabel('PSF-σ / px');ax.set_title(f'{fam}, d = {d:g} mm')
    fig.suptitle('Zulässigkeit und ausgewählter Kandidat; τ = 0,15 mm'+suffix)
    fig.text(.5,.02,'Alle drei Oberflächen + Fehler-/Identitätsbedingungen. Gezeigt: eigene Bewertung des ausgewählten Kandidaten; keine kontinuierliche Optimierung.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.94]);p=root/'Auswahl.png';fig.savefig(p,dpi=150);plt.close(fig);paths.append(str(p))
    return paths

def cloud_figures(man,root,array_reader,drill=False):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    suffix=' — TEST-ZUORDNUNG, KEINE FORSCHUNGSERGEBNISSE' if drill else ''
    paths=[]
    cases=[('BLUR2',20),('BLUR3',40),('BLUR3',100)]
    for fam,configs in [('point',['D_r1_p8x6','D_r1_p8x4']),('line',['L_w0.5_s8','L_w1_s8'])]:
        packets=[]
        for e,s in cases:
            for cfg in configs:
                q=man[(man.config==cfg)&(man.surface=='M24_curv')&(man.phase_block==1)&(man.rep==1)&(man.env==e)&(man.sigma==s)]
                C.require(len(q)==1,'Fixed point-cloud illustration key missing')
                r=C.rowdict(q.iloc[0]);a=array_reader(r);pfx=r['method']+'__ev__'
                p=a[pfx+'pts'];v=a[pfx+'valid']&a[pfx+'in_global'];pts=p[v]
                err=pts[:,2]-a[pfx+'zr'][v]
                packets.append((r,a['image'],pts,err))
        fig=plt.figure(figsize=(18,12))
        for i,(r,img,p,err) in enumerate(packets):
            row=i//2;col=(i%2)*3
            ax=fig.add_subplot(3,6,row*6+col+1);ax.imshow(img,cmap='gray',vmin=0,vmax=255);ax.set_axis_off();ax.set_title(label(r['config'])+f' mm\nPSF {r["psf_true_px"]:g}, σ {r["sigma"]:g}',fontsize=9)
            ax=fig.add_subplot(3,6,row*6+col+2,projection='3d')
            if len(p):ax.scatter(p[:,0],p[:,1],p[:,2],c=p[:,2],vmin=-16,vmax=4,cmap='viridis',s=2 if fam=='point' else .7)
            else:ax.text2D(.1,.5,'Keine gültigen Punkte',transform=ax.transAxes,color='darkred')
            ax.set(xlim=(-46,46),ylim=(-46,46),zlim=(-16,4),xlabel='x / mm',ylabel='y / mm',zlabel='z / mm')
            ax.view_init(25,-65);ax.set_title(f'n = {len(p)}',fontsize=9);ax.tick_params(labelsize=7)
            ax=fig.add_subplot(3,6,row*6+col+3)
            if len(p):
                ax.scatter(p[:,0],p[:,1],c=err,cmap='coolwarm',vmin=-.3,vmax=.3,s=4 if fam=='point' else 1)
                ax.set_title(f'Punkt-RMSE {np.sqrt(np.mean(err**2)):.3f} mm',fontsize=8)
            else:ax.text(.05,.5,'Keine gültigen Punkte',transform=ax.transAxes,color='darkred',fontsize=8)
            ax.set(xlim=(-46,46),ylim=(-46,46),xlabel='x / mm',ylabel='y / mm');ax.set_aspect('equal');ax.tick_params(labelsize=7)
        fig.suptitle('Kamerabild und Rekonstruktion — M24, Block 1, Wiederholung 1'+suffix)
        fig.text(.5,.015,'Gleiche Achsen, Graustufen und Farbskalen; gültige Punkte im globalen Gebiet. Feste Fallauswahl, keine Auswahl maximaler Gewinne.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.12,1,.94],h_pad=3.0,w_pad=2.0)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        for rect,cmap,bounds,title in [([.15,.075,.25,.013],'viridis',(-16,4),'Höhe z / mm'),([.6,.075,.25,.013],'coolwarm',(-.3,.3),'Höhenfehler / mm')]:
            cax=fig.add_axes(rect);bar=fig.colorbar(ScalarMappable(norm=Normalize(*bounds),cmap=cmap),cax=cax,orientation='horizontal',extend='both' if cmap=='coolwarm' else 'neither');bar.set_label(title,fontsize=8);bar.ax.tick_params(labelsize=7)
        p=root/f'Punktwolken_{fam}.png';C.writable(p);fig.savefig(p,dpi=150);plt.close(fig);paths.append(str(p))
    return paths

def publish(records,sens,man,root,drill=False):

    C.check_manifest(man);long,surf,choices=tables(records,sens,man);effects=paired_table(records)
    C.require(len(choices)==960,'Selection grid must be 2 families ×5 PSF×6 noise×2d×4tau×2floors')
    for name,t in [('results.csv',records),('sensitivity.csv.gz',long),('surfaces.csv',surf),('choices.csv',choices),('paired_effects.csv.gz',effects)]:
        path=root/name
        if name.endswith('.gz'):C.atomic(path,lambda p,tab=t:tab.to_csv(p,index=False,compression='gzip'))
        else:C.save_csv(path,t)
    imgs=figures(surf,choices,records,root/'figures',drill)
    C.save(root/'analysis_check.json',dict(complete=True,scientific=not drill,images=len(records),sensitivity_rows=len(long),choices=len(choices),paired_rows=len(effects),figures=imgs))
    return dict(images=len(records),sensitivity_rows=len(long),choices=len(choices),paired_rows=len(effects),figures=imgs)
