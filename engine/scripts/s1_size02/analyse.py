# Build five-radius size comparisons and figures from saved case results.

from __future__ import annotations
import json, sys, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import run as R
O=R.OUT
SF={'B_curv':'EF','M10_curv':'M10','M24_curv':'M24'}
REG={'resource_cap_11695':'Gemeinsame Obergrenzen: Q ≤ 11 695, A ≤ 1','fixed_peak_1':'Fester Peak A = 1 (Q nicht begrenzt)'}
MET=['glob_Ctau','glob_Cvalid','glob_cov_interp','glob_recovery','glob_pt_rmse','glob_pt_p95','glob_surf_rmse','id_correct','id_wrong','id_unconf','n_accepted','n_valid','glob_n_valid']
GROUP=['method','surface','regime','size_mm','sigma']
RADIUS=[.5,.75,1.,1.25,1.5]
COLORS=['#3c5488','#00a087','#e64b35','#9467bd','#e69f00']
INDEX=[]

def keepmain(t):return t[[c for c in t.columns if not c.startswith(('loc_','amp_','d_loc_'))]]
def collect(entry=False):
    man=R.csv(O/'manifest.csv');rows=[];inventory=[]
    for r in man.to_dict('records'):
        p=O/'cases'/f"{r['input_key']}.npz"
        if not p.exists():
            R.require(entry,'Missing image '+r['input_key']);continue
        meta,a=R.readpack(p);R.require(meta['complete'],'Incomplete input '+r['input_key'])
        inventory.append(dict(input_key=r['input_key'],path=str(p),sha256=R.sha(p)))
        for method in R.METHODS:
            v=meta['methods'][method];rows.append(r|dict(method=method,source='S1-SIZE-02',source_pack=str(p),source_pack_sha256=inventory[-1]['sha256'],image_array_sha256=R.arrsha(a['image']),image_record=str(p),eval_record=str(p),decode_key=v['decode_key'],eval_key=R.digest([v['decode_key'],'s1r_eval_Omega0',6.,.15,meta['bundle']]))|v['metrics'])
    n=keepmain(pd.DataFrame(rows));a=R.old();a['method']='D0';a['source_pack']=a.eval_record.map(lambda p:str((R.PROJECT/p).with_suffix('.npz')));a['source']='S12 original'
    d0=pd.concat([keepmain(a),n[n.method=='D0']],ignore_index=True)
    b=R.olddf();oldcap=a[a.regime=='resource_cap_11695'].set_index('input_key');records=[]
    identity=['surface','platform_surface','family','section','regime','size_mm','diameter_mm','layout','phase_label','phase_index','sigma','rep','noise_seed','raster_key','n_features_design','Q0','Q_budget','peak_target','peak_f32','Q_actual','budget_usage','active_limit','glob_n_expected','img_n_expected']
    for u in b.to_dict('records'):
        rec={k:oldcap.loc[u['input_key'],k] for k in identity};rec.update(u);rec.update(method='DF',section='S1',family='point',source='THESIS-REINFORCE-01 accepted');records.append(rec)
    df=pd.concat([keepmain(pd.DataFrame(records)),n[n.method=='DF']],ignore_index=True)
    dfkey=df[(df.regime==R.REGIMES[0])&df.sigma.isin([40,60,80])]
    if not entry:
        R.require(len(n)==1152 and len(d0)==1440 and len(dfkey)==405,'Result counts')
        R.require(not d0.duplicated(['surface','regime','size_mm','sigma','phase_label','rep']).any(),'D0 duplicate slot')
        R.require(not df.duplicated(['surface','regime','size_mm','sigma','phase_label','rep']).any(),'DF duplicate slot')
        R.savecsv(O/'new_cases.csv',n);R.savecsv(O/'merged_D0_cases.csv',d0);R.savecsv(O/'merged_DF_key_cases.csv',dfkey)
        R.savecsv(O/'checks/pack_inventory.csv',pd.DataFrame(inventory))
        merged=pd.concat([d0,df],ignore_index=True)
        return merged
    return n

def aggregate(t):
    rows=[]
    for keys,g in t.groupby(GROUP,sort=True):
        base=dict(zip(GROUP,keys));base.update(diameter_mm=2*base['size_mm'],n_cases=len(g),n_phases=g.phase_label.nunique(),no_output=int((g.state=='no_output').sum()),wrong_images=int((g.id_wrong>0).sum()),unconfirmed_images=int((g.id_unconf>0).sum()))
        for metric in MET:
            ph=g.groupby('phase_label')[metric].mean().reindex(['sn_phase0','sn_phase1','sn_phase2'])
            base[metric]=ph.mean();base[metric+'_min']=ph.min();base[metric+'_max']=ph.max()
            base[metric+'_phase_values']=json.dumps(R.clean(ph.tolist()));base[metric+'_n_defined_cases']=int(g[metric].notna().sum());base[metric+'_n_defined_phases']=int(ph.notna().sum())
        rows.append(base)
    c=pd.DataFrame(rows);R.savecsv(O/'conditions.csv',c);return c

def effects(t):
    rows=[];metrics=['glob_Ctau','glob_recovery','glob_Cvalid','glob_pt_rmse','glob_surf_rmse','id_wrong','id_unconf']
    def paired(a,b,keys,desc):
        R.require(not a.duplicated(keys).any() and not b.duplicated(keys).any(),'Duplicate pair')
        p=a[keys+metrics].merge(b[keys+metrics],on=keys,suffixes=('_a','_b'),validate='one_to_one')
        group=[k for k in keys if k not in ['phase_label','rep']]
        for vals,g in p.groupby(group,dropna=False):
            if not isinstance(vals,tuple):vals=(vals,)
            base=desc|dict(zip(group,vals))|dict(n_pairs=len(g),n_phases=g.phase_label.nunique())
            for met in metrics:
                d=g[met+'_a']-g[met+'_b'];h=pd.DataFrame({'phase':g.phase_label,'d':d}).groupby('phase').d.mean().reindex(['sn_phase0','sn_phase1','sn_phase2'])
                rows.append(base|dict(metric=met,difference=h.mean(),phase_min=h.min(),phase_max=h.max(),phase_values=json.dumps(R.clean(h.tolist())),positive_phases=int((h>1e-12).sum()),negative_phases=int((h< -1e-12).sum()),zero_phases=int((h.abs()<=1e-12).sum()),defined_pairs=int(d.notna().sum()),defined_phases=int(h.notna().sum())))
    for a,b in [(.75,.5),(1.,.75),(1.25,1.),(1.5,1.25),(.75,1.)]:
        paired(t[t.size_mm==a],t[t.size_mm==b],['method','surface','regime','sigma','phase_label','rep'],dict(comparison='radius',a=a,b=b))
    paired(t[t.regime=='fixed_peak_1'],t[t.regime=='resource_cap_11695'],['method','surface','size_mm','sigma','phase_label','rep'],dict(comparison='regime',a='fixed_peak_1',b='resource_cap_11695'))
    paired(t[t.method=='DF'],t[t.method=='D0'],['surface','regime','size_mm','sigma','phase_label','rep'],dict(comparison='method',a='DF',b='D0'))
    e=pd.DataFrame(rows);R.savecsv(O/'paired_effects.csv',e);return e

def finish(fig,name,title,source,keys=None):
    p=O/'figures'/name;p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=170,facecolor='white');plt.close(fig)
    INDEX.append(dict(file=str(p),title=title,data=source,case_keys=json.dumps(keys or []),domain='Omega0, 30947 nodes',definitions='point RMSE: valid recovered points in Omega0; surface RMSE: interpolable nodes; Ctau: supported accurate nodes / all fixed nodes'))

def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'savefig.bbox':'tight'})

def curves(c):
    labels={'glob_Ctau':r'Genauigkeitsabdeckung $C_\tau$','glob_recovery':'Korrekte Wiederfindungsrate','glob_pt_rmse':'Punkt-RMSE / mm','glob_surf_rmse':'Flächen-RMSE / mm'}
    for reg in R.REGIMES:
        for block,metrics in [('Abdeckung',['glob_Ctau','glob_recovery']),('Fehler',['glob_pt_rmse','glob_surf_rmse'])]:
            fig,axs=plt.subplots(2,3,figsize=(13.4,7.3),sharex=True,layout='constrained')
            for j,sf in enumerate(R.SURFS):
                for i,met in enumerate(metrics):
                    ax=axs[i,j]
                    for rad,col in zip(RADIUS,COLORS):
                        g=c[(c.method=='D0')&(c.surface==sf)&(c.regime==reg)&(c.size_mm==rad)].sort_values('sigma')
                        ax.plot(g.sigma,g[met],'-o',ms=3.5,color=col,label=f'Ø {2*rad:g} mm')
                        ax.fill_between(g.sigma,g[met+'_min'],g[met+'_max'],color=col,alpha=.10)
                    ax.set_title(SF[sf] if i==0 else '');ax.set_ylabel(labels[met]);ax.set_xticks([0,20,40,60,80,100]);ax.set_xlim(0,100)
                    if met in ['glob_Ctau','glob_recovery']:ax.set_ylim(-.025,1.025)
                    if i==1:ax.set_xlabel('Rauschstandardabweichung σ / Graustufen')
            handles,leg=axs[0,0].get_legend_handles_labels();fig.legend(handles,leg,loc='outside lower center',ncol=5)
            fig.suptitle(REG[reg]+' · D0\nMittelwerte über 3 Phasen; Schattierung: kleinster bis größter Phasenmittelwert')
            finish(fig,f'Abb_S1_FuenfDurchmesser_{reg}_{block}.png',block+' / '+reg,'conditions.csv')
    for met in ['glob_Ctau','glob_pt_rmse']:
        fig,axs=plt.subplots(3,3,figsize=(13,10),sharex=True,layout='constrained')
        for i,sig in enumerate([40,60,80]):
            for j,sf in enumerate(R.SURFS):
                ax=axs[i,j]
                for method,col in [('D0','#3c5488'),('DF','#e64b35')]:
                    g=c[(c.method==method)&(c.surface==sf)&(c.regime==R.REGIMES[0])&(c.sigma==sig)].sort_values('size_mm')
                    ax.plot(2*g.size_mm,g[met],'-o',color=col,label=method);ax.fill_between(2*g.size_mm,g[met+'_min'],g[met+'_max'],color=col,alpha=.12)
                ax.set_title(f'{SF[sf]} · σ = {sig}');ax.set_ylabel(labels[met]);ax.set_xticks([1,1.5,2,2.5,3])
                if met=='glob_Ctau':ax.set_ylim(-.025,1.025)
                if i==2:ax.set_xlabel('Punktdurchmesser / mm')
        axs[0,0].legend();fig.suptitle('Gleiche Kamerabilder: D0 und DF · Q ≤ 11 695\nVollständige gemeinsame Stichprobe: 3 Phasen × 3 Rauschrealisierungen je Bedingung')
        finish(fig,f'Abb_S1_D0_DF_{met}.png','D0 / DF '+met,'conditions.csv; merged_DF_key_cases.csv')

def resources():
    t=R.csv(O/'resource_table.csv');fig,axs=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for reg,col,label in [(R.REGIMES[0],'#3c5488','Gemeinsame Obergrenzen'),(R.REGIMES[1],'#e64b35','Fester Peak 1')]:
        for ax,met,yl in zip(axs,['Q0','peak_f32','Q_actual'],['Beleuchtete Rasterpixel S₀','Normierter Peak A','Grauwertsumme Q']):
            g=t[t.regime==reg].groupby('diameter_mm')[met].agg(['mean','min','max']);ax.plot(g.index,g['mean'],'-o',label=label,color=col);ax.fill_between(g.index,g['min'],g['max'],color=col,alpha=.15);ax.set_xlabel('Punktdurchmesser / mm');ax.set_ylabel(yl)
    axs[2].axhline(11695,ls='--',color='.35',label='Q = 11 695');axs[1].set_ylim(0,1.1);axs[2].legend(fontsize=8);fig.suptitle('Gleiche Punktpositionen, unterschiedliche Größe und Lichtverteilung')
    finish(fig,'Abb_S1_Ressourcen_FuenfDurchmesser.png','Ressourcen','resource_table.csv')

def new_frontends(c):
    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True,layout='constrained')
    for i,reg in enumerate(R.REGIMES):
        for j,sf in enumerate(R.SURFS):
            ax=axs[i,j]
            for rad,col in [(.75,COLORS[1]),(1.25,COLORS[3])]:
                for method,sty in [('D0','--'),('DF','-')]:
                    g=c[(c.surface==sf)&(c.regime==reg)&(c.size_mm==rad)&(c.method==method)].sort_values('sigma')
                    ax.plot(g.sigma,g.glob_Ctau,sty+'o',ms=3.5,color=col,label=f'Ø {2*rad:g} mm · {method}')
            ax.set_title(SF[sf]+(' · Q ≤ 11 695' if i==0 else ' · A = 1'));ax.set_ylim(-.025,1.025);ax.set_xticks([0,20,40,60,80,100]);ax.set_ylabel('Genauigkeitsabdeckung Cτ')
            if i==1:ax.set_xlabel('σ / Graustufen')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=4)
    fig.suptitle('Die beiden ergänzten Durchmesser: vollständiger D0/DF-Vergleich\n3 Phasen; σ0 einmal, σ>0 drei Realisierungen je Phase')
    finish(fig,'Abb_S1_NeueDurchmesser_D0_DF_AlleSigma.png','Neue Durchmesser, beide Frontends über alle Rauschstufen','conditions.csv; new_cases.csv')

def pairedplot(e):
    for method in ['D0','DF']:
        g=e[(e.comparison=='radius')&(e.method==method)&(e.regime==R.REGIMES[0])&(e.metric=='glob_Ctau')&(e.a.astype(str).isin(['0.75','1.25']))&(e.b.astype(str)=='1.0')]
        fig,axs=plt.subplots(1,3,figsize=(13,4.4),sharey=True,layout='constrained')
        for ax,sf in zip(axs,R.SURFS):
            for rad,col in [(.75,COLORS[1]),(1.25,COLORS[3])]:
                h=g[(g.surface==sf)&(g.a==rad)].sort_values('sigma');ax.plot(h.sigma,100*h.difference,'-o',color=col,label=f'Ø {2*rad:g} − Ø 2 mm')
                for _,r in h.iterrows():
                    v=json.loads(r.phase_values);ax.scatter([r.sigma]*3,[None if x is None else 100*x for x in v],s=14,color=col,alpha=.45)
            ax.axhline(0,color='.4',lw=.8);ax.set_title(SF[sf]);ax.set_xlabel('σ / Graustufen');ax.set_xticks([0,20,40,60,80,100])
        axs[0].set_ylabel('Gepaarte Differenz in Cτ / Prozentpunkte');axs[-1].legend(fontsize=8);fig.suptitle(f'Neue Größen gegenüber Ø 2 mm · {method} · gemeinsame Obergrenzen\nLinie: Mittelwert; Punkte: drei gepaarte Phasenmittelwerte')
        finish(fig,f'Abb_S1_GepaarteGroessen_{method}.png','Gepaarte Größen '+method,'paired_effects.csv')

def arrays(row):
    if row['source']=='S1-SIZE-02':
        meta,a=R.readpack(row['source_pack']);return a['image'],{k[4:]:v for k,v in a.items() if k.startswith('D0__')}
    with np.load((R.PROJECT/row['image_record']).with_suffix('.npz')) as z:img=z['image']
    with np.load(row['source_pack']) as z:a={k:z[k] for k in z.files}
    return img,a

def examples(t,entry=False):
    dompath=R.PROJECT/'outputs/surface_revision/s1_real01/domain/global_domain_step0p50.npz'
    with np.load(dompath) as z:dom={k:z[k] for k in z.files}
    plans=R.load(O/'protocol.json')['representatives']
    if entry:
        plan=R.csv(O/'manifest.csv');plan=plan[plan.entry_subset];fig,axs=plt.subplots(3,4,figsize=(13,9),layout='constrained')
        for ax,r in zip(axs.flat,plan.to_dict('records')):
            meta,a=R.readpack(O/'cases'/f"{r['input_key']}.npz");ax.imshow(a['image'],vmin=0,vmax=255,cmap='gray');ax.axis('off');m=meta['methods']['D0']['metrics'];ax.set_title(f"{SF[r['surface']]} · Ø{r['diameter_mm']:g} · σ{r['sigma']}\n{r['regime'].replace('resource_cap_11695','Q≤11695').replace('fixed_peak_1','A=1')} · Cτ={m['glob_Ctau']:.3f}",fontsize=9)
        finish(fig,'Eingangskontrolle_12.png','Eingangskontrolle','manifest.csv',plan.input_key.tolist());return
    d0=t[t.method=='D0']
    for plan in plans:
        q=d0.copy()
        for k,v in plan.items():q=q[q[k]==v]
        q=q.sort_values('size_mm');R.require(len(q)==5,'Example not five sizes');rs=q.to_dict('records');loaded=[arrays(r) for r in rs];keys=q.input_key.tolist()
        title=f"M24 · σ = {plan['sigma']} · "+REG[plan['regime']]
        stem=f"M24_{plan['regime']}_s{plan['sigma']}"
        fig,axs=plt.subplots(1,5,figsize=(16,4.1),layout='constrained')
        for ax,r,(im,a) in zip(axs,rs,loaded):
            ax.imshow(im,cmap='gray',vmin=0,vmax=255);ax.set_title(f"Ø {r['diameter_mm']:g} mm\nA={r['peak_f32']:.3f}; Cτ={r['glob_Ctau']:.3f}");ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
        fig.suptitle(title+'\nGleiche Position und Rauschrealisierung; Auswahl am Median des Ø-2-mm-Falls')
        finish(fig,'Abb_S1_Kamerabilder_'+stem+'.png',title,'saved images',keys)
        fig=plt.figure(figsize=(17,8.3));grid=fig.add_gridspec(2,5)
        fig.subplots_adjust(left=.035,right=.90,bottom=.07,top=.84,wspace=.22,hspace=.22)
        zz=dom['z_C_curv'];xy=np.column_stack([dom['x'],dom['y']]);norm=TwoSlopeNorm(vmin=-.5,vcenter=0,vmax=.5)
        for j,(r,(_,a)) in enumerate(zip(rs,loaded)):
            p=a['pts'][a['in_global']];err=(a['pts'][:,2]-a['zr'])[a['in_global']]
            ax=fig.add_subplot(grid[0,j],projection='3d');ax.plot_trisurf(xy[::15,0],xy[::15,1],zz[::15],color='#c2d8ba',alpha=.5,edgecolor='#8b9d85',linewidth=.12)
            for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.set_alpha(0)
            ax.scatter(p[:,0],p[:,1],p[:,2],c=err,cmap='coolwarm',norm=norm,s=8,depthshade=False)
            errlabel=f"{r['glob_pt_rmse']:.3f} mm" if pd.notna(r['glob_pt_rmse']) else 'nicht definiert'
            ax.set_xlim(-46,46);ax.set_ylim(-46,46);ax.set_zlim(-16,4);ax.view_init(27,-60);ax.set_box_aspect((1,1,.45));ax.set_title(f"Ø {r['diameter_mm']:g} mm\n{len(p)} Punkte; RMSE={errlabel}",fontsize=10);ax.set_xlabel('x / mm',fontsize=8);ax.set_ylabel('y / mm',fontsize=8);ax.set_zlabel('z / mm',fontsize=8);ax.tick_params(labelsize=7)
            bx=fig.add_subplot(grid[1,j]);bx.imshow(np.where(dom['mask'],.95,np.nan),extent=[-46.25,46.25,-46.25,46.25],origin='lower',cmap='gray',vmin=0,vmax=1,interpolation='nearest');sc=bx.scatter(p[:,0],p[:,1],c=err,cmap='coolwarm',norm=norm,s=13);bx.set_xlim(-46,46);bx.set_ylim(-46,46);bx.set_aspect('equal');bx.set_xlabel('x / mm');bx.set_ylabel('y / mm');bx.set_title(f"Cτ={r['glob_Ctau']:.3f}; Wiederfindung={r['glob_recovery']:.3f}",fontsize=9)
            if not len(p):bx.text(.5,.5,'Keine gültigen Punkte',transform=bx.transAxes,ha='center')
        cbax=fig.add_axes([.925,.20,.012,.58]);fig.colorbar(sc,cax=cbax,extend='both',label='Höhenfehler / mm (Farbgrenzen ±0,5 mm)');fig.suptitle(title+'\nD0 · Punkte und grüne Referenzfläche; unten: Punktlage und Höhenfehler',y=.97)
        finish(fig,'Abb_S1_Rekonstruktion_'+stem+'.png',title,'saved point clouds; frozen reference surface',keys)

def summary(c,e,t):
    out=[]
    for keys,g in c.groupby(['method','regime','surface','sigma']):
        if set(g.size_mm)!=set(RADIUS):continue
        g=g.sort_values('glob_Ctau',ascending=False);first=g.iloc[0]
        out.append(dict(zip(['method','regime','surface','sigma'],keys))|dict(best_radius=first.size_mm,best_diameter=first.diameter_mm,best_Ctau=first.glob_Ctau,ranking=' > '.join(f'{2*r.size_mm:g}' for r in g.itertuples()),near_best_2pp=';'.join(str(2*r.size_mm) for r in g.itertuples() if r.glob_Ctau>=first.glob_Ctau-.02)))
    R.savecsv(O/'checks/rankings.csv',pd.DataFrame(out))

def contextual_sources(c):
    p=R.PROJECT/'outputs/surface_revision/s3_code04/results/C1_dots_views.csv'
    q=R.PROJECT/'outputs/surface_revision/s5_align05/case_results.csv'
    before={str(x):R.sha(x) for x in [p,q]}
    df=R.csv(p);df=df[(df.section=='S1')&(df.rule=='DF')]
    complete=set(R.olddf().input_key);available=set(df.input_key)|complete
    catalog=R.old()[['input_key','surface','regime','size_mm','phase_label','sigma','rep']].copy()
    catalog['available_DF']=catalog.input_key.isin(available)
    catalog['five_size_curve_scope']=(catalog.regime=='resource_cap_11695')&catalog.sigma.isin([40,60,80])
    catalog['use']='not available in checked sources'
    catalog.loc[catalog.available_DF,'use']='existing but outside complete five-size comparison'
    catalog.loc[catalog.five_size_curve_scope,'use']='complete five-size comparison'
    R.savecsv(O/'checks/old_DF_availability.csv',catalog)
    s=R.csv(q);s=s[(s.family=='point')&(s.radius_mm==.75)&(s.p_parallel_mm==8)&(s.p_perp_mm==4)]
    phase=s.groupby(['surface','front_end','sigma','phase_label']).glob_Ctau.mean();sm=phase.groupby(['surface','front_end','sigma']).mean().reset_index()
    base=c[(c.size_mm==.75)&(c.regime=='resource_cap_11695')][['method','surface','sigma','glob_Ctau']]
    contrast=base.merge(sm.rename(columns={'front_end':'method','glob_Ctau':'S5_Ctau'}),on=['method','surface','sigma'],validate='one_to_one');contrast['S1_Ctau']=contrast.glob_Ctau;contrast['difference_S1_minus_S5']=contrast.S1_Ctau-contrast.S5_Ctau
    contrast['interpretation']='Unpaired descriptive comparison; different centers/phases/seeds. Not extra independent validation.'
    R.savecsv(O/'checks/S5_context_unpaired.csv',contrast.drop(columns='glob_Ctau'))
    R.require(all(R.sha(p)==h for p,h in before.items()),'Context source drift')
    R.write(O/'checks/context_sources.json',dict(files=before,S5_rows=len(s),S5_unique_images=s.image_key.nunique(),DF_availability_views=int(catalog.available_DF.sum()),scope='Catalogue only from S3-CODE-04 and THESIS-REINFORCE-01; no extra decode'))

def main():
    setup()
    if '--entry' in sys.argv:
        examples(collect(True),True);return
    t=collect();c=aggregate(t);e=effects(t);resources();curves(c);new_frontends(c);pairedplot(e);examples(t);summary(c,e,t);contextual_sources(c)
    R.savecsv(O/'figure_index.csv',pd.DataFrame(INDEX));print('ANALYSED',len(t),'method views,',len(c),'conditions,',len(e),'paired summaries,',len(INDEX),'figures',flush=True)

if __name__=='__main__':main()
