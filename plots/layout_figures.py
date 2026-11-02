# Draw reconstruction figures from saved point clouds and reference surfaces.

from layout_helpers import *
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'legend.fontsize':8})
inputs={}; outputs=[]
def track(p):
 p=Path(p);inputs[str(p)]=sha(p);return p
def nz(p):
 with np.load(track(p),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def rd(p):return pd.read_csv(track(p),low_memory=False)
def readj(p):return js(track(p))
def save(fig,name,details):
 p=FIG/(name+'.png');fig.savefig(p,dpi=210,bbox_inches='tight');plt.close(fig)
 outputs.append({'file':str(p),'sha256':sha(p),'details':details})
def de(x,n=1):return '–' if x is None or not np.isfinite(float(x)) else f'{float(x):.{n}f}'.replace('.',',')
SURFS=['B_curv','M10_curv','M24_curv'];NAMES={'B_curv':'EF','M10_curv':'M10','M24_curv':'M24'}
SID={'B_curv':'B_curv','M10_curv':'A_curv','M24_curv':'C_curv'}
D=nz(SR/'s1_real01/domain/global_domain_step0p50.npz')
X,Y=np.meshgrid(D['lattice_x'],D['lattice_y'])
def ref(sid):
 z=np.full(X.shape,np.nan);z[D['mask']]=D['z_'+SID[sid]];return z
def axis3(ax,title):
 ax.set(xlim=(-47,47),ylim=(-47,47),zlim=(-16,4),xlabel='x / mm',ylabel='y / mm',zlabel='z / mm')
 ax.set_box_aspect((1,1,.45));ax.view_init(28,-60);ax.set_title(title,pad=10)
 ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8);ax.zaxis.label.set_size(8)
 ax.xaxis.labelpad=1;ax.yaxis.labelpad=1;ax.zaxis.labelpad=1
 ax.tick_params(labelsize=7,pad=0)
def wire(ax,sid):ax.plot_wireframe(X[::5,::5],Y[::5,::5],ref(sid)[::5,::5],color='.6',lw=.35,alpha=.55)
C=rd(SR/'s5_align05/case_results.csv')


uv=(np.arange(512)-255.5)*.18;u,v=np.meshgrid(uv,uv)
patterns=[]
for px in [8,6]:
 dx=(u+px/2)%px-px/2;dy=(v+2)%4-2;patterns.append((f'Punkte · {px}×4 mm · r = 1,5 mm',dx*dx+dy*dy<=1.5**2))
patterns.append(('Linien · s = 8 mm · w = 1,5 mm',np.abs((u+4)%8-4)<=.75))
fig,axs=plt.subplots(2,3,figsize=(10,7.9));budget=45430;meta=[]
for j,(lab,b) in enumerate(patterns):
 s0=int(b.sum());assert s0>=budget;peak=budget/s0
 for i,amp in enumerate([1.,peak]):
  axs[i,j].imshow(b*amp,cmap='gray',vmin=0,vmax=1);axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
  axs[i,j].set_title(lab if i==0 else '',fontsize=9)
  axs[i,j].set_xlabel(f'A = {de(amp,3)} · S = {de(s0*amp,0)}',fontsize=9)
 meta.append({'pattern':lab,'unit_peak_pixel_sum':s0,'illustrative_budget':budget,'normalized_peak':peak})
fig.text(.5,.97,'Fester Spitzenwert: gleicher Wert je hellem Pixel, unterschiedliche Summen',ha='center',fontsize=11)
fig.text(.5,.50,'Gleiches Integral: Q = 45 430 (Darstellungsbeispiel)',ha='center',fontsize=11)
fig.subplots_adjust(left=.05,right=.95,bottom=.10,top=.89,wspace=.22,hspace=.34)
fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='gray'),cax=fig.add_axes([.32,.035,.36,.018]),orientation='horizontal',label='Normierter Projektionsgrauwert')
save(fig,'Abb_2_4_Grauwertbudget_Grundmuster',{'illustrative_only':True,'basis':'Eq. 2-4 and 2-5; pixel centres; s_nom=0.18 mm/px; no camera rendering','patterns':meta,'change':'replace unused length-code/affine patterns by ordinary point/continuous-line examples'})

def s5row(combo,sid,phase,sg,rep,fe):
 a=C[(C.combo==combo)&(C.surface==sid)&(C.phase_label==phase)&(C.sigma==sg)&(C.rep==rep)&(C.front_end==fe)];assert len(a)==1
 return a.iloc[0]
def s5pts(r):
 a=nz(RESEARCH/r.eval_record);v=a['valid'].astype(bool);p=a['pts'][v];e=p[:,2]-a['zr'][v];assert len(p)==r.n_valid
 return p,e


fig=plt.figure(figsize=(12,4.7));axes=[fig.add_subplot(1,3,i+1,projection='3d',computed_zorder=False) for i in range(3)]
z=ref('M10_curv');axes[0].plot_surface(X[::4,::4],Y[::4,::4],z[::4,::4],cmap='viridis',vmin=-16,vmax=4,linewidth=0)
axis3(axes[0],'Referenzfläche M10')
keys=[]
for ax,combo,fe,label in zip(axes[1:],['D_r1_p8x4','L_w0.5_s8'],['D0','P4'],['Punktmuster','Linienmuster']):
 r=s5row(combo,'M10_curv','ph0',0,0,fe);p,e=s5pts(r);wire(ax,'M10_curv')
 ax.scatter(*p.T,c=p[:,2],cmap='viridis',vmin=-16,vmax=4,s=6 if fe=='D0' else .6,depthshade=False,zorder=5)
 axis3(ax,label+' · '+fe);keys.append({'combo':combo,'eval_record':r.eval_record,'n_valid':len(p),'sigma':0})
fig.colorbar(ScalarMappable(norm=Normalize(-16,4),cmap='viridis'),ax=axes,orientation='horizontal',shrink=.35,pad=.12,label='Höhe z / mm')
save(fig,'Abb_2_8_M10_Gesamtfeld',{'cases':keys,'purpose':'output form only; different patterns, no family ranking','points':'all saved valid points','domain':'full saved global reference, not circular crop'})


selection=readj(SR/'thesis_reinforce05/run.json')['analysis']['cases']
for tag,method,name,labels in [('F4','D0','Abb_3_1_6_Ausfaelle_ohne_Kreis',['d = 2 mm','d = 1 mm']),('F5','P4','Abb_3_2_6_Linien_ohne_Kreis',['s = 10 mm','s = 8 mm'])]:
 fig=plt.figure(figsize=(11,9.5));axes=np.array([[fig.add_axes([x,y,.24,.26]) for x in [.065,.385,.705]] for y in [.66,.20]]);details=[]
 for row,kind in enumerate(['median','worst']):
  s=selection[tag][kind]; sid=s['surface'];data=[]
  for key in ['forward_key_left','forward_key_right']:
   a=nz(SR/'thesis_reinforce05/data'/f'{s[key]}.npz');m=json.loads(str(a['_meta']));v=a[method+'__ev__valid'].astype(bool)
   p=a[method+'__ev__pts'][v];e=p[:,2]-a[method+'__ev__zr'][v];mt=m['results'][method]['metrics'];assert len(p)==mt['n_valid']
   delta=None
   if method=='D0':delta=100*(m['results']['DF']['metrics']['glob_Ctau']-mt['glob_Ctau'])
   data.append((p,e,mt,delta));details.append({'forward_key':s[key],'kind':kind,'n_valid':len(p),'global_Ctau':mt['glob_Ctau']})
  err=np.concatenate([d[1] for d in data]);lim=max(float(np.percentile(np.abs(err),98)),.05)
  ybase=[.66,.20][row]
  a=axes[row,0];im=a.pcolormesh(X,Y,ref(sid),cmap='viridis',shading='nearest');fig.colorbar(im,cax=fig.add_axes([.065,ybase-.075,.24,.012]),orientation='horizontal',label='Referenzhöhe z / mm')
  a.set_title(NAMES[sid]+' · '+('median-nahes Paar' if kind=='median' else 'ungünstigstes Paar'))
  for col,(p,e,mt,delta) in enumerate(data,1):
   a=axes[row,col];sc=a.scatter(p[:,0],p[:,1],c=e,cmap='RdBu_r',vmin=-lim,vmax=lim,s=9 if method=='D0' else 1.8,linewidths=0)
   a.contour(X,Y,D['mask'].astype(float),levels=[.5],colors='.55',linewidths=.6)
   a.set_title(labels[col-1]+f" · {len(p)} gültige Punkte\nCτ global {de(100*mt['glob_Ctau'],3 if tag=='F4' else 1)} % · RMSE {de(mt['glob_pt_rmse'],3)} mm",fontsize=9)
   if delta is not None:fig.text([.505,.825][col-1],ybase-.145,f'DF: ΔCτ {de(delta,2)} pp (gleiches Bild)',ha='center',fontsize=8)
  fig.colorbar(sc,cax=fig.add_axes([.40,ybase-.075,.53,.012]),orientation='horizontal',label='z − zref / mm (Farbe: 98. Perzentil |Fehler|)')
  for a in axes[row]:a.set(xlim=(-47,47),ylim=(-47,47),aspect='equal',xlabel='x / mm',ylabel='y / mm')
 fig.suptitle(('Punktdurchmesser · σ80 · D0' if tag=='F4' else 'Linienabstand · σ60 · P4 · w = 1 mm, 0°'),y=.995)
 save(fig,name,{'selection':selection[tag],'cases':details,'change':'circle and local metrics removed; retained reference and both full point clouds; camera/histogram omitted for readability'})


board=readj(SR/'s3_code07/analysis/formal/current/case_board.json')
for fam,sigmas,name in [('dots',[100],'Abb_3_3_4_Identitaet_Gesamtfeld'),('lines',[40,60],'Abb_3_3_5_Linien_Gesamtfeld')]:
 chosen=[c for c in board if c['group'] in ('fixed','supplement') and c.get('surface')=='M24_curv' and c['family']==fam and c['sigma'] in sigmas]
 assert len(chosen)==len(sigmas)
 chosen.sort(key=lambda c:c['sigma'])
 fig=plt.figure(figsize=(12.5,4.6*len(chosen)));details=[]
 for row,c in enumerate(chosen):
  a=nz(SR/'s3_code07/formal_run/records/eval'/f"{c['eval_key']}.npz");ok=a['acc'].astype(bool)&np.isfinite(a['pts']).all(axis=1);p=a['pts'][ok]
  ids=a['ident'].astype(str) if 'ident' in a else np.where(a['bridge'][0].astype(bool),'co',np.where(a['bridge'][1].astype(bool),'wr','un'))
  idt=ids[ok];e=p[:,2]-a['zr'][ok]
  assert len(p)==c['n_accepted'];cnt=[int(np.char.startswith(idt,t).sum()) for t in ['co','wr']];assert cnt==[c['c'],c['w']]
  axes=[fig.add_subplot(len(chosen),3,row*3+k+1,projection='3d',computed_zorder=False) for k in [0,1]]
  z=ref(c['surface']);axes[0].plot_surface(X[::4,::4],Y[::4,::4],z[::4,::4],cmap='viridis',linewidth=0);axis3(axes[0],f"M24 · σ{int(c['sigma'])}\nReferenzfläche")
  wire(axes[1],c['surface']);color=np.where(np.char.startswith(idt,'co'),'#1f77b4',np.where(np.char.startswith(idt,'wr'),'#d62728','#9467bd'))
  axes[1].scatter(*p.T,c=color,s=7 if fam=='dots' else 1.5,depthshade=False,zorder=5);axis3(axes[1],f"Identität: {c['c']} korrekt / {c['w']} falsch\n{c['u']} unbestätigt")
  zlo=min(-16,float(np.floor(p[:,2].min()))-.5);zhi=max(4,float(np.ceil(p[:,2].max()))+.5)
  for ax3 in axes:ax3.set_zlim(zlo,zhi)
  ax=fig.add_subplot(len(chosen),3,row*3+3);sc=ax.scatter(p[:,0],p[:,1],c=e,s=9 if fam=='dots' else 2,cmap='RdBu_r',vmin=-.5,vmax=.5,linewidths=0)
  ax.set(xlim=(-47,47),ylim=(-47,47),aspect='equal',xlabel='x / mm',ylabel='y / mm');ax.set_title(f"Punktfehler in Draufsicht\nCτ global {de(c['Ctau'],3)}",fontsize=10)
  fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.17,fraction=.045,label='z − zref / mm (Farbe auf ±0,5 begrenzt)')
  details.append({'eval_key':c['eval_key'],'n_accepted':len(p),'c':c['c'],'w':c['w'],'u':c['u'],'Ctau':c['Ctau'],'max_abs_error':float(np.abs(e).max()),'points_outside_color':int((np.abs(e)>.5).sum()),'zlim_includes_all_accepted_points':[zlo,zhi]})
 handles=[Line2D([],[],marker='o',ls='',color=col,label=lab) for col,lab in [('#1f77b4','korrekt'),('#d62728','falsch'),('#9467bd','unbestätigt')]]
 fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False);fig.subplots_adjust(left=.015,right=.97,top=.90,bottom=.17 if fam=='dots' else .09,wspace=.28,hspace=.42)
 save(fig,name,{'cases':details,'change':'global truth and accepted point clouds retained; former node map replaced by point-height error view, no ROI'})


S=rd(SR/'s5_align05/condition_summary.csv');sig=[0,20,40,60,80,100]
for fam,fe,name in [('point','D0','Abb_3_5_1_Punkte_Global'),('line','P4','Abb_3_5_2_Linien_Global')]:
 combos=sorted(S.loc[S.front_end==fe,'combo'].unique())
 if fam=='point':combos=sorted([c for c in combos if c.startswith('D_')],key=lambda c:(float(c.split('_')[1][1:]),float(c.split('_p')[1].split('x')[0]),float(c.split('x')[-1])))
 else:combos=[c for c in combos if c.startswith('L_')]
 rows=[('glob_Ctau_mean8','Cτ global / %',100),('glob_pt_rmse_median','Punkt-RMSE global / mm',1),('sup_surf_rmse_median','Flächen-RMSE gestützt / mm',1)]
 fig,axs=plt.subplots(3,3,figsize=(10.2,8),sharex=True);checks=[]
 for j,sid in enumerate(SURFS):
  for i,(met,lab,factor) in enumerate(rows):
   for c in combos:
    g=S[(S.combo==c)&(S.surface==sid)&(S.front_end==fe)].set_index('sigma').reindex(sig)
    if fam=='point':
     r=float(c.split('_')[1][1:]);spacing=c.split('_p')[1];color={.75:'#2ca02c',1:'#1f77b4',1.5:'#d62728'}[r];ls={'6x4':':','8x4':'-','10x4':'--','8x6':'-.'}[spacing];label=f'r {de(r,2).rstrip("0").rstrip(",")} · {spacing.replace("x","×")}'
    else:
     w=float(c.split('_')[1][1:]);spacing=c.split('_s')[1];color='#1f77b4' if w==.5 else '#d62728';ls={'4.45':':','6.3':'--','8':'-'}[spacing];label=f'w {de(w)} · s {spacing.replace(".",",")}'
    y=pd.to_numeric(g[met],errors='coerce').to_numpy()*factor;axs[i,j].plot(sig,y,color=color,ls=ls,marker='o',ms=3,lw=1,label=label)
    checks.append({'combo':c,'surface':sid,'metric':met,'sigma':sig,'values':[None if not np.isfinite(v) else float(v) for v in y]})
    if i:
     nd=g[met.replace('_median','_n_defined')];partial=(nd>0)&(nd<g.n_cases);axs[i,j].plot(np.array(sig)[partial],y[partial],'o',mfc='white',mec=color,ms=4,ls='none')
   a=axs[i,j];a.grid(alpha=.25);a.set_xticks(sig)
   if i==0:a.set_title(NAMES[sid]);a.set_ylim(-3,103)
   else:a.set_yscale('log')
   if j==0:a.set_ylabel(lab)
   if i==2:a.set_xlabel('Eingangsrauschen σ')
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
 fig.text(.02,.015,'Abdeckung: gleichgewichtete Phasenmittel, ohne Ausgabe = 0. Fehler: Median definierter Fälle.\nOffene Marker: nicht alle Fälle definiert; Lücken: kein definierter Fehler. Größen und Abstände in mm.',fontsize=8)
 fig.subplots_adjust(left=.11,right=.99,bottom=.12,top=.85,wspace=.25,hspace=.20)
 save(fig,name,{'front_end':fe,'n_candidates':len(combos),'series':checks,'change':'local coverage row removed; all global series unchanged'})


fig,axs=plt.subplots(3,4,figsize=(11.5,9));details=[]
for i,sg in enumerate([0,40,100]):
 r=s5row('L_w0.5_s8','B_curv','ph3',sg,0 if sg==0 else 1,'P4');a=nz(RESEARCH/r.eval_record);v=a['valid'].astype(bool);p=a['pts'][v];zt=a['zr'][v];err=p[:,2]-zt
 candidates=[SR/'s345_ready/run/image'/f'{r.image_key}.npz',SR/'s5_six_formal/store/image'/f'{r.image_key}.npz']
 ip=next((p for p in candidates if p.exists()),None)
 if ip is None:
  found=list(SR.rglob(f'{r.image_key}.npz'));assert len(found)==1;ip=found[0]
 img=nz(ip)['image'];axs[i,0].imshow(img,cmap='gray',vmin=0,vmax=255);axs[i,0].axis('off');axs[i,0].set_title(f'EF · σ{sg}\nKamerabild')
 for j,(vals,title,cmap,lims) in enumerate([(zt,'Wahre Höhe an Messpunkten','viridis',(-16,4)),(p[:,2],'Rekonstruierte Höhe','viridis',(-16,4)),(err,'Höhenfehler','RdBu_r',(-.5,.5))],1):
  ax=axs[i,j]
  if len(p):ax.scatter(p[:,0],p[:,1],c=vals,s=1.5,cmap=cmap,vmin=lims[0],vmax=lims[1],linewidths=0)
  else:ax.text(.5,.5,'keine gültigen Punkte',ha='center',transform=ax.transAxes,fontsize=8)
  ax.set(xlim=(-47,47),ylim=(-47,47),aspect='equal',xlabel='x / mm',ylabel='y / mm');ax.set_title(title,fontsize=9)
 axs[i,0].text(.5,-.11,f"{len(p)} gültige Punkte\nCτ global {de(100*r.glob_Ctau)} %",ha='center',va='top',transform=axs[i,0].transAxes,fontsize=8)
 details.append({'eval_record':r.eval_record,'image':str(ip),'sigma':sg,'n_valid':len(p),'Ctau':float(r.glob_Ctau)})
fig.subplots_adjust(left=.04,right=.98,top=.95,bottom=.14,wspace=.28,hspace=.65)
for rect,cmap,lim,label in [([.35,.045,.29,.015],'viridis',(-16,4),'Höhe z / mm'),([.75,.045,.20,.015],'RdBu_r',(-.5,.5),'z − zref / mm (±0,5)')]:
 fig.colorbar(ScalarMappable(norm=Normalize(*lim),cmap=cmap),cax=fig.add_axes(rect),orientation='horizontal',label=label)
save(fig,'Abb_3_5_6_Linien_Global',{'cases':details,'change':'global-only annotations; all valid saved points'})


from matplotlib.colors import LogNorm
O=nz(SR/'pix_bound01/domains/omega1.npz')
sets=[('M10_curv','Punktpositionen · σ0',['R7E-4772d06a571fb22a7b1f1854','R7E-b4eca56f794c4c54c1cb14a4'],['S0: regelmäßig','S1: frei verschoben']),
      ('M10_curv','Randrestschutz · σ0',['R7E-32aaa0e2250b5ce974a36964','R7E-b15a18e82fbdfdd594f42d34'],['P4','CLIP']),
      ('M24_curv','Kosten des Randrestschutzes · σ0',['R7E-c308dc0d74b7963913b0a557','R7E-1beff84016228c4154f2dfd7'],['P4','CLIP'])]
loaded=[];allerr=[]
for sid,title,keys,labs in sets:
 group=[]
 for key,lab in zip(keys,labs):
  a=nz(SR/'s4_real07/evals'/f'{key}.npz');e=readj(SR/'s4_real07/evals'/f'{key}.json');v=a['valid'].astype(bool);pts=a['pts'][v];err=np.abs(pts[:,2]-a['zr'][v]);allerr.extend(err);group.append((key,lab,pts,err,e))
 loaded.append((sid,title,group))
norm=LogNorm(.001,max(allerr))
for i,(sid,title,group) in enumerate(loaded):
 xx,yy,zz=O['x'],O['y'],O['z_'+SID[sid]];xs,ys=np.unique(xx),np.unique(yy);ox,oy=np.meshgrid(xs,ys);oz=np.full(ox.shape,np.nan);oz[np.searchsorted(ys,yy),np.searchsorted(xs,xx)]=zz
 bounds=np.vstack([np.c_[xx,yy,zz]]+[g[2] for g in group]);lo,hi=bounds.min(axis=0),bounds.max(axis=0)
 fig=plt.figure(figsize=(9.8,5.4));details=[]
 for j,(key,lab,pts,err,ev) in enumerate(group):
  ax=fig.add_subplot(1,2,j+1,projection='3d',computed_zorder=False);ax.plot_wireframe(ox,oy,oz,rstride=10,cstride=10,color='.65',lw=.35,alpha=.6)
  ax.scatter(*pts.T,c=np.maximum(err,.001),norm=norm,cmap='viridis',s=8 if i==0 else 1.5,depthshade=False,zorder=5)
  ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2]-1,hi[2]+1),xlabel='x / mm',ylabel='y / mm',zlabel='z / mm');ax.view_init(28,-60);ax.set_box_aspect((1,1,.7));ax.tick_params(labelsize=7,pad=0)
  mt=ev['metrics'];rm=ev['nodes']['glob']['sup_surf_rmse'];ax.set_title(f"{lab}\nCτ global {de(100*mt['glob_Ctau'])} % · RMSE gestützt {de(rm,3)} mm",fontsize=10)
  details.append({'eval_key':key,'n_valid':len(pts),'Ctau':mt['glob_Ctau'],'supported_RMSE':rm})
 fig.suptitle(NAMES[sid]+' · '+title,y=.99);fig.subplots_adjust(left=.0,right=.94,top=.79,bottom=.26,wspace=.16)
 fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),cax=fig.add_axes([.24,.11,.52,.02]),orientation='horizontal',label='|z − zref| / mm (logarithmisch, gemeinsame Skala)')
 fig.text(.5,.005,'Alle gültigen Punkte; gemeinsame Ansicht und Achsen je Paar. Fehler < 0,001 mm farblich auf 0,001 gesetzt.',ha='center',fontsize=8)
 save(fig,f'Abb_3_4_{i+4}_Gesamtfeld',{'cases':details,'axis_min':lo.tolist(),'axis_max':hi.tolist(),'change':'removed red circle only; exact existing cases, all valid points, global metrics retained'})
assert all(sha(p)==h for p,h in inputs.items())
dump(BUILD/'figure_provenance.json',{'read_only_inputs':inputs,'outputs':outputs,'no_render_decode_or_evaluation':True})
print('Saved',len(outputs),'figures; verified',len(inputs),'read-only inputs')
