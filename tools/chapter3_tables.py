"""Export the numerical size and layout tables without assembling manuscript text."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

SURFACES=('B_curv','M10_curv','M24_curv')


def calculate(root):
    """Use the original phase means and block-paired medians for Tables 3.3, 3.4 and 3.13."""
    sr=Path(root)/'research/project_Part1_v1.7/outputs/surface_revision'
    a=pd.read_csv(sr/'s3_fe01/results/A_s12_views.csv',low_memory=False)
    a=a[a.rule=='DF'].copy()
    b=pd.read_csv(sr/'s3_fe01/results/B_size02_cases.csv',low_memory=False)
    b=b[b.rule=='DF'].copy()
    size=pd.concat([a[a.section=='S1'],b],ignore_index=True)
    assert len(size)==1440 and size.input_key.nunique()==1296
    sz=size.groupby(['regime','surface','size_mm','sigma','phase_label']).mean(numeric_only=True).reset_index()
    tables={}
    for name,regime,radius,sigmas in (
        ('table_3_3','resource_cap_11695',.75,[40,60,80,100]),
        ('table_3_4','fixed_peak_1',1.25,[60,80,100])):
        rows=[]
        for sigma in sigmas:
            for surface in SURFACES:
                subset=sz[(sz.regime==regime)&(sz.surface==surface)&(sz.sigma==sigma)]
                value=subset[subset.size_mm==radius].glob_Ctau.mean()
                reference=subset[subset.size_mm==1].glob_Ctau.mean()
                rows.append(dict(surface=surface,sigma=sigma,radius_mm=radius,coverage_pct=100*value,delta_pp=100*(value-reference)))
        tables[name]=pd.DataFrame(rows)
    c=pd.read_csv(sr/'s3_fe01/results/C_s4_cases.csv',low_memory=False)
    c=c[c.rule=='DF'].copy()
    r7=pd.read_csv(sr/'s4_real07/results.csv',low_memory=False)
    v=r7[(r7.family=='point')&(r7.method=='D0')&(r7.record_type=='view')&(r7.domain=='glob')].copy()
    assert not v.duplicated(['surface','sigma','block','phase','arm']).any()
    metric=c.set_index('image_key').glob_Ctau
    v['DF_Ctau']=v.image_key.map(metric); v['DF_S0']=v.s0_image_key.map(metric)
    assert v[['DF_Ctau','DF_S0']].notna().all().all()
    v['delta']=v.DF_Ctau-v.DF_S0
    blocks=v.groupby(['surface','sigma','block','arm']).delta.mean().unstack('arm')
    table=blocks.groupby(['surface','sigma']).median()*100
    table['S2*−S1']=(blocks['S2*']-blocks['S1']).groupby(['surface','sigma']).median()*100
    tables['table_3_13']=table.reset_index()
    return tables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root,output=args.data_root.resolve(),args.output.resolve()
    if output==root or output.is_relative_to(root):
        raise ValueError('Use an output directory outside the input tree.')
    tables=calculate(root)
    output.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():
        table.to_csv(output/(name+'.csv'),index=False)
        print(name,len(table))


if __name__=='__main__':
    main()
