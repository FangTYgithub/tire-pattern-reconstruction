# Validate illumination records and restrict output writes to the selected directory.

import core as C
import numpy as np
_old_metric=C.metric_gate
_old_pack=C.pack_gate

def metric_gate(records,expected):
    C.require({'n_valid','glob_n_valid','n_accepted','glob_Ctau','glob_Cvalid'}.issubset(records),'Missing support-domain fields')
    for r in records.to_dict('records'):
        n=r['n_valid'];ct=r['glob_Ctau'];cv=r['glob_Cvalid']
        C.require(np.isfinite(n) and n>=0 and n==int(n) and r['glob_n_valid']<=n<=r['n_accepted'],'Invalid total valid count')
        C.require(np.isfinite(ct) and np.isfinite(cv) and 0<=ct<=cv+1e-12 and cv<=1,'Invalid original coverage')
        if n<3:C.require(ct==0 and cv==0,'Insufficient total valid points for support')
    checked=records.copy()
    checked.loc[checked.glob_n_valid<3,['glob_Ctau','glob_Cvalid']]=0.
    return _old_metric(checked,expected)

def pack_gate(meta,arrays,r):
    C.require(int(arrays[r['method']+'__ev__valid'].sum())==meta['result']['n_valid'],'Total valid count differs from cloud')
    return _old_pack(meta,arrays,r)

def install():
    C.metric_gate=metric_gate
    C.pack_gate=pack_gate
