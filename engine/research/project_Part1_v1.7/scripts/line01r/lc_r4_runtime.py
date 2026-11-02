# Apply width-reading rules to a shared line front end and run the decoding arms.

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / 'src'))
import lc_pipeline as LC
import lc_r3_reliability as REL
from lc_pattern import class_of_id

ALLOWED = {'features', 'comp', 'chain', 'segments', 'guard_reason', 'cand', 'dinfo', 'guard_info', 'centrelines', 'shared_identity', 'fam', 'cfg', 'p', 'fwhm_px', 'reason_ok', 'per_segment_r2'}
FORBIDDEN_SUBSTRINGS = ('truth', 'kO', 'confirm', 'correct', 'ref_', 'error', 'sigma_true', 'pass', 'amp', 'dz', 'surface_z')


def check_whitelist(inputs: dict) -> None:
    extra = set(inputs) - ALLOWED
    if extra:
        raise ValueError(f'runtime input not allowed: {sorted(extra)}')
    for k in inputs:
        if any(s in k.lower() for s in FORBIDDEN_SUBSTRINGS):
            raise ValueError(f'forbidden runtime input name: {k}')


def inputs_from_front(front: dict, readout: dict) -> dict:

    reason_ok = np.array([str(x) == 'ok' for x in readout['reason']], bool)
    per_seg = {int(k): dict(readable=bool(v['readable']), cls=v['cls']) for k, v in readout['per_segment'].items()}
    return dict(features=front['feat'], comp=front['comp'], chain=front['chain'], segments=front['segments'], guard_reason=front['guard_reason'], cand=front['cand'], dinfo=front['dinfo'], guard_info=front['guard_info'], centrelines=front['centrelines'], shared_identity=front['shared_identity'],
                fam=front['fam'], cfg=front['cfg'], p=front['p'], fwhm_px=np.asarray(readout['fwhm_px'], np.float64), reason_ok=reason_ok, per_segment_r2=per_seg)


def filter_admissible(cand, segments, seg_table):
    adm = cand['admissible'].copy(); ids_cls = class_of_id(cand['ids']); n_filtered = n_fallback = n_conflict = n_readable = 0; seg_class = np.full(len(segments), -1, np.int64)
    for sid, s in seg_table.items():
        if not s.get('readable'):
            continue
        n_readable += 1; idx = np.flatnonzero(segments == sid); seg_class[idx] = s['cls']
        for i in idx:
            row = adm[i]; k = int(row.sum())
            if k == 0:
                continue
            if k == 1:
                if ids_cls[np.flatnonzero(row)[0]] != s['cls']:
                    n_conflict += 1
                continue
            new = row & (ids_cls == s['cls'])
            if new.sum() == 0:
                n_fallback += 1; continue
            if new.sum() < k:
                n_filtered += 1
            adm[i] = new
    return adm, dict(n_readable_segments=n_readable, n_features_filtered=n_filtered, n_features_fallback=n_fallback, n_conflicts_unique_candidate=n_conflict), seg_class


def run_arms(inputs: dict, params: dict, arms=('ignore', 'use', 'R', 'A')) -> dict:

    check_whitelist(inputs)
    front = dict(feat=inputs['features'], comp=inputs['comp'], chain=inputs['chain'], segments=inputs['segments'], guard_reason=inputs['guard_reason'], cand=inputs['cand'], dinfo=inputs['dinfo'], guard_info=inputs['guard_info'], centrelines=inputs['centrelines'], shared_identity=inputs['shared_identity'], fam=inputs['fam'], cfg=inputs['cfg'], p=inputs['p'])
    cand = inputs['cand']; seg = inputs['segments']; valid = inputs['guard_reason'] == 0; fw = inputs['fwhm_px']; ok = inputs['reason_ok']; prm = params['reliability']; old_thr = float(params['old_threshold_px'])
    out = {}; peaks = REL.image_peaks(fw, ok, inputs['chain'], prm) if any(a in arms for a in ('R', 'A')) else None
    tables = {}
    if 'use' in arms:
        tables['use'] = (inputs['per_segment_r2'], old_thr, 'r2_rule')
    for arm in ('R', 'A'):
        if arm in arms:
            thr, why = REL.choose_threshold(arm, peaks, old_thr); tables[arm] = (REL.segment_table(fw, ok, seg, valid, thr, prm, 'px'), thr, why)
    if 'ignore' in arms:
        r = LC.finish(front, cand, dict(class_mode='ignore')); r['segment_class'] = np.full(len(seg), -1, np.int64); r['threshold_px'] = None; r['decision'] = 'ignore'; out['ignore'] = r
    for arm, (tab, thr, why) in tables.items():
        adm, fc, seg_class = filter_admissible(cand, seg, tab)
        r = LC.finish(front, dict(cand, admissible=adm), dict(class_mode=arm, **fc)); r['segment_class'] = seg_class; r['threshold_px'] = thr; r['decision'] = why; r['segment_table'] = tab; r['image_fallback'] = (arm in ('R', 'A') and thr is None); out[arm] = r
    for arm, r in out.items():
        r['runtime_hash'] = hashlib.sha256(r['status'].astype(np.int8).tobytes() + r['line_id'].astype(np.int32).tobytes() + np.nan_to_num(r['points'].astype(np.float32), nan=-9e9).tobytes()).hexdigest()[:16]
    return dict(arms=out, peaks=({k: v for k, v in peaks.items() if k != 'hist'} if peaks else None), hist=(peaks.get('hist') if peaks else None))
