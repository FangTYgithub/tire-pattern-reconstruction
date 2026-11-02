# Build the spacing-layout batch and execution plan.

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2p_common as K

K.setup()
FORMAL_SETS = ('core-main', 'core-gauss')
RETRY_CAP = dict(render=4, decode=4)
FREEZE_FORMAL = K.RUN_DIR / 'freeze_formal_batch.json'
PROGRESS = K.OUT / 'progress.json'
CHECKS_FORMAL = K.RUN_DIR / 'checks_formal.json'
DRILL = K.OUT / 'drill_formal'
TASK_FILE = K.ROOT.parents[1] / 'protocols' / 'S2-REAL-01_protocol.md'
RUN_CODE = ('s2p_common.py', 's2p_case.py', 's2p_manifest.py', 's2p_reuse.py', 's2p_freeze.py', 's2p_formal.py', 's2p_run.py')
REPORT_CODE = ('s2p_stats.py', 's2p_collect.py', 's2p_figs.py', 's2p_checks.py', 's2p_report.py')
NUMERIC_OWN = ('s2p_common.py', 's2p_case.py', 's2p_manifest.py', 's2p_reuse.py', 's2p_freeze.py')
SURF_ORDER = {s: i for i, s in enumerate(K.SURFACES)}


class Stop(Exception):
    def __init__(self, code, msg):
        super().__init__(msg); self.code = code


def permit_path(set_name):
    return K.RUN_DIR / f'permit_{set_name}.json'


def formal_files() -> list[str]:
    old = K.load_json(K.FREEZE)['files']
    own = {K.rel(K.HERE / f) for f in RUN_CODE}
    rep = {K.rel(K.HERE / f) for f in REPORT_CODE}
    return sorted((set(old) - rep) | own)


def snapshot() -> dict:
    import s2p_freeze as FZ
    files = {f: (K.sha_file(K.ROOT / f) if (K.ROOT / f).exists() else None) for f in formal_files()}
    s = dict(files=files, meshes=FZ.mesh_hashes())
    missing = [f for p_ in FZ.PARTS.values() for f in p_ if f not in files]
    if missing:
        raise Stop(12, f'formal freeze does not cover the stage files {missing[:5]}')
    s['part_digests'] = {p_: FZ.part_digest(s, p_) for p_ in FZ.PARTS}
    return s


def bridge(snap: dict) -> dict:

    old = K.load_json(K.FREEZE)
    changed_numeric = sorted(f for f, h in old['files'].items() if f.split('/')[-1] not in REPORT_CODE and snap['files'].get(f) != h)
    rep_now = {f: K.sha_file(K.ROOT / f) for f in old['files'] if f.split('/')[-1] in REPORT_CODE}
    rep_changed = sorted(f for f, h in rep_now.items() if h != old['files'][f])
    ok = (not changed_numeric and snap['meshes'] == old['meshes'] and snap['part_digests'] == old['part_digests'])
    return dict(ok=ok, joint10_freeze_sha256=K.sha_file(K.FREEZE), numeric_files_changed=changed_numeric, report_files_changed=rep_changed,
                meshes_equal=snap['meshes'] == old['meshes'], part_digests_equal=snap['part_digests'] == old['part_digests'], part_digests=snap['part_digests'],
                own_numeric_files_equal={f: K.sha_file(K.HERE / f) == old['files'].get(K.rel(K.HERE / f)) for f in NUMERIC_OWN},
                basis='joint10 records were written with the joint10 part digests; every numerical file of the joint10 freeze (S1 dependency closure, PARTS, list, '
                      'config, release, own numerical code) and the four meshes are byte-identical now, so the stored identities stay valid; only reporting / '
                      'scheduling code differs (not a run dependency)')


def write_freeze() -> dict:
    s = snapshot(); b = bridge(s)
    if not b['ok']:
        raise Stop(12, f"bridge to the joint10 freeze fails: numeric changes {b['numeric_files_changed'][:5]}, meshes {b['meshes_equal']}, parts {b['part_digests_equal']}")
    rec = dict(task='S2-REAL-01', created_local=K.now_local(), files=s['files'], meshes=s['meshes'], part_digests=s['part_digests'], run_code=list(RUN_CODE), report_code_not_frozen=list(REPORT_CODE),
               bridge_joint10=b, note='verified against the real files at start, resume, every 200 cases and every set switch; any drift stops the run')
    K.save_json(FREEZE_FORMAL, rec)
    return rec


def verify_freeze(path=None) -> dict:
    p = Path(path or FREEZE_FORMAL)
    if not p.exists():
        raise Stop(12, f'formal freeze missing: {K.rel(p)}')
    fz = K.load_json(p); now = snapshot()
    drift = [f for f, h in fz['files'].items() if now['files'].get(f) != h]
    mesh = [s for s, h in fz['meshes'].items() if now['meshes'].get(s) != h]
    if drift or mesh or now['part_digests'] != fz['part_digests']:
        raise Stop(12, f'formal freeze drift: files {drift[:6]}, meshes {mesh}, parts equal {now["part_digests"] == fz["part_digests"]}')
    return dict(ok=True, sha256=K.sha_file(p), part_digests=fz['part_digests'])


_RAS: dict = {}


def _raster(t):
    import s2p_case as CS
    if t['raster_key'] not in _RAS:
        _RAS[t['raster_key']] = CS.raster_for(t)
    return _RAS[t['raster_key']]


class _B:
    version = None


def plan(set_name: str, parts: dict, out=None) -> pd.DataFrame:
    import s2p_case as CS
    import s2p_reuse as RU
    import s2p_run as RN
    if set_name not in FORMAL_SETS:
        raise Stop(14, f'set {set_name} is not a formal set of S2-REAL-01 (optional / joint10 are not run here)')
    out = Path(out or K.RUN_DIR)
    d = RN.load_list(); rows = RN.set_rows(d, set_name)
    want_cls = 'main' if set_name == 'core-main' else 'supplement'
    if (rows.tier != 'core').any() or (rows.surface_class != want_cls).any():
        raise Stop(16, f'{set_name}: rows outside the core / {want_cls} scope')
    T, _ = RU.build(d); idx = {r['image_key']: r for r in T.to_dict('records')}
    _B.version = K.platform_state()['version']
    acts = []
    for _, r in rows.iterrows():
        t = r.to_dict(); u = idx.get(t['image_key']) or {}
        if u.get('image_status') == 'ready' and u.get('eval_status') == 'ready':
            acts.append(dict(cand_id=t['cand_id'], action='reuse_s1', s1_image_key=u['s1_image_key'], s1_decode_key=u['s1_decode_key'], need_render=False, need_decode=False, need_eval=False)); continue
        R = _raster(t); peak = CS.peak_for(t, R); ids = CS.identities(t, _B, parts, peak)
        ip = out / 'images' / f"{t['image_key']}.json"; rp = out / 'raw' / f"{t['decode_key']}.json"; ep = out / 'eval' / f"{t['decode_key']}.json"
        ni, nd, ne = not CS.complete(ip, ids['image']), not CS.complete(rp, ids['decode']), not CS.complete(ep, ids['eval'])
        acts.append(dict(cand_id=t['cand_id'], action=('stored' if not (ni or nd or ne) else 'new'), s1_image_key='', s1_decode_key='', need_render=ni, need_decode=nd, need_eval=ne))
    A = rows.merge(pd.DataFrame(acts), on='cand_id', how='left')
    A['_o'] = [(SURF_ORDER[s], f, l, float(g), p, int(r)) for s, f, l, g, p, r in zip(A.surface, A.family, A.layout, A.sigma, A.phase_label, A.rep)]
    return A.sort_values('_o', kind='mergesort').drop(columns='_o').reset_index(drop=True)


def plan_summary(A: pd.DataFrame) -> dict:
    new = A[A.action == 'new']
    return dict(cases=int(len(A)), images=int(A.image_key.nunique()), decodes=int(A.decode_key.nunique()), reuse_s1=int((A.action == 'reuse_s1').sum()), stored=int((A.action == 'stored').sum()),
                new=int(len(new)), renders_needed=int(new.need_render.sum()), decodes_needed=int(new.need_decode.sum()), evals_needed=int(new.need_eval.sum()),
                by_surface={s: dict(cases=int(len(g)), reuse_s1=int((g.action == 'reuse_s1').sum()), stored=int((g.action == 'stored').sum()), new=int((g.action == 'new').sum()))
                            for s, g in A.groupby('surface')},
                by_family={f: int(len(g)) for f, g in A.groupby('family')})


class SetLedger:
    def __init__(self, set_name, caps, path=None):
        import s2p_run as RN
        self.set = set_name; self.caps = caps; self.L = RN.Ledger(Path(path or K.LEDGER))
        self._ev = [e for e in self.L.events() if e.get('event') == 'start']

    def tally(self) -> dict:
        ev = self._ev
        out = {}
        for kind in ('render', 'decode'):
            mine = [e for e in ev if e.get('kind') == kind and e.get('set') == self.set]
            seen, n_norm, n_retry = set(), 0, 0
            for e in mine:
                if e['key'] in seen:
                    n_retry += 1
                else:
                    seen.add(e['key']); n_norm += 1
            out[kind] = dict(normal=n_norm, retry=n_retry)
            out[f'{kind}_joint10'] = sum(1 for e in ev if e.get('kind') == kind and not e.get('set'))
            out[f'{kind}_other_formal_sets'] = sum(1 for e in ev if e.get('kind') == kind and e.get('set') and e.get('set') != self.set)
        return out

    def start(self, kind, key):
        t = self.tally()[kind]
        retry = any(e.get('kind') == kind and e.get('set') == self.set and e.get('key') == key for e in self._ev)
        if retry and t['retry'] + 1 > RETRY_CAP[kind]:
            raise Stop(16, f'{self.set}: {kind} retry cap {RETRY_CAP[kind]} reached')
        if not retry and t['normal'] + 1 > self.caps[f'{kind}_normal']:
            raise Stop(16, f"{self.set}: {kind} normal cap {self.caps[f'{kind}_normal']} reached")
        tok = f'{self.set}:{kind}:{key}:{os.getpid()}:{len(self._ev)}'
        e = dict(event='start', kind=kind, key=key, token=tok, set=self.set, attempt=('retry' if retry else 'normal'), n_before=t)
        self.L._w(e); self._ev.append(e)
        return tok

    def end(self, tok, outcome, detail=None):
        self.L.end(tok, outcome, detail)

    def note(self, **kw):
        self.L.note(set=self.set, **kw)


def permit_expect(set_name: str, A: pd.DataFrame, fz_sha: str) -> dict:
    import s2p_manifest as MF
    s = plan_summary(A)
    return dict(kind='production-formal', task='S2-REAL-01', set=set_name, cand_ids_sha256=K.digest(sorted(A.cand_id)), n_cases=s['cases'], actions=dict(reuse_s1=s['reuse_s1'], stored=s['stored'], new=s['new']),
                caps=dict(render_normal=s['renders_needed'], decode_normal=s['decodes_needed'], render_retry=RETRY_CAP['render'], decode_retry=RETRY_CAP['decode']),
                candidates_sha256=K.sha_file(MF.CSV), config_sha256=K.sha_file(MF.CONFIG), freeze_sha256=fz_sha, platform_version=K.platform_state()['version'])


def write_permit(set_name: str) -> dict:
    v = verify_freeze()
    c = K.load_json(CHECKS_FORMAL) if CHECKS_FORMAL.exists() else {}
    if c.get('n_failed') != 0 or c.get('freeze_sha256') != v['sha256']:
        raise Stop(17, 'formal checks must pass on the current formal freeze before a permit is written')
    A = plan(set_name, v['part_digests'])
    p = permit_expect(set_name, A, v['sha256'])
    p.update(authorised_by=(f'S2-REAL-01 task book ({K.rel(TASK_FILE)}, sha256 {K.sha_file(TASK_FILE)}) and the user instruction of 2026-10-12 (run the formal section-2 matrix after the '
                            'pre-run fixes and checks); written by the executor (execution) after passed checks; NOT an acceptance of formal results by reference'),
             checks_sha256=K.sha_file(CHECKS_FORMAL), created_local=K.now_local())
    K.save_json(permit_path(set_name), p)
    return p


def check_permit(set_name: str, A: pd.DataFrame, fz_sha: str) -> dict:
    pp = permit_path(set_name)
    if not pp.exists():
        raise Stop(14, f'no permit for {set_name}')
    p = K.load_json(pp); exp = permit_expect(set_name, A, fz_sha)

    stable = ('kind', 'task', 'set', 'cand_ids_sha256', 'n_cases', 'candidates_sha256', 'config_sha256', 'freeze_sha256', 'platform_version')
    bad = [k for k in stable if K.clean(p.get(k)) != K.clean(exp.get(k))]
    if bad or not p.get('authorised_by'):
        raise Stop(14, f'permit of {set_name} does not match: {bad}')
    if p['actions']['reuse_s1'] != exp['actions']['reuse_s1']:
        raise Stop(16, f"reuse count changed since the permit: {p['actions']['reuse_s1']} -> {exp['actions']['reuse_s1']}")
    return p


def write_progress(**kw):
    cur = K.load_json(PROGRESS) if PROGRESS.exists() else {}
    sets = dict(cur.get('sets') or {})
    s = kw.pop('set_name')
    sets[s] = dict(sets.get(s) or {}, **kw, t_local=K.now_local())
    K.save_json(PROGRESS, dict(task='S2-REAL-01', updated_local=K.now_local(), sets=sets, note='single progress file of the formal run (one research runner, sequential)'))


def first_case_check(out, t) -> dict:
    import s2p_case as CS
    ip = out / 'images' / f"{t['image_key']}.json"; rp = out / 'raw' / f"{t['decode_key']}.json"; ep = out / 'eval' / f"{t['decode_key']}.json"
    r = dict(records_complete=bool(CS.complete(ip) and CS.complete(rp) and CS.complete(ep)))
    with np.load(ip.with_suffix('.npz')) as z:
        img = z['image']; r['image_uint8_512'] = bool(img.dtype == np.uint8 and img.shape == (512, 512))
    with np.load(ep.with_suffix('.npz')) as z:
        E = {k: np.asarray(z[k]) for k in z.files}
    with np.load(rp.with_suffix('.npz')) as z:
        Rw = set(z.files)
    m = K.load_json(ep)['metrics']
    r['both_domains_and_node_fields'] = all(k in E for k in ('in_local', 'in_global', 'loc_node_supported', 'glob_node_supported', 'loc_node_err', 'glob_node_z_est')) and m.get('loc_n_nodes') == 22605 and m.get('glob_n_nodes') == 30947
    v = E['valid']; z = E['pts'][:, 2]; neg = E['accepted'] & np.isfinite(z) & (z < 0) & (z >= K.WINDOW[0])
    r['negative_z_kept_valid'] = bool((neg & ~v & np.isfinite(E['zr'])).sum() == 0)
    r['identity_closure'] = int(E['point_correct'].sum() + E['point_wrong'].sum() + E['point_unconf'].sum()) == int(E['accepted'].sum())
    if t['family'] == 'point':
        pf = E['point_feature']
        r['correspondence_index_closed'] = bool(len(pf) == len(E['pts']) and (pf >= 0).all() and E['feat_point_idx'][pf].tolist() == list(range(len(pf))))
        r['raw_component_reasons'] = 'comp_reason' in Rw and 'cand_table' in Rw
    else:
        r['line_reference_rows'] = all(k in E for k in ('ref_k', 'ref_v', 'ref_rec_id', 'truth_k', 'line_id', 'status'))
        r['raw_line_arrays'] = all(f'r_{k}' in Rw for k in ('features', 'line_id', 'status', 'segment_id'))
    r['ok'] = all(v_ for k, v_ in r.items())
    return r


# Run the selected spacing set after checking inputs, dependencies and execution limits.
def run_set(set_name: str, case_fn=None, out=None, ledger_path=None, bd=None, dom_g=None, freeze_path=None, lock_path=None, is_drill=False, every_verify=200, drill_caps=None):
    import s2p_case as CS
    import s2p_reuse as RU
    import s2p_run as RN
    out = Path(out or K.RUN_DIR); case_fn = case_fn or CS.run_case
    v = verify_freeze(freeze_path)
    if not is_drill:
        st = K.platform_state()
        if not st['bound_ok']:
            raise Stop(12, f"platform not bound / drift {st.get('dependency_drift')}")
    A = plan(set_name, v['part_digests'], out=(out if is_drill else None))
    p = check_permit(set_name, A, v['sha256']) if not is_drill else dict(caps=drill_caps)
    led = SetLedger(set_name, p['caps'], ledger_path)
    with RN.Lock(Path(lock_path) if lock_path else K.LOCK):
        tal0 = led.tally(); s = plan_summary(A)
        led.note(msg='formal run start', freeze_sha256=v['sha256'], plan=s, tally=tal0)
        R = A[A.action == 'reuse_s1']
        if not is_drill:
            T, _ = RU.build(RN.load_list()); idx = {r['image_key']: r for r in T.to_dict('records')}
            bad = []
            for _, r in R.iterrows():
                lay = RU.verify_layers(idx[r.image_key])
                if not (lay['image_ok'] and lay['decode_ok'] and lay['eval_ok']):
                    bad.append((r.cand_id, lay['reasons']))
            led.note(msg='reuse layers verified', n=len(R), failed=bad[:10])
            if bad:
                raise Stop(16, f'{len(bad)} reuse rows fail a layer: {bad[:3]} (no silent new imaging)')
        if not is_drill:
            import adp_binding as BD
            import s1r_common as S1K
            bd = BD.bind(required_methods=sorted(set(K.ARM.values()))); dom_g = S1K.global_domain(0.5)
        todo = A[A.action == 'new']
        t0 = time.time(); done = 0; firsts = {}
        write_progress(set_name=set_name, state='running', plan=s, done_new=0, tally=tal0, technical_issues=[])
        for i, (_, r) in enumerate(todo.iterrows()):
            t = r.to_dict()
            if i and i % every_verify == 0:
                verify_freeze(freeze_path)
            try:
                res = case_fn(t, bd, dom_g, v['part_digests'], None, out, led)
            except Stop:
                write_progress(set_name=set_name, state='stopped', done_new=done, tally=led.tally(), technical_issues=[f"cap / stop at {t['cand_id']}"]); raise
            except BaseException as e:
                led.note(msg='technical stop', cand_id=t['cand_id'], error=repr(e)[:400])
                write_progress(set_name=set_name, state='stopped', done_new=done, tally=led.tally(), technical_issues=[f"{t['cand_id']}: {e!r}"[:400]])
                raise Stop(16, f"technical stop at {t['cand_id']}: {e!r}") from e
            done += 1
            if not is_drill and t['family'] not in firsts:
                fc = first_case_check(out, t); firsts[t['family']] = fc; led.note(msg='first case check', cand_id=t['cand_id'], family=t['family'], **fc)
                if not fc['ok']:
                    raise Stop(16, f"first {t['family']} case check failed: {fc}")
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t0; rate = done / el if el > 0 else None
                write_progress(set_name=set_name, state='running', plan=s, done_new=done, todo_new=int(len(todo)), tally=led.tally(), technical_issues=[],
                               seconds_elapsed=round(el, 1), cases_per_min=(round(60 * rate, 2) if rate else None), eta_min=(round((len(todo) - done) / rate / 60, 1) if rate else None),
                               last=dict(cand_id=t['cand_id'], surface=t['surface'], layout=t['layout'], sigma=t['sigma'], state=res.get('state')))
        verify_freeze(freeze_path)
        tal = led.tally(); led.note(msg='formal run end', done_new=done, tally=tal)
        write_progress(set_name=set_name, state='complete', plan=s, done_new=done, todo_new=int(len(todo)), tally=tal, technical_issues=[], seconds_elapsed=round(time.time() - t0, 1),
                       first_case_checks=firsts)
    return dict(set=set_name, done_new=done, plan=s, tally=tal)


# Check spacing plans, reuse decisions and interrupted-run handling.
def checks() -> dict:
    import s2p_case as CS
    import s2p_reuse as RU
    import s2p_run as RN
    res = {}; v = verify_freeze()
    fz = K.load_json(FREEZE_FORMAL); res['bridge_joint10'] = dict(ok=fz['bridge_joint10']['ok'], numeric_changed=fz['bridge_joint10']['numeric_files_changed'],
                                                                report_changed=fz['bridge_joint10']['report_files_changed'])
    st = K.platform_state(); res['platform'] = dict(ok=st['bound_ok'], version=st['version'], drift=st.get('dependency_drift'))
    plans = {s: plan(s, v['part_digests']) for s in FORMAL_SETS}
    sm = {s: plan_summary(A) for s, A in plans.items()}
    exp = {'core-main': dict(cases=1728, reuse_s1=72, stored=8, new=1648), 'core-gauss': dict(cases=576, reuse_s1=24, stored=0, new=552)}
    res['counts'] = dict(ok=all(all(sm[s][k] == x for k, x in e.items()) for s, e in exp.items()), summary=sm, expected=exp)
    d = RN.load_list()
    both = pd.concat(plans.values())
    res['scope'] = dict(ok=bool((both.tier == 'core').all() and both.cand_id.is_unique and not set(both.cand_id) & set(d[d.tier == 'optional'].cand_id)
                                and set(both.cand_id) == set(d[d.tier == 'core'].cand_id)), n=int(len(both)))
    reg = []
    for key, g in both.groupby(['surface', 'family', 'layout']):
        s0, s4 = g[g.sigma == 0.0], g[g.sigma == 40.0]
        if not (s0.phase_label.nunique() == 8 and len(s0) == 8 and (s0.rep == 0).all() and s4.phase_label.nunique() == 8 and len(s4) == 16 and sorted(set(s4.rep)) == [0, 1]
                and s0.noise_seed.map(K.norm_seed).isna().all() and s4.noise_seed.map(K.norm_seed).notna().all()):
            reg.append(key)
    res['registered_phases_reps_seeds'] = dict(ok=not reg, bad=reg[:5])
    T, _ = RU.build(d); idx = {r['image_key']: r for r in T.to_dict('records')}; badr = []
    for s, A in plans.items():
        for _, r in A[A.action == 'reuse_s1'].iterrows():
            lay = RU.verify_layers(idx[r.image_key])
            if not (lay['image_ok'] and lay['decode_ok'] and lay['eval_ok']):
                badr.append(r.cand_id)
    res['reuse_three_layers_all_96'] = dict(ok=not badr and sum(sm[s]['reuse_s1'] for s in sm) == 96, failed=badr[:5])
    st8 = plans['core-main'][plans['core-main'].action == 'stored']
    res['joint10_stored_8'] = dict(ok=sorted(st8.cand_id) == sorted(K.JOINT10['new']), ids=sorted(st8.cand_id))

    shutil.rmtree(DRILL, ignore_errors=True); (DRILL / 'out').mkdir(parents=True)
    shutil.copy(FREEZE_FORMAL, DRILL / 'freeze.json')
    calls = []

    def stub(t, bd, dom_g, parts, reuse, out, led, fail_at=None):
        if fail_at and t['cand_id'] == fail_at:
            raise RuntimeError('constructed technical failure')
        R = _raster(t); peak = CS.peak_for(t, R); ids = CS.identities(t, _B, parts, peak)
        for stage, kind in (('images', 'render'), ('raw', 'decode')):
            tok = led.start(kind, t['image_key'] if kind == 'render' else t['decode_key']); led.end(tok, 'ok')
        for stage, key, ident in (('images', t['image_key'], ids['image']), ('raw', t['decode_key'], ids['decode']), ('eval', t['decode_key'], ids['eval'])):
            p = Path(out) / stage / f'{key}.json'; h = CS._save_npz(p.with_suffix('.npz'), a=np.zeros(1)); K.save_json(p, dict(identity=ident, npz_sha256=h, complete=True, constructed=True))
        calls.append((t['cand_id'], t['surface'])); return dict(state='ok', calls=dict(render=1, decode=1))
    dr = {}
    for s in FORMAL_SETS:
        A = plans[s]; n_new_all = int((A.action != 'reuse_s1').sum())
        dcaps = dict(render_normal=n_new_all, decode_normal=n_new_all)
        calls.clear(); fail = A[A.action != 'reuse_s1'].cand_id.iloc[len(A) // 3]
        try:
            run_set(s, case_fn=lambda *a, **k: stub(*a, fail_at=fail), out=DRILL / 'out', ledger_path=DRILL / 'ledger.jsonl', freeze_path=DRILL / 'freeze.json', lock_path=DRILL / 'run.lock', is_drill=True, drill_caps=dcaps)
            stopped = False
        except Stop as e:
            stopped = e.code == 16
        n_before = len(calls); tal1 = SetLedger(s, {}, DRILL / 'ledger.jsonl').tally()
        r2 = run_set(s, case_fn=stub, out=DRILL / 'out', ledger_path=DRILL / 'ledger.jsonl', freeze_path=DRILL / 'freeze.json', lock_path=DRILL / 'run.lock', is_drill=True, drill_caps=dcaps)
        tal2 = SetLedger(s, {}, DRILL / 'ledger.jsonl').tally(); visited = [c for c, _ in calls]
        reuse_ids = set(A[A.action == 'reuse_s1'].cand_id)
        order_surfaces = list(dict.fromkeys(sf for _, sf in calls))
        r3 = run_set(s, case_fn=stub, out=DRILL / 'out', ledger_path=DRILL / 'ledger.jsonl', freeze_path=DRILL / 'freeze.json', lock_path=DRILL / 'run.lock', is_drill=True, drill_caps=dcaps)
        dr[s] = dict(technical_failure_stops=stopped, reached_end=(visited[-1] == A[A.action != 'reuse_s1'].cand_id.iloc[-1]), all_new_visited_once=(len(set(visited)) == len(visited) == n_new_all),
                     reuse_never_called=not (set(visited) & reuse_ids), counts_inherited=(tal2['render']['normal'] == tal1['render']['normal'] + (n_new_all - n_before)),
                     rerun_zero_calls=(r3['done_new'] == 0), surfaces_in_order=order_surfaces, stopped_after=n_before)
        dr[s]['ok'] = all(v_ for k, v_ in dr[s].items() if k not in ('surfaces_in_order', 'stopped_after'))

    L = SetLedger('core-main', dict(render_normal=1, decode_normal=1), DRILL / 'cap.jsonl'); tok = L.start('render', 'x'); L.end(tok, 'ok')
    try:
        SetLedger('core-main', dict(render_normal=1, decode_normal=1), DRILL / 'cap.jsonl').start('render', 'y'); cap_ok = False
    except Stop as e:
        cap_ok = e.code == 16
    rt = SetLedger('core-main', dict(render_normal=1, decode_normal=1), DRILL / 'cap.jsonl')
    n_retry_ok = 0
    for _ in range(6):
        try:
            tk = rt.start('render', 'x'); rt.end(tk, 'failed'); n_retry_ok += 1
        except Stop:
            break
    dr['caps_persist_and_retry_cap'] = dict(ok=bool(cap_ok and n_retry_ok == RETRY_CAP['render']), cap_stop=cap_ok, retries_allowed=n_retry_ok)

    fz2 = K.load_json(DRILL / 'freeze.json'); fz2['files'][next(iter(fz2['files']))] = '0' * 64; K.save_json(DRILL / 'freeze_bad.json', fz2)
    dcaps = dict(render_normal=10 ** 6, decode_normal=10 ** 6)
    try:
        run_set('core-main', case_fn=stub, out=DRILL / 'out', ledger_path=DRILL / 'ledger.jsonl', freeze_path=DRILL / 'freeze_bad.json', lock_path=DRILL / 'run.lock', is_drill=True, drill_caps=dcaps); vm = False
    except Stop as e:
        vm = e.code == 12
    dr['freeze_mismatch_refuses'] = dict(ok=vm)
    try:
        plan('optional', v['part_digests']); opt = False
    except Stop as e:
        opt = e.code == 14
    dr['optional_refused'] = dict(ok=opt)
    try:
        check_permit('core-main', plans['core-main'], 'x' * 64); pm = False
    except Stop as e:
        pm = e.code == 14
    dr['permit_mismatch_refused'] = dict(ok=pm)
    shutil.rmtree(DRILL, ignore_errors=True)
    res['drills'] = dict(ok=all(x['ok'] for x in dr.values()), **dr)

    import s2p_report as RP
    rec = RP.records('core-main', plans['core-main'], v['part_digests'])
    j = rec[rec.cand_id.isin(K.JOINT10['new'] + K.JOINT10['reuse'])]
    res['joint10_in_formal_collection'] = dict(ok=bool(len(j) == 10 and j.complete.all() and (j.source.str.startswith('S2 stored').sum() == 8) and (j.source.str.startswith('S1 reuse').sum() == 2)),
                                               sources=j.source.value_counts().to_dict())
    failed = [k for k, x in res.items() if not x.get('ok')]
    outp = dict(task='S2-REAL-01', t_local=K.now_local(), freeze_sha256=v['sha256'], checks=res, n_failed=len(failed), failed=failed,
                note='Isolated stub-backend checks; no rendering.')
    K.save_json(CHECKS_FORMAL, outp)
    return outp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['plan', 'freeze', 'check', 'permit', 'run', 'status'])
    ap.add_argument('--set', default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'freeze':
            r = write_freeze(); print(json.dumps(K.clean(dict(n_files=len(r['files']), parts=r['part_digests'], bridge=r['bridge_joint10'])), ensure_ascii=False, indent=1))
        elif a.cmd == 'plan':
            v = verify_freeze(); print(json.dumps(K.clean(plan_summary(plan(a.set, v['part_digests']))), ensure_ascii=False, indent=1))
        elif a.cmd == 'check':
            r = checks(); print(json.dumps(K.clean(dict(failed=r['failed'], counts=r['checks']['counts']['summary'])), ensure_ascii=False, indent=1))
            if r['n_failed']:
                raise Stop(17, f"{r['n_failed']} checks failed: {r['failed']}")
        elif a.cmd == 'permit':
            print(json.dumps(K.clean(write_permit(a.set)), ensure_ascii=False, indent=1))
        elif a.cmd == 'run':
            if a.set not in FORMAL_SETS:
                raise Stop(14, f'set {a.set} not authorised for the formal run')
            print(json.dumps(K.clean(run_set(a.set)), ensure_ascii=False, indent=1))
        elif a.cmd == 'status':
            print(json.dumps(K.load_json(PROGRESS) if PROGRESS.exists() else {}, ensure_ascii=False, indent=1))
    except Stop as e:
        print(f'STOP [{e.code}] {e}', file=sys.stderr); return e.code
    except BaseException:
        traceback.print_exc(); return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
