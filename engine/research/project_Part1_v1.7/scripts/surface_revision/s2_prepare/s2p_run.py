# Execute spacing-layout cases and store completed results.

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import pandas as pd

import s2p_common as K

K.setup()
AUTHORISED_SETS = ('joint10',)


class Stop(Exception):
    def __init__(self, code, msg):
        super().__init__(msg); self.code = code


def load_list() -> pd.DataFrame:
    import s2p_manifest as MF
    return pd.read_csv(MF.CSV, keep_default_na=False, float_precision='round_trip')


def set_rows(d: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == 'joint10':
        ids = K.JOINT10['new'] + K.JOINT10['reuse']
        x = d[d.cand_id.isin(ids)]
        if sorted(x.cand_id) != sorted(ids) or (x.tier != 'core').any():
            raise Stop(16, 'joint10 whitelist does not resolve to 10 core rows')
        return x
    sets = {'core-main': (d.tier == 'core') & (d.surface_class == 'main'), 'core-gauss': (d.tier == 'core') & (d.surface_class == 'supplement'), 'optional': d.tier == 'optional'}
    if name not in sets:
        raise Stop(3, f'unknown set {name}')
    return d[sets[name]]


class Lock:
    def __init__(self, p=K.LOCK):
        self.p = K.guard(p); self.fd = None

    def __enter__(self):
        self.p.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise Stop(16, f'run lock present: {K.rel(self.p)} (another run or an interrupted one; inspect before removing)') from e
        os.write(self.fd, f'{os.getpid()} {K.now_local()}'.encode()); return self

    def __exit__(self, *exc):
        os.close(self.fd); self.p.unlink(missing_ok=True)
        return False


class Ledger:

    def __init__(self, p=K.LEDGER, caps=None):
        self.p = K.guard(p); self.caps = caps or {}

    def events(self):
        if not self.p.exists():
            return []
        return [json.loads(l) for l in self.p.read_text(encoding='utf-8').splitlines() if l.strip()]

    def count(self, kind):
        return sum(1 for e in self.events() if e.get('event') == 'start' and e.get('kind') == kind)

    def _w(self, rec):
        self.p.parent.mkdir(parents=True, exist_ok=True)
        with open(self.p, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(K.clean(dict(rec, t_local=K.now_local())), ensure_ascii=False) + '\n'); fh.flush(); os.fsync(fh.fileno())

    def start(self, kind, key):
        n = self.count(kind)
        if kind in self.caps and n + 1 > self.caps[kind]:
            raise Stop(16, f'{kind} cap {self.caps[kind]} reached ({n} attempts recorded); no further real call')
        tok = f'{kind}:{key}:{os.getpid()}:{len(self.events())}'
        self._w(dict(event='start', kind=kind, key=key, token=tok, n_before=n, cap=self.caps.get(kind))); return tok

    def end(self, tok, outcome, detail=None):
        self._w(dict(event='end', token=tok, outcome=outcome, detail=detail))

    def note(self, **kw):
        self._w(dict(event='note', **kw))


def _permit_expect(set_name):
    import s2p_manifest as MF
    st = K.platform_state()
    return dict(kind='production', set=set_name, cand_ids=sorted(K.JOINT10['new'] + K.JOINT10['reuse']), caps=K.CAPS_JOINT10, candidates_sha256=K.sha_file(MF.CSV),
                config_sha256=K.sha_file(MF.CONFIG), freeze_sha256=(K.sha_file(K.FREEZE) if K.FREEZE.exists() else None), platform_version=st['version'])


def cmd_permit(a):
    import s2p_freeze as FZ
    if a.set not in AUTHORISED_SETS:
        raise Stop(14, f'set {a.set} is not authorised by S2-PREP-02')
    v = FZ.verify()
    if not v['ok']:
        raise Stop(12, f"freeze drift / missing: {v.get('reason') or v['drift'][:5]}")
    ck = K.OUT / 'checks.json'
    c = K.load_json(ck) if ck.exists() else {}
    if c.get('n_failed') != 0 or c.get('t_local', '') < K.load_json(K.FREEZE)['created_local']:
        raise Stop(17, 'preparation checks must pass AFTER the freeze before a permit is written')
    task = K.ROOT.parents[1] / 'protocols' / 'S2-PREP-02_protocol.md'
    p = _permit_expect(a.set)
    p.update(authorised_by=f'S2-PREP-02 task book ({K.rel(task)}, sha256 {K.sha_file(task)}) and the user instruction of 2026-10-12; written by the executor (execution) after passed checks; NOT an independent acceptance',
             checks_sha256=K.sha_file(ck), created_local=K.now_local())
    K.save_json(K.PERMIT, p); print(json.dumps(K.clean(p), ensure_ascii=False, indent=1))


def check_permit(path, set_name):
    if not path or not os.path.exists(path):
        raise Stop(14, 'no permit')
    p = K.load_json(path); exp = _permit_expect(set_name); bad = [k for k, v in exp.items() if K.clean(p.get(k)) != K.clean(v)]
    if bad or not p.get('authorised_by'):
        raise Stop(14, f'permit does not match: {bad}')
    return p


def first_case_check(out, t) -> dict:

    import numpy as np
    import s2p_case as CS
    r = dict()
    ip = out / 'images' / f"{t['image_key']}.json"; rp = out / 'raw' / f"{t['decode_key']}.json"; ep = out / 'eval' / f"{t['decode_key']}.json"
    r['records_complete'] = CS.complete(ip) and CS.complete(rp) and CS.complete(ep)
    with np.load(ip.with_suffix('.npz')) as z:
        img = z['image']; r['image_uint8_512'] = bool(img.dtype == np.uint8 and img.shape == (512, 512))
    with np.load(ep.with_suffix('.npz')) as z:
        E = {k: np.asarray(z[k]) for k in z.files}
    m = K.load_json(ep)['metrics']
    pf = E['point_feature']; r['correspondence_index_closed'] = bool(len(pf) == len(E['pts']) and (pf >= 0).all() and E['feat_point_idx'][pf].tolist() == list(range(len(pf))))
    r['both_domains'] = all(k in E for k in ('in_local', 'in_global', 'loc_node_supported', 'glob_node_supported')) and m.get('loc_n_nodes') == 22605 and m.get('glob_n_nodes') == 30947
    v = E['valid']; z = E['pts'][:, 2]; neg = E['accepted'] & np.isfinite(z) & (z < 0) & (z >= K.WINDOW[0])
    r['negative_z_accepted'] = int(neg.sum()); r['negative_z_kept_valid'] = bool((neg & ~v & np.isfinite(E['zr'])).sum() == 0)
    r['identity_closure'] = int(E['point_correct'].sum() + E['point_wrong'].sum() + E['point_unconf'].sum()) == int(E['accepted'].sum())
    r['raw_has_component_reasons'] = 'comp_reason' in np.load(rp.with_suffix('.npz')).files
    r['ok'] = all(r.values())
    return r


def run_loop(set_name: str, permit: dict):
    import adp_binding as BD
    import s1r_common as S1K
    import s2p_case as CS
    import s2p_freeze as FZ
    import s2p_reuse as RU
    v = FZ.verify()
    if not v['ok']:
        raise Stop(12, f"freeze drift: {v['drift'][:5]} {v['mesh_drift']}")
    st = K.platform_state()
    if not st['bound_ok']:
        raise Stop(12, f"platform not bound / drift {st.get('dependency_drift')}")
    d = load_list(); rows = set_rows(d, set_name)
    led = Ledger(caps=permit['caps']); out = K.OUT / 'run'
    with Lock():
        led.note(msg='run start', set=set_name, freeze_sha256=v['freeze_sha256'], renders_before=led.count('render'), decodes_before=led.count('decode'))
        T, _ = RU.build(d); reuse = {r['image_key']: r for r in T.to_dict('records')}
        for cid in K.JOINT10['reuse']:
            t = rows[rows.cand_id == cid].iloc[0].to_dict(); lay = RU.verify_layers(reuse[t['image_key']])
            led.note(msg='reuse check', cand_id=cid, **{k: lay[k] for k in ('image_ok', 'decode_ok', 'eval_ok')}, reasons=lay['reasons'])
            if not (lay['image_ok'] and lay['decode_ok'] and lay['eval_ok']):
                raise Stop(16, f'reuse case {cid} no longer valid: {lay["reasons"]} (no fallback to new imaging)')
        bd = BD.bind(required_methods=sorted(set(K.ARM.values()))); dom_g = S1K.global_domain(0.5)
        order = [K.FIRST_REAL] + [c for c in K.JOINT10['new'] if c != K.FIRST_REAL]
        res = []
        for i, cid in enumerate(order):
            t = rows[rows.cand_id == cid].iloc[0].to_dict()
            try:
                r = CS.run_case(t, bd, dom_g, v['part_digests'], None, out, led)
            except Stop:
                raise
            except Exception as e:
                led.note(msg='technical stop', cand_id=cid, error=repr(e)[:400]); raise Stop(16, f'technical stop at {cid}: {e!r}') from e
            res.append(dict(cand_id=cid, **r)); led.note(msg='case done', cand_id=cid, state=r['state'], calls=r['calls'], evaluated=r['evaluated'])
            if i == 0:
                fc = first_case_check(out, t); led.note(msg='first case check', **fc)
                if not fc['ok']:
                    raise Stop(16, f'first real case check failed: {fc}')
        led.note(msg='run end', renders=led.count('render'), decodes=led.count('decode'))
    return res


def cmd_run(a):
    if a.set not in AUTHORISED_SETS:
        raise Stop(14, f'set {a.set} not authorised in S2-PREP-02')
    p = check_permit(a.permit, a.set)
    res = run_loop(a.set, p)
    led = Ledger()
    print(json.dumps(dict(cases=res, renders_total=led.count('render'), decodes_total=led.count('decode')), ensure_ascii=False, indent=1))


def cmd_collect(a):
    import s2p_collect as CO
    print(json.dumps(K.clean(CO.collect(a.set)), ensure_ascii=False, indent=1))


def cmd_manifest(a):
    import s2p_manifest as MF
    df, bud, cfg = MF.write()
    print(json.dumps(K.clean(dict(core=cfg['counts']['core']['total'], optional=cfg['counts']['optional']['total'], rasters=cfg['budget']['n_unique_rasters'])), ensure_ascii=False))


def cmd_reuse(a):
    import s2p_reuse as RU
    T, s = RU.write(load_list()); print(json.dumps(K.clean(s), ensure_ascii=False, indent=1))


def cmd_plan(a):
    import s2p_reuse as RU
    d = load_list(); T, _ = RU.build(d); ready = set(T[T.image_status == 'ready'].image_key)
    x = set_rows(d, a.set); imgs = set(x.image_key)
    print(json.dumps(dict(set=a.set, images=len(imgs), reuse_ready=len(imgs & ready), to_render=len(imgs - ready), decodes=int(x.decode_key.nunique())), ensure_ascii=False))


def cmd_check(a):
    import s2p_checks as CK
    r = CK.main()
    if r['n_failed']:
        raise Stop(17, f"{r['n_failed']} checks failed: {r['failed']}")


def cmd_freeze(a):
    import s2p_freeze as FZ
    s = FZ.write(); print(json.dumps(dict(n_files=len(s['files']), meshes=s['meshes'], part_digests=s['part_digests']), ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['manifest', 'reuse', 'plan', 'check', 'freeze', 'permit', 'run', 'collect'])
    ap.add_argument('--set', default='joint10'); ap.add_argument('--permit', default=None)
    a = ap.parse_args(argv)
    fn = dict(manifest=cmd_manifest, reuse=cmd_reuse, plan=cmd_plan, check=cmd_check, freeze=cmd_freeze, permit=cmd_permit, run=cmd_run, collect=cmd_collect)
    try:
        fn[a.cmd](a)
    except Stop as e:
        print(f'STOP {e}', file=sys.stderr); return e.code
    except Exception:
        traceback.print_exc(); return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
