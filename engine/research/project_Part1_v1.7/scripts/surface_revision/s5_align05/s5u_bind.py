# Bind joint-study inputs and check reusable completed cases.

from __future__ import annotations

import time

import s5u_common as K
import s5r_bind as B
import s5r_gates as G
import s5r_main as E3
import s5r_store as S
import s5t_view as V

K4 = K.K4


class Registry:
    def __init__(self):
        self.m = {}

    @classmethod
    def load(cls, compat_a4: dict, auth_id: str | None, ledger_path=None) -> 'Registry':
        R = cls()
        for k, v in (compat_a4 or {}).get('records', {}).items():
            st, key = k.split('|', 1); R.m[(st, key)] = dict(v, source='ALIGN-04 final hashes')
        if auth_id:
            for e in K.json_lines(ledger_path or K.LEDGER):
                inf = e.get('info')
                if e.get('auth') == auth_id and e.get('status') == 'ok' and isinstance(inf, dict) and inf.get('json_sha256'):
                    R.m[(inf['stage'], e['key'])] = dict(identity=inf['identity'], npz_sha256=inf['npz_sha256'], json_sha256=inf['json_sha256'], source=f'ledger {auth_id}')
        return R


class Ledger5(K4.Ledger):

    def __init__(self, path, auth_id: str, caps: dict, keys: set):
        self.path = K.Path(path); self.auth_id = auth_id; self.caps = dict(caps); self.keys = set(keys)
        K.root_of(self.path)

    def _write(self, e: dict) -> None:
        import json
        import os
        K.root_of(self.path); self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(K.clean(e), ensure_ascii=False) + chr(10)); fh.flush(); os.fsync(fh.fileno())


class BoundStore(S.Store):
    def __init__(self, root, base, registry: Registry):
        super().__init__(root, base); self.reg = registry

    def status(self, stage, key, identity):
        st, why = super().status(stage, key, identity)
        if st != 'reuse':
            return st, why
        b = self.reg.m.get((stage, key))
        if b is None:
            return 'unbound', 'complete record without a registered completion binding (unknown origin)'
        p = self._p(stage, key)
        if b['identity'] != identity:
            return 'mismatch', 'registered identity differs'
        if K.sha(p) != b['json_sha256'] or K.sha(p.with_suffix('.npz')) != b['npz_sha256']:
            return 'corrupt', 'JSON or NPZ differs from the hashes registered at completion'
        return 'reuse', 'identity, NPZ and JSON equal the registered completion'

    def commit_bound(self, stage, key, identity, arrays, meta) -> dict:
        self.commit(stage, key, identity, arrays, meta)
        p = self._p(stage, key)
        b = dict(stage=stage, identity=identity, npz_sha256=K.sha(p.with_suffix('.npz')), json_sha256=K.sha(p))
        self.reg.m[(stage, key)] = dict(b, source='this process')
        return b


def compat_align04(write=True) -> dict:
    a4 = K.load_json(K3reg(K.A4RUN)); fin = a4['final']['outputs']; recs = {}; bad = []
    for stage in ('image', 'decode', 'eval'):
        for p in sorted((K.FORMAL_STORE / stage).glob('*.json')):
            rj, rz = K.rel(p), K.rel(p.with_suffix('.npz'))
            if rj not in fin or rz not in fin:
                bad.append(f'{rj}: not in the ALIGN-04 final record'); continue
            if K.sha(p) != fin[rj] or K.sha(p.with_suffix('.npz')) != fin[rz]:
                bad.append(f'{rj}: differs from the ALIGN-04 final hashes'); continue
            recs[f'{stage}|{p.stem}'] = dict(identity=K.load_json(p)['identity'], json_sha256=fin[rj], npz_sha256=fin[rz], json=rj)
    led = K.LEDGER.read_text(encoding='utf-8').splitlines() if K.LEDGER.exists() else []
    a4_auth = a4['authorisation']['auth_id']; a4_lines = [x for x in led if f'"auth": "{a4_auth}"' in x]
    out = dict(note='compatibility mapping of the 8 ALIGN-04 images (32 stage records) to the ALIGN-04 final hashes; records not modified, not re-evaluated',
               n_records=len(recs), records=recs, a4_auth=a4_auth, a4_ledger_events=len(a4_lines), a4_ledger_digest=K.digest(a4_lines, 32), problems=bad)
    if len(recs) != 32 or bad or len(a4_lines) != 64:
        raise K.Stop('version', f'ALIGN-04 compatibility not closed: {len(recs)} records, {len(a4_lines)} ledger events', detail=bad[:10])
    if write:
        K.run_fix('compat_align04', out)
    return out


def K3reg(p):
    return K.K3.reg(p)


def check_compat_align04(cp4: dict) -> dict:
    bad = []
    for k, v in cp4['records'].items():
        st, key = k.split('|', 1); p = K.FORMAL_STORE / st / f'{key}.json'
        if not p.exists() or K.sha(p) != v['json_sha256'] or K.sha(p.with_suffix('.npz')) != v['npz_sha256']:
            bad.append(k)
    led = K.LEDGER.read_text(encoding='utf-8').splitlines() if K.LEDGER.exists() else []
    lines = [x for x in led if f'"auth": "{cp4["a4_auth"]}"' in x]
    if K.digest(lines, 32) != cp4['a4_ledger_digest']:
        bad.append('ALIGN-04 ledger events changed')
    if bad:
        raise K.Stop('version', 'ALIGN-04 records or ledger events differ from their registration', detail=bad[:10])
    return dict(ok=True, n=len(cp4['records']))


def check_compat_align03() -> dict:
    a3 = K.load_json(K3reg(K.A3RUN)); cp = a3['compat']; src = a3['sources_read']
    vb = B.verify_binding(cp)
    if not vb['ok']:
        raise K.Stop('version', 'S5-ALIGN-03 binding of the existing records failed', detail=vb)
    files = [r['eval']['json'] for r in cp['records'].values()] + [d['json'] for d in cp['df'].values()] + \
        ['outputs/surface_revision/s345_ready/results/S5-main_cases.csv', 'outputs/surface_revision/s345_ready/results/S5-gauss_cases.csv',
         'outputs/surface_revision/s345_ready/results/S5_cases_supplement.csv']
    bad = [f for f in files if f not in src or K.sha(K.ROOT / f) != src[f]]
    if bad:
        raise K.Stop('version', 'existing JSON / tables consumed by the statistics differ from their S5-ALIGN-03 registration', detail=bad[:10])
    return dict(ok=True, n_records=vb['n_checked'], n_json_tables=len(files))


def methods_of(row) -> list:
    return ['D0', 'DF'] if row['family'] == 'point' else ['P4']


def rows_of(keys, M) -> list:
    rows = E3.manifest_rows(keys, M)
    for r, k in zip(rows, keys):
        r.update(combo=k[0], phase_label=k[2], rep=int(k[4]))
    return rows


def complete_keys(store: BoundStore, rows: list, pr: dict) -> tuple[set, list]:

    done, probs = set(), []
    for r in rows:
        k = r['key']; iid = S.image_identity(r['f'], pr); st, why = store.status('image', k, iid)
        if st == 'todo':
            continue
        if st != 'reuse':
            probs.append(dict(key=k, stage='image', status=st, why=why)); continue
        ok = True; isha = store.content('image', k)
        for m in methods_of(r):
            dk = f'{k}__{m}'; did = S.decode_identity(iid, isha, m, S.method_params(m, r['f']['peak_f32'], r.get('radius_mm')))
            s2, w2 = store.status('decode', dk, did)
            if s2 != 'reuse':
                ok = False
                if s2 != 'todo':
                    probs.append(dict(key=dk, stage='decode', status=s2, why=w2))
                continue
            s3, w3 = store.status('eval', dk, S.eval_identity(store.content('decode', dk), pr))
            if s3 != 'reuse':
                ok = False
                if s3 != 'todo':
                    probs.append(dict(key=dk, stage='eval', status=s3, why=w3))
        if ok:
            done.add(tuple([r['combo'], r['surface'], r['phase_label'], float(r['sigma']), int(r['rep'])]))
    return done, probs


def census(store: BoundStore, allowed_keys: set) -> dict:

    unknown, unbound = [], []
    for stage in ('image', 'decode', 'eval'):
        d = store.root / stage
        for p in (d.glob('*.json') if d.exists() else []):
            mk = p.stem.split('__')[0]
            if mk not in allowed_keys:
                unknown.append(K.rel(p))
            elif (stage, p.stem) not in store.reg.m:
                rec = K.load_json(p)
                if rec.get('complete') is True:
                    unbound.append(K.rel(p))
    if unknown or unbound:
        raise K.Stop('unbound', 'store holds records of unknown origin / without completion binding', detail=dict(unknown=unknown[:10], unbound=unbound[:10]))
    return dict(ok=True)


def make_auth() -> dict:
    t = K3reg(K.TASK); M = K.read_csv(K3reg(K.K3.MANIFEST)); pr = S.platform_refs()
    cp4 = K.run_load().get('compat_align04') or compat_align04()
    reg = Registry.load(cp4, None); store = BoundStore(K.FORMAL_STORE, K.FORMAL_ROOT, reg)
    new_keys = [tuple(x) for x in M[M.status == 'to_image'][['combo', 'surface', 'phase_label', 'sigma', 'rep']].itertuples(index=False, name=None)]
    rows = rows_of(new_keys, M); done, probs = complete_keys(store, rows, pr)
    if probs:
        raise K.Stop('unbound', 'chain problems while deriving the complete set', detail=probs[:10])
    a4keys = {tuple(x) for x in K.load_json(K.A4RUN)['authorisation']['keys']}
    if done != a4keys:
        raise K.Stop('scope', f'verified complete set ({len(done)}) differs from the accepted ALIGN-04 keys ({len(a4keys)})')
    remaining = sorted(k for k in new_keys if k not in done)
    rem_rows = [r for r in rows if (r['combo'], r['surface'], r['phase_label'], float(r['sigma']), int(r['rep'])) in set(remaining)]
    n_dot = sum(1 for r in rem_rows if r['family'] == 'point'); n_line = len(rem_rows) - n_dot
    derived = dict(render=len(rem_rows), decode_D0=n_dot, decode_DF=n_dot, decode_P4=n_line, evaluate=2 * n_dot + n_line)
    if derived != K.CAPS:
        raise K.Stop('auth', f'derived quotas {derived} differ from the task book {K.CAPS}')
    auth = dict(task_book=str(t), task_book_sha256=K.sha(t), derived_from='manifest_next.csv to_image keys minus the content-verified complete set (ALIGN-04 next8)',
                n_keys=len(remaining), keys_digest=K.digest([list(k) for k in remaining], 32), manifest_keys_digest=K.digest(sorted(r['manifest_key'] for r in rem_rows), 32),
                caps=K.CAPS, manifest=K.rel(K.K3.MANIFEST), manifest_sha256=K.sha(K.K3.MANIFEST), store=K.rel(K.FORMAL_STORE), ledger=K.rel(K.LEDGER),
                excluded_complete=sorted([list(k) for k in done]), compat_align04_digest=K.digest(cp4['records'], 32))
    auth['auth_id'] = 'AUTH5-' + K.digest(auth, 16)
    K.run_fix('authorisation', auth)
    return dict(auth_id=auth['auth_id'], n_keys=auth['n_keys'], caps=auth['caps'])


def load_auth() -> dict:
    a = K.run_load().get('authorisation')
    if not a:
        raise K.Stop('auth', 'no authorisation (run `auth`)')
    if K.sha(a['task_book']) != a['task_book_sha256']:
        raise K.Stop('auth', 'task book hash differs from the authorisation')
    chk = dict(a); aid = chk.pop('auth_id')
    if 'AUTH5-' + K.digest(chk, 16) != aid or a['caps'] != K.CAPS:
        raise K.Stop('auth', 'authorisation edited after its derivation')
    return a


def remaining_keys(a: dict, M) -> list:

    excl = {tuple(x) for x in a['excluded_complete']}
    keys = sorted(tuple(x) for x in M[M.status == 'to_image'][['combo', 'surface', 'phase_label', 'sigma', 'rep']].itertuples(index=False, name=None) if tuple(x) not in excl)
    if K.digest([list(k) for k in keys], 32) != a['keys_digest'] or len(keys) != a['n_keys']:
        raise K.Stop('auth', 'remaining key set differs from the authorised set')
    return keys


def check_freeze(fr: dict | None = None) -> dict:
    f = fr if fr is not None else K.run_load().get('freeze')
    if not f:
        raise K.Stop('freeze', 'no freeze recorded')
    changed = [p for p, h in f['files'].items() if not (K.ROOT / p).exists() or K.sha(K.ROOT / p) != h]
    if changed:
        raise K.Stop('freeze', f'{len(changed)} frozen files changed since the freeze', detail=changed[:10])
    return dict(ok=True, n_files=len(f['files']), digest=f['digest'])


def preflight(purpose: str, need_freeze: bool = True) -> dict:
    t0 = time.time(); chk = {}
    chk['freeze'] = check_freeze() if need_freeze else dict(ok=None)
    a = load_auth(); chk['auth'] = dict(ok=True, auth_id=a['auth_id'], n_keys=a['n_keys'])
    M = K.read_csv(K3reg(K.K3.MANIFEST))
    if K.sha(K.K3.MANIFEST) != a['manifest_sha256']:
        raise K.Stop('auth', 'manifest differs from the authorised manifest')
    gv = G.gate_manifest(M, 'full', None, G.registered())
    if not gv['ok']:
        raise K.Stop('refused', 'manifest gate failed', detail=gv['by_check'])
    chk['manifest_gate'] = dict(ok=True, n_rows=gv['n_rows'])
    keys = remaining_keys(a, M); chk['keys'] = dict(ok=True, n=len(keys))
    cp4 = K.run_load().get('compat_align04'); chk['compat_align04'] = check_compat_align04(cp4)
    chk['compat_align03'] = check_compat_align03()
    dep = B.dependency_state()
    if not dep['ok']:
        raise K.Stop('version', 'method / evaluation dependencies differ from their freezes', detail=dep)
    chk['dependencies'] = dict(ok=True, n=len(dep['files']))
    reg = Registry.load(cp4, a['auth_id']); store = BoundStore(K.FORMAL_STORE, K.FORMAL_ROOT, reg)
    rows = rows_of(keys, M); allowed = {r['manifest_key'] for r in rows} | {k.split('|', 1)[1].split('__')[0] for k in cp4['records']}
    chk['census'] = census(store, allowed)
    chk['seconds'] = round(time.time() - t0, 1); chk['purpose'] = purpose
    return dict(checks=chk, auth=a, M=M, keys=keys, rows=rows, store=store, cp4=cp4)


# Execute missing joint-study stages after checking dependency closure and stored identities.
def run_formal(store: BoundStore, rows: list, ex, pr: dict, ledger, progress_every: int = 20, closure_files: set | None = None) -> dict:
    log = dict(reuse=0, computed=0, images_done=0, closure_checked=[]); t0 = time.time(); n_rows = len(rows)

    def stage(kind, quota_kind, rec_key, ident, fn, row):
        st, why = store.status(kind, rec_key, ident)
        if st in ('mismatch', 'corrupt', 'unbound'):
            raise K.Stop(st if st != 'unbound' else 'unbound', f'{kind} {rec_key}: {why}')
        if st == 'reuse':
            log['reuse'] += 1; return False
        tok = ledger.reserve(quota_kind, rec_key, row['manifest_key'])
        try:
            a, m = fn()
            b = store.commit_bound(kind, rec_key, ident, a, m)
        except BaseException as e:
            ledger.close(tok, quota_kind, rec_key, 'failed', repr(e)[:300]); raise
        ledger.close(tok, quota_kind, rec_key, 'ok', b); log['computed'] += 1
        return True
    with store.lock():
        store.set_status('running', n_rows=n_rows, auth=ledger.auth_id)
        try:
            for i, row in enumerate(rows, 1):
                k = row['key']; iid = S.image_identity(row['f'], pr)
                stage('image', 'render', k, iid, lambda: ex.render(row), row)
                isha = store.content('image', k)
                for method in methods_of(row):
                    dk = f'{k}__{method}'
                    did = S.decode_identity(iid, isha, method, S.method_params(method, row['f']['peak_f32'], row.get('radius_mm')))
                    stage('decode', f'decode_{method}', dk, did, lambda: ex.decode(row, method, store.arrays('image', k)), row)
                    eid = S.eval_identity(store.content('decode', dk), pr)

                    def _ev():
                        row['_decode_meta'] = K.load_json(store._p('decode', dk))['meta']
                        return ex.evaluate(row, method, store.arrays('decode', dk))
                    stage('eval', 'evaluate', dk, eid, _ev, row)
                log['images_done'] = i
                if closure_files is not None and row['family'] not in log['closure_checked'] and log['computed']:
                    extra = sorted(set(K.loaded_project_files()) - closure_files)
                    if extra:
                        raise K.Stop('freeze', 'loaded project files outside the frozen closure', detail=extra[:10])
                    log['closure_checked'].append(row['family'])
                if i % progress_every == 0 or i == n_rows:
                    el = time.time() - t0
                    eta = (n_rows - i) * el / i if i else None
                    store.set_status('running', n_rows=n_rows, images_done=i, computed=log['computed'], reuse=log['reuse'], seconds=round(el, 1), images_per_min=(round(60 * i / el, 2) if el > 0 else None),
                                     eta_seconds=(round(eta) if eta is not None else None), auth=ledger.auth_id)
                    print(f'[{K.now()}] {i}/{n_rows} images, computed {log["computed"]}, reuse {log["reuse"]}, {el / 60:.1f} min, eta {eta / 60 if eta else 0:.1f} min', flush=True)
            store.set_status('complete', n_rows=n_rows, computed=log['computed'], reuse=log['reuse'], auth=ledger.auth_id, seconds=round(time.time() - t0, 1))
        except K.Stop as e:
            store.set_status('failed', code=e.code, reason=e.msg[:300], images_done=log['images_done'], auth=ledger.auth_id); raise
        except Exception as e:
            store.set_status('failed', code='technical', reason=repr(e)[:300], images_done=log['images_done'], auth=ledger.auth_id)
            raise K.Stop('technical', repr(e)[:300]) from e
    log['seconds'] = round(time.time() - t0, 1)
    return log


def view_rows(store: BoundStore, rows: list, pr: dict) -> tuple[list, list]:

    return V.store_rows(store, rows, pr, methods_of)
