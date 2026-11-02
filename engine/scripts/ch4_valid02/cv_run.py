# Run and collect the Chapter 4 new-phase validation cases.

from __future__ import annotations
import os, sys, json, time, argparse, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import cv_common as V

G, C, B = V.G, V.C, V.B
MAN = V.OUT / 'manifest.csv'
ROOT = V.OUT / 'main'


class Store(C.Store):
    def __init__(self):
        super().__init__(ROOT)

    def reserve(self, r, fid):
        ck = r['case_key']; b = C.canon(C.binding(r, fid))
        G.require(r['action'] == 'new', 'Only registered new inputs are rendered')
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            G.require(con.execute('SELECT count(*) FROM cases').fetchone()[0] < V.CAP, 'Attempt cap reached')
            G.require(con.execute('SELECT 1 FROM cases WHERE ck=?', (ck,)).fetchone() is None, 'Previously attempted case: use the retry stage')
            con.execute('INSERT INTO cases VALUES(?,?,?,?,?)', (ck, b, 'reserved', None, None))
        self.event(ck, 'case', 'reserved')

    def stopped(self):
        with self.connect() as con:
            return con.execute("SELECT ck,error FROM cases WHERE status='technical_stop'").fetchall()

    def retries(self):
        with self.connect() as con:
            return dict(con.execute("SELECT ck,count(*) FROM events WHERE stage='retry' GROUP BY ck").fetchall())

    def dangling(self):

        with self.connect() as con:
            rows = con.execute("SELECT ck FROM cases WHERE status='reserved'").fetchall()
        return [ck for (ck,) in rows if not self.path(ck).exists()]

    def rearm(self, ck):

        G.require(not self.path(ck).exists(), 'Stopped case has a pack; resume it instead')
        with self.connect() as con:
            n = con.execute("UPDATE cases SET status='reserved',error=NULL WHERE ck=? AND status='technical_stop'", (ck,)).rowcount
        G.require(n == 1, 'Retry of a case that is not technically stopped')
        self.event(ck, 'retry', 'image_checkpoint' if self.image_path(ck).exists() else 'render')


def verify():
    import cv_prepare as P
    C.install_guard(); f = C.load(V.OUT / 'freeze.json')
    G.require(C.digest(f['files']) == f['freeze_id'], 'Freeze record digest mismatch')
    G.require(f['policy'] == C.POLICY, 'Selection / evaluation rule changed after registration')
    C.verify_files(f['files'])
    G.require(C.sha(MAN) == f['manifest_sha256'] and C.sha(P.REG) == f['registration_sha256'], 'Manifest / registration drift')
    m = C.csv(MAN); reg = C.load(P.REG); E, _ = P.entry_rows(m); P.gate(m, reg, E)
    return f, m


def init_worker(f):
    C.install_guard(); C.verify_files(f['files'])
    C.ALLOWED_PROJECT_READS = {p for p in f['files'] if Path(p).is_relative_to(C.PROJECT)}


def work(r, fid):

    import cb_adapter as A
    st = Store()
    try:
        t0 = time.monotonic(); S = A.platform(); ch = S['ch']; ck = r['case_key']
        G.require(int(r['rays_per_pixel']) == 1, 'One centre ray per projector pixel')
        pr = A.prep(r)
        cached = st.load_image(r)
        if cached is None:
            st.event(ck, 'render', 'started'); img, diag = ch.render(pr); B.sandbox(); st.save_image(r, img, diag); st.event(ck, 'render', 'finished')
        else:
            img, diag = cached; st.event(ck, 'render', 'reused_checkpoint')
        st.event(ck, 'decode', 'started'); d = ch.decode_eval(pr, img); B.sandbox(); st.event(ck, 'decode', 'finished')
        result = ch.result_row(pr, img, diag, d, time.monotonic() - t0); result.update({k: r[k] for k in C.IDENTITY})
        sens = B.multi_eval(d['ev'], r['surface']); main = next(a for a in sens if a['d_mm'] == 6 and a['tau_mm'] == .15)
        G.require(abs(main['Ctau'] - result['glob_Ctau']) < 1e-10 and abs(main['Cvalid'] - result['glob_Cvalid']) < 1e-10, 'Main evaluation mismatch')
        m = r['method']; arrays = {'image': img}
        arrays.update({m + '__raw__' + k: v for k, v in d['raw'].items()}); arrays.update({m + '__ev__' + k: v for k, v in d['ev'].items()})
        meta = dict(case_key=ck, forward_key=r['forward_key'], method=m, result=result, sensitivity=sens, inputs=pr['inputs'], input_key=pr['input_key'],
                    decode_meta=d['dmeta'], diag=diag, metrics=d['rec']['metrics'], support=d['rec'].get('support'), full_identity=d['rec'].get('full_identity'),
                    scientific=True, backend='frozen_DF_P4_1x1', kind='CV02_VALIDATION')
        st.commit(r, fid, meta, arrays); return dict(ok=True, case_key=ck)
    except BaseException as e:
        st.fail(r['case_key'], e); return dict(ok=False, case_key=r['case_key'], error=repr(e)[:600], trace=traceback.format_exc()[-2500:])


def disk_ok(todo):
    import shutil
    free = shutil.disk_usage(V.OUT).free / 2 ** 30; need = len(todo) * 0.30 / 1024 + 1.0
    G.require(free - need >= 5.0, f'Disk: {free:.1f} GB free, {need:.1f} GB still to write; < 5 GB would remain')
    return dict(free_gb=round(free, 2), still_to_write_gb=round(need, 2), remaining_after_gb=round(free - need, 2))


# Run pending cases and record completed outputs.
def execute(stage, workers):
    import run as ENGINE
    f, m = verify(); fid = f['freeze_id']; st = Store()
    if stage == 'main':
        a = C.load(V.OUT / 'entry_audit.json'); G.require(a['ok'] and a['freeze_id'] == fid, 'Real entry not passed')
    if stage == 'retry':
        G.require(not (V.OUT / 'run.lock').exists(), 'Run lock present: inspect the previous process before a retry')
        for ck in st.dangling():
            st.fail(ck, 'interrupted reservation without pack (process ended)')
        stops = dict(st.stopped()); done = st.retries()
        G.require(not set(stops) & set(done), 'A case failed again after its retry: technical stop, inspect before continuing')
        G.require(sum(done.values()) + len(stops) <= V.RETRY_CAP, 'Retry cap (60 extra attempts) would be exceeded')
        rows = m[m.case_key.isin(stops)]
        for ck in rows.case_key:
            st.rearm(ck)
        todo = [G.rowdict(r) for r in rows.to_dict('records')]
    else:
        G.require(not st.stopped() and not st.dangling(), 'Stopped or interrupted cases exist: run the retry stage after fixing the cause')
        rows = m[m.entry_subset] if stage == 'entry' else m[~m.entry_subset]
        todo = [G.rowdict(r) for r in rows.to_dict('records') if st.resume(G.rowdict(r), fid) is None]
    if not todo:
        print('Already complete; nothing rendered'); return
    space = disk_ok(todo)
    lock = C.writable(V.OUT / 'run.lock'); fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
    stats = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in f['files']}; count = [0]; start = time.time()
    def before(r):
        for p, s in stats.items():
            v = Path(p).stat(); G.require((v.st_size, v.st_mtime_ns) == s, 'Frozen input changed: ' + p)
        if stage != 'retry':
            st.reserve(r, fid)
    def after(r, res):
        if not res['ok']:
            C.save(V.OUT / 'technical_stop.json', dict(res, stage=stage, t=time.time())); return
        st.resume(r, fid); count[0] += 1
        if count[0] % 100 == 0 or count[0] == len(todo):
            el = time.time() - start
            print(f'{stage} {count[0]}/{len(todo)}; {el:.0f}s; eta {el / count[0] * (len(todo) - count[0]) / 60:.0f} min', flush=True)
            C.save(V.OUT / 'progress.json', dict(stage=stage, done=count[0], todo=len(todo), elapsed=el, t=time.time()))
    try:
        ENGINE.bounded_dispatch(todo, work, before, after, lambda r: st.fail(r['case_key'], 'cancelled after technical stop'), workers, init_worker, (f,), (fid,))
        C.save(V.OUT / f'{stage}_execution.json', dict(freeze_id=fid, computed=len(todo), resumed=len(rows) - len(todo), seconds=time.time() - start, workers=workers, disk=space))
    finally:
        lock.unlink(missing_ok=True)


def result_gate(R, expected, full=True):
    C.metric_gate(R, expected)
    for k, g in R.groupby(['config', 'env', 'surface', 'phase_block', 'sigma']):
        want = {0} if k[4] == 0 else {1, 2}
        G.require(not g.rep.duplicated().any() and (set(g.rep) == want if full else set(g.rep) <= want), 'Repetition structure broken in ' + str(k))
    return True


def _drill_job(r):

    if r.get('bad'):
        raise RuntimeError('drill: worker exception')
    return dict(ok=True, case_key=r['case_key'])


def pair(a, b, left, right, keys=('surface', 'env', 'sigma', 'phase_block', 'rep', 'seed', 'method')):

    keys = list(keys); a = a[a.config == left]; b = b[b.config == right]
    G.require(len(a) > 0 and len(a) == len(b), f'Unequal paired sets {left} / {right}')
    G.require(not a.duplicated(keys).any() and not b.duplicated(keys).any(), f'Duplicated pairing key {left} / {right}')
    q = a[keys + ['case_key', 'glob_Ctau', 'glob_Cvalid']].merge(b[keys + ['case_key', 'glob_Ctau', 'glob_Cvalid']], on=keys, suffixes=('_left', '_right'),
                                                              validate='one_to_one', how='outer', indicator=True)
    G.require(q._merge.eq('both').all() and len(q) == len(a), f'Unmatched pairs {left} / {right}')
    q['delta_Ctau'] = q.glob_Ctau_right - q.glob_Ctau_left; q['left_config'] = left; q['right_config'] = right
    return q.drop(columns='_merge')


def touch_runtime(M):

    import cb_adapter as A
    S = A.platform(); p = S['p']
    import rdy_platform as RP
    for sf in G.SURFS:
        sid = S['RM'].SURF_SID[sf]
        if sid not in p['mesh']:
            p['mesh'][sid] = RP.s2().build_mesh(sid)
    import mp3_forward, r_forward, r_fields
    import m6_platform as PL
    PL.platform()
    for r in M[M.entry_subset].to_dict('records'):
        A.prep(G.rowdict(r))


def collect(m, f):
    st = Store(); rows = []; sens = []
    for r in m.to_dict('records'):
        meta = st.resume(G.rowdict(r), f['freeze_id']); G.require(meta is not None and meta.get('scientific') is True, 'Missing pack: ' + r['case_key'])
        rows.append(meta['result']); sens.extend([dict(case_key=r['case_key'], **v) for v in meta['sensitivity']])
    return pd.DataFrame(rows), pd.DataFrame(sens)


def load_pack(r):
    with np.load(Store().path(r['case_key']), allow_pickle=False) as z:
        meta = json.loads(str(z['_meta'])); arr = {k: z[k] for k in z.files if k != '_meta'}
    return meta, arr


def audit_entry():
    import analysis as A0
    import cv_prepare as P
    f, m = verify(); sub = m[m.entry_subset]; R, T = collect(sub, f)
    result_gate(R, sub, full=False); A0.sensitivity_gate(T, R)

    long, surf, ch = A0.tables(R, T, sub[C.IDENTITY]); checks = dict(selection_rows=len(ch), surface_rows=len(surf))
    sys.path.insert(0, str(G.HOME / 'outputs/mp06d_blur'))
    from audit_execution import independent_surface
    ind = []
    for fam in ('point', 'line'):
        r = G.rowdict(sub[(sub.family == fam) & (sub.sigma > 0)].sort_values('case_key').iloc[0]); meta, a = load_pack(r); px = r['method'] + '__ev__'
        val = independent_surface(a[px + 'pts'][a[px + 'valid']], r['surface']); q = T[T.case_key == r['case_key']]
        diff = max(max(abs(v.Ctau - val[v.d_mm, v.tau_mm][0]), abs(v.Cvalid - val[v.d_mm, v.tau_mm][1])) for v in q.itertuples())
        G.require(diff < 1e-10, 'Independent surface evaluation mismatch'); ind.append(dict(case_key=r['case_key'], maxdiff=diff))
    per = []
    for r in sub.to_dict('records'):
        r = G.rowdict(r); meta, a = load_pack(r); d = meta['diag']; inp = meta['inputs']; res = meta['result']; px = r['method'] + '__ev__'
        G.require(d['rays_per_pixel'] == 1 and d['psf_true_px'] == r['psf_true_px'], 'Entry forward settings')
        G.require((d.get('noise_sha') is None) == (r['sigma'] == 0), 'Zero-noise draw mismatch')
        G.require(inp['raster_key'] == r['raster_key'] and abs(inp['peak'] - r['peak']) < 1e-12 and inp['lit_px'] == r['lit_px'], 'Entry raster / resource mismatch')
        G.require(meta['method'] == r['method'] and meta['forward_key'] == r['forward_key'] and meta['case_key'] == r['case_key'], 'Entry method / key mismatch')
        if r['env'] == 'GRAD_Y_REV':
            G.require(.5 <= d['gain_min'] <= d['gain_max'] <= 1, 'Gradient outside [0.5, 1]')
        v = a[px + 'valid'] & a[px + 'in_global']
        rm = float(np.sqrt(np.mean((a[px + 'pts'][v, 2] - a[px + 'zr'][v]) ** 2))) if v.any() else None
        G.require((rm is None and res['glob_pt_rmse'] is None) or abs(rm - res['glob_pt_rmse']) < 2e-6, 'Cloud RMSE differs from the result row')
        G.require(res['state'] != 'no_output' or (res['glob_Ctau'] == 0 and res['glob_pt_rmse'] is None), 'No-output coverage / error')
        per.append(dict(case_key=r['case_key'], config=r['config'], sigma=r['sigma'], route=r['forward_route'], peak=inp['peak'], S0=inp['lit_px'], state=res['state'],
                        n_accepted=res['n_accepted'], c=res['c'], w=res['w'], u=res['u'], glob_Ctau=res['glob_Ctau'], glob_pt_rmse=res['glob_pt_rmse'], rmse_from_cloud=rm,
                        seed=r['seed'], noise_sha=d.get('noise_sha'), seconds=res['seconds']))
    Pd = pd.DataFrame(per)

    q = R[(R.surface == 'M24_curv') & (R.env == 'NOMINAL') & (R.sigma == 40) & (R.phase_block == 'V1')]
    pr = pair(q, q, 'D_r1_p8x4', 'D_r1_T8x4'); G.require(len(pr) == 2, 'Entry pairing group must give exactly two pairs')

    st = Store(); again = [r for r in sub.to_dict('records') if st.resume(G.rowdict(r), f['freeze_id']) is None]
    G.require(not again, 'Resume would re-render entry cases')
    import shutil
    rec = dict(ok=True, freeze_id=f['freeze_id'], cases=len(R), states=R.state.value_counts().to_dict(), routes=Pd.route.value_counts().to_dict(),
               peak_capped=int((Pd.peak >= 1).sum()), integral_limited=int((Pd.peak < 1).sum()), configs=int(Pd.config.nunique()), sigmas=sorted(Pd.sigma.unique().tolist()),
               independent_surface=ind, entry_pairs=len(pr), resume_skips=len(sub), formal_tables=checks, sigma0_noise_free=bool(Pd[Pd.sigma == 0].noise_sha.isna().all()),
               mean_seconds=Pd.seconds.mean(), pack_mb=float(np.mean([Store().path(ck).stat().st_size for ck in sub.case_key]) / 2 ** 20),
               disk_free_gb=shutil.disk_usage(V.OUT).free / 2 ** 30)
    C.save_csv(V.OUT / 'entry_cases.csv', Pd); C.save(V.OUT / 'entry_audit.json', rec)
    print(C.canon({k: rec[k] for k in ('ok', 'cases', 'states', 'routes', 'peak_capped', 'integral_limited', 'entry_pairs', 'mean_seconds', 'pack_mb', 'disk_free_gb')}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['entry', 'audit-entry', 'main', 'retry']); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    try:
        if a.stage == 'audit-entry':
            audit_entry()
        else:
            G.require(1 <= a.workers <= 10, 'workers 1..10'); execute(a.stage, a.workers)
    except Exception:
        traceback.print_exc(); sys.exit(2)
