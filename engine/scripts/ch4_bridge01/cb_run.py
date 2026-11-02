# Execute Chapter 4 candidate cases and store images, metrics and evaluation arrays.

from __future__ import annotations
import os, sys, json, time, argparse, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import cb_common as G

C, B = G.C, G.B
KINDS = dict(main=('manifest_main.csv', G.CAPS['main']), subray=('manifest_subray.csv', G.CAPS['subray']), holdout=('manifest_holdout.csv', G.CAPS['holdout']))


class Store(C.Store):
    def __init__(self, kind):
        self.kind = kind; super().__init__(G.OUT / kind)

    def reserve(self, r, fid):
        ck = r['case_key']; b = C.canon(C.binding(r, fid))
        G.require(r['action'] == 'new', 'Reused case must never render')
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            G.require(con.execute('SELECT count(*) FROM cases').fetchone()[0] < KINDS[self.kind][1], f'{self.kind} attempt cap reached')
            G.require(con.execute('SELECT 1 FROM cases WHERE ck=?', (ck,)).fetchone() is None, 'Previously attempted case: inspect before retry')
            con.execute('INSERT INTO cases VALUES(?,?,?,?,?)', (ck, b, 'reserved', None, None))
        self.event(ck, 'case', 'reserved')


def check_policy(policy):
    G.require(policy == C.POLICY, 'Selection / evaluation rule changed after registration')
    f = G.OUT / 'freeze.json'
    if f.exists():
        G.require(C.load(f)['policy'] == policy, 'Selection rule differs from the frozen rule')


def verify(kind='main'):
    C.install_guard(); f = C.load(G.OUT / 'freeze.json')
    G.require(C.digest(f['files']) == f['freeze_id'], 'Freeze record digest mismatch'); check_policy(f['policy'])
    C.verify_files(f['files'])
    m = C.csv(G.OUT / KINDS[kind][0])
    if kind == 'main':
        G.require(C.sha(G.OUT / 'manifest_main.csv') == f['manifest_sha256'], 'Main manifest drift')
        import cb_prepare as P
        P.gate_main(m, P.entry_cases())
    else:
        reg = C.load(G.OUT / f'{kind}_registration.json')
        G.require(reg['freeze_id'] == f['freeze_id'] and C.sha(G.OUT / KINDS[kind][0]) == reg['manifest_sha256'], f'{kind} manifest not bound to the freeze')
        G.require(len(m) <= KINDS[kind][1] and not m.case_key.duplicated().any() and not m.forward_key.duplicated().any(), f'{kind} manifest size / keys')
    return f, m


def init_worker(f):
    C.install_guard(); C.verify_files(f['files'])
    C.ALLOWED_PROJECT_READS = {p for p in f['files'] if Path(p).is_relative_to(C.PROJECT)}


def render(pr, rays=1):
    import cb_adapter as A
    S = A.platform(); ch = S['ch']
    if rays == 1:
        img, diag = ch.render(pr); B.sandbox(); return img, diag, pr['forward_key']
    import mp3_forward as F
    G.require(pr['route'] == 'mp3_forward', '3x3 only on the mp3_forward route')
    spec = dict(pr['spec']); spec['rays_per_pixel'] = int(rays)
    p = S['p']
    if pr['sid'] not in p['mesh']:
        import rdy_platform as RP
        p['mesh'][pr['sid']] = RP.s2().build_mesh(pr['sid'])
    out = F.render(p['mesh'][pr['sid']], pr['pattern'], spec); B.sandbox()
    img = out['image']
    G.require(isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.shape == (512, 512) and int(out['diag'].get('rays_per_pixel', 0)) == rays, '3x3 render integrity')
    return img, out['diag'], S['K'].K5.digest(dict(pr['forward_record'], spec=spec))


def subray_key(pr, rays=3):

    import cb_adapter as A
    spec = dict(pr['spec']); spec['rays_per_pixel'] = int(rays)
    return A.platform()['K'].K5.digest(dict(pr['forward_record'], spec=spec))


def work(r, fid, kind):
    import cb_adapter as A
    st = Store(kind)
    try:
        t0 = time.monotonic(); S = A.platform(); ch = S['ch']; ck = r['case_key']
        if kind == 'subray':
            pr = A.prep(dict(r, case_key=r['parent_case_key'], forward_key=r['parent_forward_key'], rays_per_pixel=1)); pr['case_id'] = ck
            G.require(subray_key(pr, int(r['rays_per_pixel'])) == r['forward_key'], '3x3 forward key differs from the registration')
        else:
            pr = A.prep(r)
        cached = st.load_image(r)
        if cached is None:
            st.event(ck, 'render', 'started'); img, diag, fk = render(pr, int(r['rays_per_pixel'])); st.save_image(r, img, diag); st.event(ck, 'render', 'finished')
            G.require(fk == r['forward_key'], 'Rendered forward identity differs')
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
                    scientific=True, backend=f'frozen_DF_P4_{"3x3" if kind == "subray" else "1x1"}', kind=kind)
        st.commit(r, fid, meta, arrays); return dict(ok=True, case_key=ck)
    except BaseException as e:
        st.fail(r['case_key'], e); return dict(ok=False, case_key=r['case_key'], error=repr(e)[:600], trace=traceback.format_exc()[-2500:])


def rows_for(stage, m):
    if stage == 'entry':
        return m[m.entry_subset]
    if stage == 'main':
        return m[(m.action == 'new') & ~m.entry_subset]
    return m[m.action == 'new']


# Run pending cases and record completed outputs.
def execute(stage, workers):
    import run as ENGINE
    kind = 'main' if stage in ('entry', 'main') else stage
    f, m = verify(kind); fid = f['freeze_id']; st = Store(kind)
    if stage == 'main':
        a = C.load(G.OUT / 'entry_audit.json'); G.require(a['ok'] and a['freeze_id'] == fid, 'Real entry not passed')
    if stage == 'subray':
        par = C.csv(G.OUT / 'manifest_main.csv').set_index('case_key'); pst = Store('main')
        for r in m.to_dict('records'):
            G.require(pst.resume(G.rowdict(par.loc[r['parent_case_key']].to_dict() | {'case_key': r['parent_case_key']}), fid) is not None, 'Subray parent not complete: ' + r['parent_case_key'])
    rows = rows_for(stage, m)
    todo = [G.rowdict(r) for r in rows.to_dict('records') if st.resume(G.rowdict(r), fid) is None]
    if not todo:
        print('Already complete; nothing rendered'); return
    lock = C.writable(G.OUT / f'run_{kind}.lock'); fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
    stats = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in f['files']}; count = [0]; start = time.time()
    def before(r):
        for p, s in stats.items():
            v = Path(p).stat(); G.require((v.st_size, v.st_mtime_ns) == s, 'Frozen input changed: ' + p)
        st.reserve(r, fid)
    def after(r, res):
        if not res['ok']:
            C.save(G.OUT / f'technical_stop_{kind}.json', dict(res, t=time.time())); return
        st.resume(r, fid); count[0] += 1
        if count[0] % 50 == 0 or count[0] == len(todo):
            el = time.time() - start
            print(f'{stage} {count[0]}/{len(todo)}; {el:.0f}s; eta {el / count[0] * (len(todo) - count[0]) / 60:.0f} min', flush=True)
            C.save(G.OUT / f'progress_{kind}.json', dict(stage=stage, done=count[0], todo=len(todo), elapsed=el, t=time.time()))
    try:
        ENGINE.bounded_dispatch(todo, work, before, after, lambda r: st.fail(r['case_key'], 'cancelled after technical stop'), 1 if stage == 'entry' else workers,
                                init_worker, (f,), (fid, kind))
        verify(kind); C.save(G.OUT / f'{stage}_execution.json', dict(freeze_id=fid, computed=len(todo), resumed=len(rows) - len(todo), seconds=time.time() - start))
    finally:
        lock.unlink(missing_ok=True)


def result_gate(R, expected):

    C.metric_gate(R, expected)
    for k, g in R.groupby(['config', 'env', 'surface', 'phase_block', 'sigma']):
        G.require(not g.rep.duplicated().any() and set(g.rep) <= ({0} if k[4] == 0 else {1, 2}), 'Repetition structure broken in ' + str(k))
    return True


def pair(a, b, left, right, keys=('surface', 'env', 'sigma', 'phase_block', 'rep', 'seed')):

    keys = list(keys); a = a[a.config == left]; b = b[b.config == right]
    q = a[keys + ['case_key', 'glob_Ctau', 'glob_Cvalid']].merge(b[keys + ['case_key', 'glob_Ctau', 'glob_Cvalid']], on=keys, suffixes=('_left', '_right'),
                                                              validate='one_to_one', how='outer', indicator=True)
    G.require(q._merge.eq('both').all(), f'Unmatched pairs {left} / {right}')
    q['delta_Ctau'] = q.glob_Ctau_right - q.glob_Ctau_left; q['left_config'] = left; q['right_config'] = right
    return q.drop(columns='_merge')


def touch_runtime():

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
    M = C.csv(G.OUT / 'manifest_main.csv')
    for r in M[M.entry_subset].to_dict('records'):
        A.prep(G.rowdict(r))


def collect(kind, m, f):
    st = Store(kind); rows = []; sens = []
    for r in m[m.action == 'new'].to_dict('records'):
        meta = st.resume(G.rowdict(r), f['freeze_id']); G.require(meta is not None and meta.get('scientific') is True, 'Missing new pack: ' + r['case_key'])
        rows.append(meta['result']); sens.extend([dict(case_key=r['case_key'], **v) for v in meta['sensitivity']])
    return pd.DataFrame(rows), pd.DataFrame(sens)


def load_pack(kind, r, fid=None):
    st = Store(kind)
    with np.load(st.path(r['case_key']), allow_pickle=False) as z:
        meta = json.loads(str(z['_meta'])); arr = {k: z[k] for k in z.files if k != '_meta'}
    return meta, arr


def audit_entry():
    import analysis as A0
    f, m = verify('main'); sub = m[m.entry_subset]; R, T = collect('main', sub, f)
    result_gate(R, sub); A0.sensitivity_gate(T, R)
    checks = []
    sys.path.insert(0, str(G.HOME / 'outputs/mp06d_blur'))
    from audit_execution import independent_surface
    for fam in ('point', 'line'):
        r = G.rowdict(sub[(sub.family == fam) & (sub.sigma > 0)].sort_values('case_key').iloc[0]); meta, a = load_pack('main', r); px = r['method'] + '__ev__'
        val = independent_surface(a[px + 'pts'][a[px + 'valid']], r['surface']); q = T[T.case_key == r['case_key']]
        diff = max(max(abs(v.Ctau - val[v.d_mm, v.tau_mm][0]), abs(v.Cvalid - val[v.d_mm, v.tau_mm][1])) for v in q.itertuples())
        G.require(diff < 1e-10, 'Independent surface evaluation mismatch'); checks.append(dict(case_key=r['case_key'], maxdiff=diff))
    per = []
    for r in sub.to_dict('records'):
        r = G.rowdict(r); meta, a = load_pack('main', r); d = meta['diag']; inp = meta['inputs']; res = meta['result']; px = r['method'] + '__ev__'
        G.require(d['rays_per_pixel'] == 1 and d['psf_true_px'] == r['psf_true_px'], 'Entry forward settings')
        G.require((d.get('noise_sha') is None) == (r['sigma'] == 0), 'Zero-noise draw mismatch')
        G.require(inp['raster_key'] == r['raster_key'] and abs(inp['peak'] - r['peak']) < 1e-12 and inp['lit_px'] == r['lit_px'], 'Entry raster / resource mismatch')
        if r['env'] == 'GRAD_Y_REV':
            G.require(.5 <= d['gain_min'] <= d['gain_max'] <= 1, 'Gradient outside [0.5, 1]')
        v = a[px + 'valid'] & a[px + 'in_global']
        rm = float(np.sqrt(np.mean((a[px + 'pts'][v, 2] - a[px + 'zr'][v]) ** 2))) if v.any() else None
        per.append(dict(case_key=r['case_key'], state=res['state'], n_accepted=res['n_accepted'], c=res['c'], w=res['w'], u=res['u'], glob_Ctau=res['glob_Ctau'],
                        glob_pt_rmse=res['glob_pt_rmse'], rmse_from_cloud=rm, peak=inp['peak'], integral=r['integral'], S0=inp['lit_px'], route=r['forward_route'],
                        frac_pre_clip_above_1=d.get('frac_pre_clip_above_1'), received_light_ratio=d.get('received_light_ratio'), seconds=res['seconds']))
    P = pd.DataFrame(per); C.save_csv(G.OUT / 'entry_cases.csv', P)
    fig = entry_figure(sub)
    rec = dict(ok=True, freeze_id=f['freeze_id'], cases=len(R), states=R.state.value_counts().to_dict(), independent_surface=checks, figure=fig,
               mean_seconds=dict(P.assign(fam=sub.family.values).groupby('fam').seconds.mean()))
    C.save(G.OUT / 'entry_audit.json', rec); print(C.canon({k: rec[k] for k in ('ok', 'cases', 'states', 'mean_seconds')}))


def entry_figure(sub):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = sub[sub.sigma > 0].sort_values(['config', 'surface']).to_dict('records')
    fig, ax = plt.subplots(len(rows) // 3, 6, figsize=(18, 3.1 * len(rows) // 3))
    for i, r in enumerate(rows):
        r = G.rowdict(r); meta, a = load_pack('main', r); px = r['method'] + '__ev__'; v = a[px + 'valid'] & a[px + 'in_global']; p = a[px + 'pts'][v]
        y, x = divmod(i, 3); k = 2 * x
        ax[y, k].imshow(a['image'], cmap='gray', vmin=0, vmax=255); ax[y, k].set_axis_off()
        ax[y, k].set_title(f"{G.label(r['config'])}\n{r['surface'].split('_')[0]}, {r['env']}, σ {r['sigma']:g}, Block {r['phase_block']}", fontsize=7)
        if len(p):
            ax[y, k + 1].scatter(p[:, 0], p[:, 1], c=p[:, 2] - a[px + 'zr'][v], cmap='coolwarm', vmin=-.3, vmax=.3, s=1)
        ax[y, k + 1].set(xlim=(-46, 46), ylim=(-46, 46)); ax[y, k + 1].set_aspect('equal'); ax[y, k + 1].tick_params(labelsize=6)
        ax[y, k + 1].set_title(f"n = {len(p)}, Cτ = {meta['result']['glob_Ctau']:.3f}", fontsize=7)
    fig.suptitle('CH4-BRIDGE-01 Eingangsprüfung: Kamerabild und Höhenfehler der gültigen Punkte (−0,3 … 0,3 mm), verrauschte Eingangsfälle', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .97]); p = C.writable(G.OUT / 'figures/entry_check.png'); fig.savefig(p, dpi=110); plt.close(fig)
    return str(p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['entry', 'audit-entry', 'main', 'subray', 'holdout']); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    try:
        if a.stage == 'audit-entry':
            audit_entry()
        else:
            G.require(1 <= a.workers <= 10, 'workers 1..10'); execute(a.stage, a.workers)
    except Exception:
        traceback.print_exc(); sys.exit(2)
