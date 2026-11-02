# Run size-spacing cases with checkpointed rendering, decoding and evaluation.

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_common as K

faulthandler.enable()
STAGES = {'main_s1': 0, 'main_s2': 1, 'gauss_s1': 2, 'gauss_s2': 3}
LOG = K.OUT / 'run' / 'run.log'


def log(msg, path=None):
    line = f'{K.now_local()} {msg}'; print(line, flush=True)
    with open(path or LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


class Ledger:


    def __init__(self, path: Path, caps: dict | None):
        self.path = Path(path); self.caps = caps; self.count = {}; self.open = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for i, ln in enumerate(self.path.read_text(encoding='utf-8').splitlines()):
                if ln.strip():
                    try:
                        e = json.loads(ln)
                    except json.JSONDecodeError as ex:
                        raise K.Stop('technical_stop', f'corrupt ledger line {i}: {ex}') from ex
                    if e.get('event') == 'start':
                        self.count[e['kind']] = self.count.get(e['kind'], 0) + 1

    def n(self, kind):
        return self.count.get(kind, 0)

    def start(self, kind, key, purpose='new'):
        if self.caps is not None and self.n(kind) + 1 > int(self.caps[kind]):
            raise K.Stop('cap', f'{kind} cap {self.caps[kind]} exhausted (attempts {self.n(kind)})')
        tok = f'{kind}:{key}:{self.n(kind) + 1}'
        self._w(dict(event='start', kind=kind, key=key, purpose=purpose, token=tok)); self.count[kind] = self.n(kind) + 1
        return tok

    def end(self, tok, outcome, detail=None):
        self._w(dict(event='end', token=tok, outcome=outcome, detail=detail))

    def _w(self, rec):
        with open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(K.clean(dict(rec, t_local=K.now_local())), ensure_ascii=False) + '\n'); fh.flush(); os.fsync(fh.fileno())


class NoLedger:


    def start(self, kind, key, purpose='new'):
        return f'{kind}:{key}'

    def end(self, tok, outcome, detail=None):
        return None


class Lock:
    def __init__(self, p=None):
        self.p = Path(p or K.LOCK); self.fd = None

    def __enter__(self):
        try:
            self.fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise K.Stop('lock', f'run lock present: {K.rel(self.p)} (another run or a crashed run; inspect before removing)') from e
        os.write(self.fd, f'{os.getpid()} {K.now_local()}'.encode()); return self

    def __exit__(self, *exc):
        os.close(self.fd); self.p.unlink(missing_ok=True)
        return False


def peak_mem_mb() -> dict:
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL('kernel32'); psapi = ctypes.WinDLL('psapi')
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return dict(peak_ws_mb=round(c.PeakWorkingSetSize / 2 ** 20), peak_commit_mb=round(c.PeakPagefileUsage / 2 ** 20))
    except Exception:
        return {}


_W = {}


def _init_worker():
    import s1r_eval
    S = K.s2(); _W['bd'] = K.bound(); _W['dom_g'] = K.platform_ids()['s1k'].global_domain(0.5); _W['dom_l'] = {}; _W['S'] = S


def _work(u: dict) -> dict:
    try:
        if not _W:
            _init_worker()
        import s12_case as C
        sid = u['platform_surface']
        if sid not in _W['dom_l']:
            _W['dom_l'][sid] = _W['S'].roi_domain(_W['S'].build_mesh(sid))
        t0 = time.time()
        r = C.run_new(u, _W['bd'], _W['dom_g'], _W['dom_l'][sid], NoLedger())
        r['wall_s'] = round(time.time() - t0, 3); r['mem'] = peak_mem_mb(); r['pid'] = os.getpid()
        return r
    except K.Stop as e:
        raise RuntimeError(f'{e.code}: {e}') from None


def load_inputs() -> pd.DataFrame:
    U = pd.read_csv(K.INPUTS, keep_default_na=False, float_precision='round_trip')
    return U


def pending(U: pd.DataFrame, keys=None, stage=None) -> list[dict]:
    rows = U[U.source == 'new']
    if keys is not None:
        rows = rows[rows.input_key.isin(keys)]
    if stage is not None:
        rows = rows[rows.stage == STAGES[stage]]
    out = []
    for r in rows.to_dict('records'):
        if K.complete(K.EVAL_DIR / f"{r['decode_key']}.json", r['eval_key']):
            continue
        r['need_render'] = not K.complete(K.IMG_DIR / f"{r['input_key']}.json", r['input_key'])
        r['need_decode'] = not K.complete(K.RAW_DIR / f"{r['decode_key']}.json", r['decode_key'])
        out.append(r)
    order = {s: i for i, s in enumerate(K.SURFACES)}
    out.sort(key=lambda t: (t['stage'], order[t['surface']], t['family'], t['layout'], t['phase_label'], t['sigma'], t['rep'], t['size_mm']))
    return out


# Schedule pending cases and record successful or failed computation stages.
def run_tasks(tasks: list[dict], workers: int, label: str, ledger: Ledger, work=None) -> dict:

    work = work or _work
    t0 = time.time(); done = []; failed = None
    toks = {}

    def begin(t):
        tk = []
        if t['need_render']:
            tk.append(('render', ledger.start('render', t['input_key'], label)))
        if t['need_decode'] or t['need_render']:
            tk.append(('decode', ledger.start('decode', t['decode_key'], label)))
        toks[t['input_key']] = tk

    def finish(t, res):
        for kind, tok in toks.pop(t['input_key']):
            ledger.end(tok, 'ok' if res['calls'].get(kind) else 'not_called', dict(state=res['state']))
        done.append(res)
        if len(done) % 100 == 0 or len(done) == len(tasks):
            el = time.time() - t0
            log(f"  {label}: {len(done)}/{len(tasks)} done, {el:.0f} s, eta {el / len(done) * (len(tasks) - len(done)):.0f} s")

    def fail(t, e):
        for kind, tok in toks.pop(t['input_key'], []):
            ledger.end(tok, 'failed_or_unknown', repr(e)[:300])

    if workers <= 1:
        for t in tasks:
            begin(t)
            try:
                res = work(t)
            except Exception as e:
                fail(t, e); failed = (t, e); break
            finish(t, res)
    else:
        it = iter(tasks); running = {}
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            def submit():
                t = next(it, None)
                if t is None:
                    return False
                begin(t); running[ex.submit(work, t)] = t
                return True
            for _ in range(workers):
                if not submit():
                    break
            while running:
                fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in fin:
                    t = running.pop(fut)
                    try:
                        res = fut.result()
                    except Exception as e:
                        fail(t, e)
                        failed = failed or (t, e)
                        continue
                    finish(t, res)
                    if failed is None:
                        submit()
    secs = time.time() - t0
    if failed is not None:
        t, e = failed
        log(f"TECHNICAL STOP at {t['input_key']} ({t['surface']} {t['family']} {t['layout']} size {t['size_mm']} sigma {t['sigma']} {t['phase_label']} r{t['rep']}): {e!r}")
        raise K.Stop('technical_stop', f"worker failure at {t['input_key']}: {e!r}")
    return dict(n=len(done), wall_s=round(secs, 1), per_case_wall_s=(round(secs / len(done), 3) if done else None),
                parts_mean_s={k: float(np.mean([d['seconds'][k] for d in done])) for k in ('render', 'decode', 'eval')} if done else {},
                peak_commit_mb_max=max([d.get('mem', {}).get('peak_commit_mb', 0) for d in done] or [0]), states={s: sum(1 for d in done if d['state'] == s) for s in ('ok', 'no_output')})


def cmd_manifest(a):
    if K.PROTOCOL.exists():
        raise K.Stop('usage', 'protocol frozen: the manifest is not rewritten after the freeze')
    import s12_manifest as M
    V, U, RT, c = M.write()
    log(f"manifest: views {c['counts']['views_total']}, unique {c['counts']['unique_total']}, new {c['counts']['new']}, reuse {c['counts']['reuse']}, problems {len(c['problems'])}")
    if c['problems']:
        raise K.Stop('matrix_drift', f"manifest problems {c['problems'][:4]}")


def cmd_reuse(a):
    import s12_reuse as RU
    r = RU.verify()
    log(f"reuse: {r['n_reused']} inputs, all three layers {r['all_three_layers']}, failed {r['n_failed']}, source files {r.get('source_files_hashed')}")
    if not r['all_three_layers']:
        raise K.Stop('reuse_refused', f"{r['n_failed']} stored inputs fail a layer")


def cmd_freeze(a):
    import s12_freeze as FZ
    if K.PROTOCOL.exists():
        raise K.Stop('usage', 'protocol already frozen (not rewritten)')
    p = FZ.write()
    log(f"freeze: protocol written, first batch {len(p['first_batch']['input_keys'])} inputs, run code {list(p['dependencies']['run_code'])}")


REPLAY_RULE = ('<= 12 stored inputs, deterministic: per family one S1 fixed-peak and one S1 equal-integral (dots 5850 dark, lines 11695) case and one S2 case on the three '
               'main surfaces, sigma 0 and 40, plus one S2 no-output case (dense lines s 4.45) and one low-recovery dot case')


def replay_selection(U: pd.DataFrame, V: pd.DataFrame) -> list[str]:
    R = U[(U.source != 'new') & (U.surface_class == 'main')].copy()
    sel = []

    def pick(q, n=1):
        q = q.sort_values(['surface', 'input_key'])
        for k in q.input_key:
            if k not in sel:
                sel.append(k); n -= 1
                if n == 0:
                    return

    vv = V.set_index('input_key')
    for fam, size in (('point', 1.5), ('line', 1.0)):
        for reg, surf, sig in (('fixed_peak_1', 'M10_curv', 0.0), ('equal_integral_5850' if fam == 'point' else 'resource_cap_11695', 'M24_curv', 40.0)):
            ks = V[(V.section == 'S1') & (V.family == fam) & (V.regime == reg) & (V.surface == surf) & (V.sigma == sig) & (V.size_mm == size) & (V.source != 'new')].input_key
            pick(R[R.input_key.isin(ks)])
        ks = V[(V.section == 'S2') & (V.family == fam) & (V.surface == 'B_curv') & (V.sigma == 40.0) & (V.source == 's2')].input_key
        pick(R[R.input_key.isin(ks)])
    return sel


def cmd_replay(a):
    import s12_case as C
    import s12_freeze as FZ
    FZ.verify()
    U = load_inputs(); V = pd.read_csv(K.MANIFEST, keep_default_na=False, float_precision='round_trip')
    sel = replay_selection(U, V)

    import s12_reuse as RU
    extra = []
    for r in U[(U.source == 's2') & (U.surface_class == 'main') & (U.family == 'line') & (U.layout == 's4.45_phi0') & (U.sigma == 0.0)].sort_values('input_key').itertuples():
        if K.load_json(RU.eval_record_path(r))['metrics']['state'] == 'no_output':
            extra.append(r.input_key); break
    for r in U[(U.source == 's1') & (U.surface_class == 'main') & (U.family == 'point') & (U.size_mm == 0.5) & (U.sigma == 100.0)].sort_values('input_key').itertuples():
        m = K.load_json(RU.eval_record_path(r))['metrics']
        if m['state'] == 'ok' and (m.get('glob_recovery') or 0) < 0.2:
            extra.append(r.input_key); break
    sel = (sel + extra)[:12]
    S = K.s2(); bd = K.bound(); dom_g = K.platform_ids()['s1k'].global_domain(0.5)
    led = Ledger(K.LEDGER, K.load_json(K.PROTOCOL)['caps'])
    out = []
    with Lock():
        for k in sel:
            u = U[U.input_key == k].iloc[0].to_dict()
            dom_l = S.roi_domain(S.build_mesh(u['platform_surface']))
            r = C.replay(u, bd, dom_g, dom_l, led); out.append(r)
            log(f"replay {k} {u['surface']} {u['family']} {u['regimes']} s{u['sigma']:g}: ok {r['ok']} (pts {r['points_max_abs_diff_mm']}, scalar {r['scalar_max_abs_diff']:.2e}, state {r['state_old']}->{r['state_new']})")
    res = dict(t_local=K.now_local(), rule=REPLAY_RULE, n=len(out), all_equal=all(r['ok'] for r in out),
               tolerances=dict(masks_and_int_arrays='exact (stored dtype)', points=f'rtol 0, atol {C.REPLAY_PTS_ATOL} mm (stored dtype)', scalars=f'abs {C.REPLAY_SCALAR_ATOL}',
                               undefined='None pattern equal'), cases=out)
    K.save_json(K.CHECKS / 'replay_check.json', res)
    if not res['all_equal']:
        raise K.Stop('platform_mismatch', 'replay differs from the stored records (checks/replay_check.json)')


def _stage_complete(U, st) -> bool:
    x = U[(U.source == 'new') & (U.stage == st)]
    return all(K.complete(K.EVAL_DIR / f'{r.decode_key}.json', r.eval_key) for r in x.itertuples())


def cmd_first(a):
    import s12_freeze as FZ
    FZ.verify()
    if not (K.CHECKS / 'replay_check.json').exists() or not K.load_json(K.CHECKS / 'replay_check.json')['all_equal']:
        raise K.Stop('usage', 'replay check missing or not equal')
    if not (K.CHECKS / 'drills.json').exists() or not K.load_json(K.CHECKS / 'drills.json')['all_ok']:
        raise K.Stop('usage', 'drills missing or failed')
    U = load_inputs(); keys = K.load_json(K.PROTOCOL)['first_batch']['input_keys']
    tasks = pending(U, keys=keys)
    led = Ledger(K.LEDGER, K.load_json(K.PROTOCOL)['caps'])
    log(f'first batch: {len(tasks)} pending of {len(keys)}; ledger render {led.n("render")}, decode {led.n("decode")}')
    with Lock():
        r = run_tasks(tasks, a.workers, 'first batch', led)
    chk = first_case_checks(U[U.input_key.isin(keys)])
    K.save_json(K.CHECKS / 'first_batch.json', dict(t_local=K.now_local(), timing=r, checks=chk, ok=chk['ok']))
    log(f"first batch: {r['n']} cases, {r['wall_s']} s ({r['per_case_wall_s']} s/case wall, workers {a.workers}), parts {r['parts_mean_s']}, states {r['states']}, checks ok {chk['ok']}")
    if not chk['ok']:
        raise K.Stop('technical_stop', f"first-batch checks failed: {chk['problems'][:4]}")


def first_case_checks(U1: pd.DataFrame) -> dict:

    probs = []; rows = []
    for r in U1.itertuples():
        ip = K.IMG_DIR / f'{r.input_key}.json'; ep = K.EVAL_DIR / f'{r.decode_key}.json'
        if not (K.complete(ip, r.input_key) and K.complete(K.RAW_DIR / f'{r.decode_key}.json', r.decode_key) and K.complete(ep, r.eval_key)):
            probs.append(f'{r.input_key}: records incomplete'); continue
        ir = K.load_json(ip); er = K.load_json(ep); m = er['metrics']
        if abs(ir['peak_f32'] - float(np.float32(r.peak_f32))) > 0 or ir['peak_f32'] > 1.0:
            probs.append(f'{r.input_key}: peak')
        with np.load(ep.with_suffix('.npz')) as z:
            v = np.asarray(z['valid'], bool); p = np.asarray(z['pts'], np.float64)
        if v.any() and not ((p[v, 2] >= K.WINDOW[0] - 1e-9) & (p[v, 2] <= K.WINDOW[1] + 1e-9)).all():
            probs.append(f'{r.input_key}: valid point outside the window')
        if m['id_correct'] + m['id_wrong'] + m['id_unconf'] != m['id_denominator']:
            probs.append(f'{r.input_key}: identity counts not closed')
        for pre in ('loc', 'glob'):
            if f'{pre}_Ctau' not in m:
                probs.append(f'{r.input_key}: {pre} domain missing')
        if m['state'] == 'no_output' and (m['glob_Ctau'] != 0 or m['glob_pt_rmse'] is not None):
            probs.append(f'{r.input_key}: no-output semantics')
        rows.append(dict(input_key=r.input_key, surface=r.surface, family=r.family, layout=r.layout, size_mm=r.size_mm, sigma=r.sigma, peak_f32=r.peak_f32, state=m['state'], n_valid=m['n_valid'],
                         n_valid_negative_z=int((v & (p[:, 2] < 0)).sum()) if len(p) else 0, glob_Ctau=m['glob_Ctau'], loc_Ctau=m['loc_Ctau'], glob_pt_rmse=m['glob_pt_rmse'],
                         id=(m['id_correct'], m['id_wrong'], m['id_unconf'], m['id_denominator']), end_grey=ir.get('end_grey')))
    return dict(ok=not probs, problems=probs, cases=rows)


def cmd_run(a):
    import s12_freeze as FZ
    FZ.verify()
    if not (K.CHECKS / 'first_batch.json').exists() or not K.load_json(K.CHECKS / 'first_batch.json')['ok']:
        raise K.Stop('usage', 'first batch missing or not ok')
    U = load_inputs(); st = STAGES[a.stage]
    for s_prev in range(st):
        if not _stage_complete(U, s_prev):
            raise K.Stop('usage', f'stage {s_prev} not complete (order main S1 -> main S2 -> Gauss S1 -> Gauss S2)')
    tasks = pending(U, stage=a.stage)
    led = Ledger(K.LEDGER, K.load_json(K.PROTOCOL)['caps'])
    log(f"run {a.stage}: {len(tasks)} pending (workers {a.workers}); ledger render {led.n('render')}, decode {led.n('decode')}")
    with Lock():
        r = run_tasks(tasks, a.workers, a.stage, led)
    FZ.verify()
    tm = K.CHECKS / 'run_timing.json'
    allt = K.load_json(tm) if tm.exists() else {}
    allt[a.stage] = dict(r, t_local=K.now_local(), workers=a.workers); K.save_json(tm, allt)
    log(f"run {a.stage} done: {r['n']} cases, {r['wall_s']} s ({r['per_case_wall_s']} s/case wall), parts {r['parts_mean_s']}, states {r['states']}, peak commit {r['peak_commit_mb_max']} MB")


def cmd_status(a):
    U = load_inputs()
    n = U[U.source == 'new']
    led = Ledger(K.LEDGER, None)
    st = {s: dict(new=int((n.stage == i).sum()), complete=int(sum(K.complete(K.EVAL_DIR / f'{r.decode_key}.json', r.eval_key) for r in n[n.stage == i].itertuples()))) for s, i in STAGES.items()}
    print(json.dumps(dict(unique=int(len(U)), reused=int((U.source != 'new').sum()), new=int(len(n)), stages=st, ledger=dict(render=led.n('render'), decode=led.n('decode'), replay_decode=led.n('replay_decode'))),
                     ensure_ascii=False))


# Test resuming, incomplete-file rejection and execution limits with isolated cases.
def cmd_drills(a):
    import shutil
    import subprocess
    import s12_reuse as RU
    root = K.OUT / 'drills'
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    U = load_inputs(); res = {}

    q = pending(U)
    keys = [t['input_key'] for t in q]
    shared = U[(U.source == 'new') & (U.sections == 'S1;S2')]
    calls = {}

    def stub(t):
        calls[t['input_key']] = calls.get(t['input_key'], 0) + 1
        return dict(input_key=t['input_key'], state='ok', calls=dict(render=1, decode=1), seconds=dict(render=0.0, decode=0.0, eval=0.0))
    led1 = Ledger(root / 'ledger_1.jsonl', dict(render=10 ** 6, decode=10 ** 6))
    sub = [t for t in q if t['input_key'] in set(shared.input_key)][:40] + [t for t in q if t['input_key'] not in set(shared.input_key)][:40]
    run_tasks(sub, 1, 'drill-shared', led1, work=stub)
    res['shared_once'] = dict(ok=(len(keys) == len(set(keys)) and len(keys) == int((U.source == 'new').sum()) and max(calls.values()) == 1 and all(t['stage'] in (0, 2) for t in q if t['input_key'] in set(shared.input_key))),
                              queue=len(keys), unique_new=int((U.source == 'new').sum()), shared_new=int(len(shared)), max_calls_per_input=max(calls.values()))

    led2 = Ledger(root / 'ledger_2.jsonl', dict(render=10 ** 6, decode=10 ** 6)); finished = set(); n_calls = [0]

    def stub2(t):
        n_calls[0] += 1
        if n_calls[0] == 6:
            raise RuntimeError('simulated interruption')
        finished.add(t['input_key'])
        return dict(input_key=t['input_key'], state='ok', calls=dict(render=1, decode=1), seconds=dict(render=0.0, decode=0.0, eval=0.0))
    q2 = [dict(t) for t in q[:10]]
    try:
        run_tasks(q2, 1, 'drill-resume', led2, work=stub2); crashed = False
    except K.Stop:
        crashed = True
    led2b = Ledger(root / 'ledger_2.jsonl', dict(render=10 ** 6, decode=10 ** 6))
    remaining = [t for t in q2 if t['input_key'] not in finished]
    run_tasks(remaining, 1, 'drill-resume-2', led2b, work=lambda t: dict(input_key=t['input_key'], state='ok', calls=dict(render=1, decode=1), seconds=dict(render=0.0, decode=0.0, eval=0.0)))
    led2c = Ledger(root / 'ledger_2.jsonl', None)
    res['resume_counts'] = dict(ok=crashed and led2c.n('render') == 11 and len(remaining) == 5, crashed=crashed, render_starts_after_resume=led2c.n('render'),
                                expected='10 tasks + 1 interrupted attempt = 11 starts; the 5 finished inputs are not queued again')

    r1 = U[U.source == 's1'].iloc[0]; src = RU.eval_record_path(r1)
    dst = root / 'reuse_copy'; dst.mkdir()
    shutil.copy(src, dst / src.name); shutil.copy(src.with_suffix('.npz'), dst / src.with_suffix('.npz').name)
    ok_before = K.complete(dst / src.name)
    with open(dst / src.with_suffix('.npz').name, 'r+b') as fh:
        fh.seek(100); b = fh.read(1); fh.seek(100); fh.write(bytes([(b[0] + 1) % 256]))
    ok_corrupt = K.complete(dst / src.name)
    (dst / src.with_suffix('.npz').name).unlink(); ok_missing = K.complete(dst / src.name)
    fake = dict(K.load_json(K.PROTOCOL)) if K.PROTOCOL.exists() else None
    drift_refused = None
    if fake is not None:
        import s12_freeze as FZ
        fake['dependencies']['run_code']['s12_case.py'] = '0' * 64
        fp = root / 'protocol_fake.json'; K.save_json(fp, fake)
        orig = K.PROTOCOL
        try:
            K.PROTOCOL = fp
            try:
                FZ.verify(check_s2=False); drift_refused = False
            except K.Stop:
                drift_refused = True
        finally:
            K.PROTOCOL = orig
    res['reuse_refusal'] = dict(ok=bool(ok_before and not ok_corrupt and not ok_missing and drift_refused), intact=ok_before, corrupted_refused=not ok_corrupt, missing_refused=not ok_missing,
                                hash_drift_refused=drift_refused)

    p = subprocess.run([sys.executable, '-B', str(Path(__file__)), 'drill-crash'], capture_output=True, text=True, timeout=600)
    res['nonzero_exit'] = dict(ok=p.returncode != 0, returncode=p.returncode, tail=p.stderr[-300:])
    res['all_ok'] = all(v['ok'] for v in res.values())
    res['t_local'] = K.now_local(); res['note'] = 'isolated root outputs/surface_revision/s12_resource01/drills (stub workers; no imaging / decoding)'
    K.save_json(K.CHECKS / 'drills.json', res)
    log(f"drills: {dict((k, v['ok']) for k, v in res.items() if isinstance(v, dict))} -> {res['all_ok']}")
    if not res['all_ok']:
        raise K.Stop('technical_stop', 'drill failed')


def cmd_drill_crash(a):
    try:
        raise RuntimeError('drill: simulated entry exception')
    finally:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='S12-RESOURCE-01 entry')
    ap.add_argument('cmd', choices=['manifest', 'reuse', 'freeze', 'drills', 'replay', 'first', 'run', 'status', 'drill-crash'])
    ap.add_argument('--stage', default='main_s1', choices=list(STAGES))
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args(argv)
    fn = dict(manifest=cmd_manifest, reuse=cmd_reuse, freeze=cmd_freeze, drills=cmd_drills, replay=cmd_replay, first=cmd_first, run=cmd_run, status=cmd_status)
    fn['drill-crash'] = cmd_drill_crash
    try:
        fn[a.cmd](a)
    except K.Stop as e:
        log(f'STOP ({e.code}) {e}')
        return e.exit_code
    except Exception:
        traceback.print_exc()
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
