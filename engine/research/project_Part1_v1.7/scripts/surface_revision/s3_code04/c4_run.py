# Run encoded and uncoded decoding arms on the registered images.

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

import c4_common as K

KAPPA_A3 = 6.05 / np.sqrt((0.75 / 0.18) * (1.25 / 0.18))

RULES = dict(dots=[('B0', 'dot', dict(rule='B0')), ('DF', 'dot', dict(rule='DF')), ('DS', 'dot', dict(rule='DS'))],
             lines_coded=[('ignore', 'line', dict(rule='ignore')), (f'R@{KAPPA_A3:.4f}', 'line', dict(rule='R', kappa=float(KAPPA_A3))),
                          ('LB@1.0000', 'line', dict(rule='LB', kappa=1.0))],
             lines_plain=[('P4', 'p4', {})])
DIAG_ARMS = dict(dots_coded=[('TC_src', 'source'), ('TI_src', 'source'), ('TC', 'bridge_geom'), ('TI', 'bridge_geom')],
                 dots_plain=[('TI_src', 'source'), ('TI', 'bridge_geom')], lines_coded=[('TC', 'lines_v1'), ('TI', 'lines_v1')], lines_plain=[])


def rules_for(row: dict) -> list:
    if row['family'] == 'dots':
        return RULES['dots']
    return RULES['lines_plain'] if row['pattern'] == 'plain' else RULES['lines_coded']


def arms_for(row: dict, method: str) -> list:
    if method == 'dot':
        return ['single'] if row['pattern'] == 'plain' else ['use', 'ignore']
    return None


def diag_for(row: dict, rule: str) -> list:
    if row['family'] == 'dots':
        return DIAG_ARMS['dots_plain' if row['pattern'] == 'plain' else 'dots_coded']
    return DIAG_ARMS['lines_coded'] if (row['pattern'] != 'plain' and rule == 'ignore') else []


def render_contract(row: dict, inp: dict) -> tuple:
    sid = K.RK.SURF_MAP.get(row['surface'])
    if sid is None:
        raise K.Stop('technical', f"unknown surface {row['surface']}")
    peak = float(row['peak_f32']); sg = float(row['sigma']); seed = K.render_seed(sg, row.get('seed'))
    pat = np.asarray(inp['pattern'])
    if not (np.isfinite(peak) and 0.0 < peak <= 1.0):
        raise K.Stop('technical', f'peak {peak} outside (0, 1]')
    if pat.dtype != np.float32 or pat.shape != (512, 512) or not np.isfinite(pat).all() or float(pat.max()) > peak + 1e-7:
        raise K.Stop('technical', 'pattern contract violated (float32 512x512, finite, max <= peak)')
    if sg < 0 or not np.isfinite(sg):
        raise K.Stop('technical', f'sigma {sg}')
    return sid, pat, peak, sg, seed


def backend_real(row: dict, inp: dict, permit: Path | None):

    sid, pat, peak, sg, seed = render_contract(row, inp)
    K.require_permit(permit)
    import rdy_platform as P
    bd = P.bound()
    res = bd.call('render', bd.sid(row['surface']), pat, peak, sg, seed, K.RK.PSF_PX)
    img = np.asarray(res['image'])
    if img.dtype != np.uint8 or img.shape != (512, 512):
        raise K.Stop('technical', f'render returned {img.dtype} {img.shape}')
    return img


def row_contract(row: dict) -> None:

    if K.RK.SURF_MAP.get(row['surface']) is None:
        raise K.Stop('technical', f"{row['image_key']}: unknown surface {row['surface']}")
    peak = float(row['peak_f32']); sg = float(row['sigma'])
    if not (np.isfinite(peak) and 0.0 < peak <= 1.0):
        raise K.Stop('technical', f"{row['image_key']}: peak {peak} outside (0, 1]")
    if sg < 0 or not np.isfinite(sg):
        raise K.Stop('technical', f"{row['image_key']}: sigma {sg}")
    K.render_seed(sg, row.get('seed'))


def backend_replay(row: dict, image_map: dict):
    src = image_map.get(row['image_key'])
    if src is None:
        raise K.Stop('technical', f"replay: no stored image for {row['image_key']}")
    return src


def _load_decode(dp: Path) -> tuple[dict, dict]:
    r = K.load_json(dp)
    meta = K.load_json(K.ROOT / r['source_json'])['meta'] if r.get('kind') == 'ref' else r['meta']
    return K.load_npz(K.record_npz(dp)), meta


# Render or load one image and evaluate its registered decoder and class-use arms.
def run_unit(row: dict, root: Path, backend: str, fps: dict, led_file: Path, image_map: dict | None = None, permit=None, truth=None) -> dict:

    import c3_diag as DG
    import c3_eval as EV
    import c4_store as ST
    t0 = time.monotonic(); ik = row['image_key']; fam = row['family']; done = dict(key=ik, image='stored', method_exec=0, detection=0, eval=0, diag_recon=0)

    def led(kind, status, **kw):
        K.ledger(led_file, kind, ik, status, **kw)
    inp = ST.inputs_of(row)

    def image_of(key, r_):
        p = root / 'images' / f'{key}.json'
        if K.complete(p, key):
            return K.load_npz(K.record_npz(p))['image']
        if backend == 'replay':
            src = backend_replay(dict(r_, image_key=key), image_map)
            K.save_json(p, dict(identity=key, kind='ref', source_npz=src['npz'], source_sha256=src['sha256'], backend='replay', complete=True, t_local=K.now_local()))
            led('image_ref', 'ok', image=key); return K.load_npz(K.ROOT / src['npz'])['image']
        render_contract(r_, inp if key == ik else ST.inputs_of(r_))
        K.require_permit(permit)
        led('render_attempt', 'start', image=key)
        try:
            img_ = backend_real(r_, inp if key == ik else ST.inputs_of(r_), permit)
        except Exception as e:
            led('technical_failure', 'render_failed', error=repr(e)[:300]); raise
        h = K.save_npz(p.with_suffix('.npz'), image=img_)
        K.save_json(p, dict(identity=key, kind='own', backend='real', npz_sha256=h, complete=True, t_local=K.now_local())); led('render_attempt', 'ok', image=key); done['image'] = 'rendered'
        return img_
    img = image_of(ik, row)
    ref_img = None
    if fam == 'dots':
        rk = row['ref_image_key']; rp = root / 'images' / f'{rk}.json'
        if K.complete(rp, rk):
            ref_img = K.load_npz(K.record_npz(rp))['image']
        elif backend == 'replay':
            ref_img = image_of(rk, dict(row, image_key=rk, sigma=0.0, seed=None))
        else:
            raise K.Stop('technical', f'sigma-0 reference {rk} of {ik} missing (the sigma-0 wave runs first; never rendered inside a dependent unit)')
    ctx = EV.ctx_of(row, inp); F_cache = {}; line_recs = {}; mesh = None
    for name, method, params in rules_for(row):
        dk = K.decode_key(ik, fam, method, name, params, fps['decode']); dp = root / 'dec' / f'{dk}.json'
        if K.complete(dp, dk):
            A, meta = _load_decode(dp)
        else:
            A, meta = ST.decode_one(img, inp, row, method, params, F_cache); h = K.save_npz(dp.with_suffix('.npz'), **A)
            K.save_json(dp, dict(identity=dk, kind='own', image_key=ik, rule=name, params=params, fp=fps['decode'], meta=meta, npz_sha256=h, complete=True, t_local=K.now_local()))
            led('method_exec', 'ok', rule=name); done['method_exec'] += 1
            if method == 'line' and 'F' in F_cache and done['detection'] == 0:
                done['detection'] = 1; led('detection', 'ok', rule='line_front (shared by the coded rules)')
        D = ST.unpack(fam, A, meta)
        if method == 'dot':
            arms = arms_for(row, method); keys = {a: K.eval_key(dk, a, fps['eval']) for a in arms}
            miss = [a for a in arms if not K.complete(root / 'eval' / f'{keys[a]}.json', keys[a])]
            if miss:
                res = EV.eval_dots(img, ref_img, inp, row, D, ctx)
                for a in miss:
                    rowm, arrays = res[a]; ep = root / 'eval' / f'{keys[a]}.json'; h = K.save_npz(ep.with_suffix('.npz'), **arrays)
                    K.save_json(ep, dict(identity=keys[a], kind='own', decode_key=dk, image_key=ik, rule=name, arm=a, row=rowm, npz_sha256=h, complete=True, t_local=K.now_local()))
                done['eval'] += len(miss); led('eval', 'ok', rule=name, n=len(miss))

            dl = diag_for(row, name); dkeys = {(a, rt): K.diag_key(dk, a, rt, fps['diag']) for a, rt in dl}
            dmiss = [(a, rt) for (a, rt), v in dkeys.items() if not K.complete(root / 'diag' / f'{v}.json', v)]
            if dmiss:
                import rdy_platform as P
                mesh = mesh or P.mesh(row['surface']); inp_t = _truth_inputs(inp, truth); ref_t = _truth_ref(ref_img, truth)
                res = {}
                if any(rt == 'source' for a, rt in dmiss):
                    sm = DG.source_map(D, ref_t, inp_t, mesh); res.update({(a, 'source'): v for a, v in DG.eval_src(row, inp_t, D, sm, ctx, mesh).items()})
                if any(rt == 'bridge_geom' for a, rt in dmiss):
                    res.update({(a, 'bridge_geom'): v for a, v in DG.bridge_dot_eval(row, inp_t, D, ctx, mesh).items()})
                for a, rt in dmiss:
                    rowm, arrays = res[(a, rt)]; v = dkeys[(a, rt)]; gp = root / 'diag' / f'{v}.json'; h = K.save_npz(gp.with_suffix('.npz'), **arrays)
                    K.save_json(gp, dict(identity=v, kind='own', decode_key=dk, image_key=ik, rule=name, arm=a, reference=rt, row=rowm, npz_sha256=h, complete=True,
                                         truth=('replaced (drill)' if truth else 'diagnostic truth'), t_local=K.now_local()))
                done['diag_recon'] += len(dmiss); led('diag_recon', 'ok', rule=name, n=len(dmiss), arms=[f'{a}/{rt}' for a, rt in dmiss])
        else:
            line_recs[name] = (dk, D)
    if fam == 'lines':
        keys = {n: K.eval_key(dk, n, fps['eval']) for n, (dk, _) in line_recs.items()}
        miss = [n for n in keys if not K.complete(root / 'eval' / f'{keys[n]}.json', keys[n])]
        kO = conf = None
        if miss:
            res, kO, conf = EV.eval_lines(row, inp, {n: r for n, (dk, r) in line_recs.items()}, ctx)
            for n in miss:
                rowm, arrays = res[n]; ep = root / 'eval' / f'{keys[n]}.json'; h = K.save_npz(ep.with_suffix('.npz'), **arrays)
                K.save_json(ep, dict(identity=keys[n], kind='own', decode_key=line_recs[n][0], image_key=ik, rule=n, arm=n, row=rowm, npz_sha256=h, complete=True, t_local=K.now_local()))
            done['eval'] += len(miss); led('eval', 'ok', n=len(miss))
        if 'ignore' in line_recs:
            dk0 = line_recs['ignore'][0]; dl = diag_for(row, 'ignore'); dkeys = {(a, rt): K.diag_key(dk0, a, rt, fps['diag']) for a, rt in dl}
            dmiss = [(a, rt) for (a, rt), v in dkeys.items() if not K.complete(root / 'diag' / f'{v}.json', v)]
            if dmiss:
                if kO is None:
                    _, kO, conf = EV.eval_lines(row, inp, {'ignore': line_recs['ignore'][1]}, ctx)
                if truth:
                    kO = _shuffle_ids(kO, truth)
                rd, rerun = DG.line_diag(row, inp, img, line_recs['ignore'][1], kO, conf, ctx, F_cache.get('F'))
                if rerun:
                    done['detection'] += 1; led('detection', 'ok', rule='line_front re-run (lines_v1 diagnostic; method decode was a stored record)')
                for a, rt in dmiss:
                    rowm, arrays = rd[a]; v = dkeys[(a, rt)]; gp = root / 'diag' / f'{v}.json'; h = K.save_npz(gp.with_suffix('.npz'), acc=arrays['acc'])
                    K.save_json(gp, dict(identity=v, kind='own', decode_key=dk0, image_key=ik, rule='ignore', arm=a, reference=rt, row=rowm, npz_sha256=h, complete=True,
                                         truth=('replaced (drill)' if truth else 'diagnostic truth'), t_local=K.now_local()))
                done['diag_recon'] += len(dmiss); led('diag_recon', 'ok', n=len(dmiss))
    done['seconds'] = round(time.monotonic() - t0, 2)
    return done


def _truth_inputs(inp: dict, truth):
    if not truth:
        return inp
    cls = np.asarray(inp['classes']).copy() if inp.get('classes') is not None else None
    return dict(inp, classes=(1 - cls) if cls is not None else None)


def _truth_ref(ref_img, truth):
    if not truth or ref_img is None:
        return ref_img
    return np.ascontiguousarray(np.asarray(ref_img)[::-1, ::-1])


def _shuffle_ids(kO, truth):
    rng = np.random.default_rng(int(truth['shuffle_seed'])); k = np.asarray(kO).copy(); u = np.unique(k[k >= 0]) if (k >= 0).any() else np.zeros(0, np.int64)
    if len(u):
        mp = dict(zip(u, rng.permutation(u))); k = np.array([mp.get(int(x), x) for x in k], np.int64)
    return k


class Lock:
    def __init__(self, p):
        self.p = p

    def __enter__(self):
        K.guard(self.p).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(self.p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise K.Stop('lock', f'another run holds {self.p}')
        os.write(self.fd, str(os.getpid()).encode()); return self

    def __exit__(self, *exc):
        os.close(self.fd); os.remove(self.p)


def _worker(args):
    row, root, backend, fps, image_map, permit, truth = args
    import c4_deps as DP
    led = Path(root) / 'run' / f'ledger_w{os.getpid()}.jsonl'
    try:
        d = run_unit(row, Path(root), backend, fps, led, image_map, permit, (truth or {}).get(row['image_key']))
    except Exception as e:


        raise RuntimeError(f'{type(e).__name__}: {e}') from None
    d['loaded'] = DP.loaded_project_files()
    return d


def verify_frozen(frozen: dict, skip=()) -> None:
    bad = K.verify_closure(frozen['closure'])
    if bad:
        raise K.Stop('freeze', f'freeze closure mismatch: {bad[:5]}')
    for layer in ('decode', 'eval', 'diag'):
        if layer in skip:
            continue
        if K.fingerprint(layer, deps=frozen['deps']) != frozen['fingerprints'][layer]:
            raise K.Stop('freeze', f'{layer} fingerprint differs from the frozen one')


# Schedule encoding cases after checking the manifest and input dependencies.
def run(manifest: pd.DataFrame, root: Path, frozen: dict, backend: str, workers: int = 8, image_map: dict | None = None, permit=None, truth=None,
        max_units=None, fp_override: dict | None = None) -> dict:

    root = Path(root); K.guard(root)
    if fp_override and (backend != 'replay' or K.SANDBOX.resolve() not in root.resolve().parents):
        raise K.Stop('check', 'fingerprint override only for replay drills in the sandbox')
    with Lock(root / 'run' / 'run.lock'):
        verify_frozen(frozen, skip=set(fp_override or {}))
        rows = manifest.to_dict('records')
        for r in rows:
            r['seed'] = K.norm_seed(r.get('seed'))
        if backend == 'real':
            for r in rows:
                row_contract(r)
            K.require_permit(permit)
        rows = sorted(rows, key=lambda r: (r['sigma'] > 0, r['image_key']))
        if max_units is not None:
            rows = rows[:max_units]
        fps = dict(frozen['fingerprints'], **(fp_override or {})); out = []; fails = []; cancelled = []
        for wave in ([r for r in rows if r['sigma'] == 0], [r for r in rows if r['sigma'] > 0]):
            if not wave or fails:
                cancelled += [r['image_key'] for r in wave] if fails else []
                continue
            it = iter(wave); inflight = {}
            with ProcessPoolExecutor(max_workers=workers) as ex:
                def submit_next():
                    r = next(it, None)
                    if r is not None:
                        inflight[ex.submit(_worker, (r, str(root), backend, fps, image_map, permit, truth))] = r['image_key']
                    return r is not None
                while len(inflight) < 2 * workers and submit_next():
                    pass
                while inflight:
                    dn, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                    for f in dn:
                        key = inflight.pop(f)
                        try:
                            out.append(f.result())
                        except Exception as e:
                            fails.append(dict(key=key, error=traceback.format_exception_only(type(e), e)[-1][:300]))
                    if fails:
                        for f in list(inflight):
                            if f.cancel():
                                cancelled.append(inflight.pop(f))
                        continue
                    while len(inflight) < 2 * workers and submit_next():
                        pass
                cancelled += [r['image_key'] for r in it] if fails else []
        done_keys = {d['key'] for d in out}
        K.save_json(root / 'run' / 'state.json', dict(n_rows=len(rows), done=len(out), failed=fails, not_started=[r['image_key'] for r in rows if r['image_key'] not in done_keys
                                                                                                                  and r['image_key'] not in {f['key'] for f in fails}],
                                                       cancelled=len(cancelled), t_local=K.now_local()))
        if fails:
            raise K.Stop('technical', f'{len(fails)} units failed; submission stopped, {len(cancelled)} queued / unstarted units not run: {fails[:2]}')
        verify_frozen(frozen, skip=set(fp_override or {}))
        import c4_deps as DP
        loaded = sorted(set().union(*[set(d.pop('loaded', [])) for d in out])) if out else []
        summ = dict(n_rows=len(rows), done=len(out), failures=fails, method_exec=sum(d['method_exec'] for d in out), detection=sum(d['detection'] for d in out),
                    eval=sum(d['eval'] for d in out), diag_recon=sum(d['diag_recon'] for d in out), rendered=sum(d['image'] == 'rendered' for d in out),
                    deps_check=DP.check(loaded), seconds=round(sum(d['seconds'] for d in out), 1), t_local=K.now_local())
        if fails:
            raise K.Stop('technical', f'{len(fails)} units failed: {fails[:2]}')
        if not summ['deps_check']['ok']:
            raise K.Stop('check', f"loaded files outside the frozen closure: {summ['deps_check']['missing_from_closure'][:5]}")
        return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--manifest'); ap.add_argument('--root'); ap.add_argument('--frozen'); ap.add_argument('--images')
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--max-units', type=int)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'formal':
            for f in ('frozen.json', 'manifest.csv'):
                if not (K.FORMAL / f).exists():
                    raise K.Stop('freeze', f'formal/{f} missing: no fallback to an older entry or default parameters')
            fz = K.load_json(K.FORMAL / 'frozen.json'); man = pd.read_csv(K.FORMAL / 'manifest.csv', low_memory=False)
            if K.sha_file(K.FORMAL / 'manifest.csv') != fz.get('manifest_sha256'):
                raise K.Stop('freeze', 'formal manifest differs from the frozen one')
            pm = K.require_permit(K.PERMIT)
            if pm['manifest_sha256'] != fz['manifest_sha256'] or pm['closure_sha256'] != K.digest(fz['closure'], 64):
                raise K.Stop('permit', 'permit was issued for another manifest or package closure')
            print(run(man, K.FORMAL, fz, 'real', a.workers, permit=K.PERMIT, max_units=a.max_units)); return 0
        if a.cmd == 'replay':
            if not (a.frozen and a.manifest and a.images and a.root):
                raise K.Stop('technical', 'replay needs --frozen --manifest --images --root (no defaults)')
            fz = K.load_json(a.frozen); man = pd.read_csv(a.manifest, low_memory=False); im = pd.read_csv(a.images)
            imap = {r.image_key: dict(npz=r.npz, sha256=r.sha256) for r in im.itertuples()}
            print(run(man, Path(a.root), fz, 'replay', a.workers, image_map=imap, max_units=a.max_units)); return 0
        raise K.Stop('technical', f'unknown command {a.cmd}')
    except K.Stop as e:
        print(e, file=sys.stderr); return K.EXIT.get(e.code, 2)


if __name__ == '__main__':
    sys.exit(main())
