# Dispatch joint-study manifest checks and case execution.

"""S5-ALIGN-03 entry (zero imaging / decoding / interpolation in this round). Writes only outputs/surface_revision/s5_align03/.

  python -B s5r_main.py gate-manifest [--manifest CSV] [--mode full|subset --keys JSON]   manifest gate A (exit 0 pass, 3 refused)
  python -B s5r_main.py compat                                                          post-hoc compatibility registration (C2)
  python -B s5r_main.py analyze [--mode full|subset --keys JSON]                        gates -> binding -> replay (B, D)
  python -B s5r_main.py drills                                                          A3 / B2 / C3 drills (isolated directory, cleaned)
  python -B s5r_main.py next8                                                           selection of the 8 technical images (E)
  python -B s5r_main.py status --keys JSON --store DIR                                  stage decisions only (read only)
  python -B s5r_main.py run --keys JSON --store DIR [--permit JSON]                     formal runner; without a permit every computation is refused
  python -B s5r_main.py fig2 | final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import s5r_common as K

ISO = K.OUT / '_isolation'


def _keys(path):
    return [tuple(x) for x in json.loads(Path(path).read_text(encoding='utf-8'))] if path else None


def cmd_gate_manifest(M: pd.DataFrame, mode='full', keys=None, reg=None) -> dict:
    import s5r_gates as G
    v = G.gate_manifest(M, mode, keys, reg if reg is not None else G.registered())
    if not v['ok']:
        raise K.Stop('refused', 'manifest refused', detail=v)
    return v


def manifest_rows(keys: list, M: pd.DataFrame | None = None) -> list:

    import s5r_gates as G
    M = K.read_csv(K.reg(K.MANIFEST)) if M is None else M
    P = K.read_csv(K.reg(K.S345 / 'plan_s345.csv')); P = P[(P.set_formal == 'S5-main') & (P.sigma == 0.0)].set_index(['pair_role', 'surface', 'phase_label'])
    out = []
    for k in keys:
        x = M[(M.combo == k[0]) & (M.surface == k[1]) & (M.phase_label == k[2]) & (M.sigma == float(k[3])) & (M.rep == int(k[4]))]
        if len(x) != 1:
            raise K.Stop('scope', f'key {k} selects {len(x)} manifest rows (exactly 1 required)')
        r = x.iloc[0].to_dict(); f = G.image_identity_fields(r)
        out.append(dict(key=r['manifest_key'], manifest_key=r['manifest_key'], f=f, family=r['family'], surface=r['surface'], sigma=float(r['sigma']), seed=f['seed'],
                        radius_mm=(None if r['family'] == 'line' else float(r['radius_mm'])), status=r['status'], cand_id_sigma0=P.loc[(k[0], k[1], k[2]), 'cand_id']))
    return out


def methods_of(row) -> list:
    return ['D0', 'DF'] if row['family'] == 'point' else ['P4']


def cmd_run(store_dir, keys, permit_path=None, executor=None, rows=None) -> dict:
    import s5r_store as S
    store_dir = Path(store_dir)
    if executor == 'stub':
        if ISO.resolve() not in store_dir.resolve().parents:
            raise K.Stop('guard', 'the stub executor is allowed only below the isolation directory')
        ex = S.Stub()
    elif executor is not None and not isinstance(executor, str):
        ex = executor
    elif permit_path:
        ex = S.Real(json.loads(Path(permit_path).read_text(encoding='utf-8')))
    else:
        ex = S.Forbidden()
    rows = rows if rows is not None else manifest_rows(keys)
    st = S.Store(store_dir)
    log = S.run_rows(st, rows, ex, S.platform_refs(), methods_of)
    return dict(log={k: v for k, v in log.items() if k != 'stages'}, stages=log['stages'], executor_calls=list(getattr(ex, 'calls', [])))


def cmd_status(store_dir, keys) -> dict:
    import s5r_store as S
    st = S.Store(Path(store_dir)); pr = S.platform_refs(); out = []
    for r in manifest_rows(keys):
        iid = S.image_identity(r['f'], pr); s, why = st.status('image', r['key'], iid)
        out.append(dict(key=r['key'], image_identity=iid, image=s, why=why, decodes=('decided after the image exists' if s != 'reuse' else 'see store')))
    return dict(store=K.rel(store_dir), rows=out)


def entry(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('cmd')
    ap.add_argument('--manifest'); ap.add_argument('--mode', default='full'); ap.add_argument('--keys'); ap.add_argument('--store'); ap.add_argument('--permit'); ap.add_argument('--executor')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'gate-manifest':
            M = K.read_csv(a.manifest or K.MANIFEST)
            r = cmd_gate_manifest(M, a.mode, _keys(a.keys))
            r = {k: v for k, v in r.items() if k != 'reasons'}
        elif a.cmd == 'compat':
            import s5r_bind as B
            r = B.compat(); r = dict(n_records=r['n_records'], n_df_images=r['n_df_images'], dependencies_ok=r['dependencies']['ok'], mismatches=r['mismatches'])
        elif a.cmd == 'analyze':
            import s5r_bind as B
            r = B.analyze(a.mode, _keys(a.keys))
            r = {k: v for k, v in r.items() if k not in ('errata',)}
        elif a.cmd == 'drills':
            import s5r_drills as D
            r = D.run_all()
            if not r['all_ok']:
                raise K.Stop('refused', 'at least one drill did not behave as required', detail=r['summary'])
        elif a.cmd == 'next8':
            import s5r_drills as D
            r = D.next8()
        elif a.cmd == 'status':
            r = cmd_status(a.store, _keys(a.keys))
        elif a.cmd == 'run':
            r = cmd_run(a.store, _keys(a.keys), a.permit, a.executor)
        elif a.cmd == 'fig2':
            import s5r_drills as D
            r = D.fig2()
        elif a.cmd == 'final':
            import s5r_drills as D
            r = D.final()
        else:
            print(__doc__); return K.EXIT['refused']
    except K.Stop as e:
        print(json.dumps(dict(code=e.code, msg=e.msg, detail=K.clean(e.detail)), ensure_ascii=False, default=str)[:3000], file=sys.stderr)
        return K.EXIT.get(e.code, 2)
    except Exception as e:
        import traceback
        print(json.dumps(dict(code='technical', msg=repr(e)[:500], trace=traceback.format_exc()[-1500:]), ensure_ascii=False), file=sys.stderr)
        return K.EXIT['technical']
    print(json.dumps(K.clean(r), ensure_ascii=False, default=str)[:6000])
    return 0


if __name__ == '__main__':
    code = entry()
    if code == 0:
        rr = K.run_load(); sl = rr.get('sentinel', [])
        sl.append(dict(cmd=sys.argv[1] if len(sys.argv) > 1 else '', heavy_modules_loaded=K.heavy_loaded(), t=K.now()))
        if K.RUN.exists() or sys.argv[1:2] in (['compat'], ['analyze'], ['drills'], ['next8'], ['final'], ['fig2']):
            K.run_put('sentinel', sl[-40:])
    sys.exit(code)
