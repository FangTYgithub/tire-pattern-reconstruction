# Record source dependencies for encoding execution and analysis.

from __future__ import annotations

import json
import sys

import c4_common as K
import c3_deps as D3

ENTRIES = dict(input=['c4_parity', 'c3_raster', 's3c_formal'], decode=['c4_store', 'c4_parity', 'c3_store', 'c3_raster', 's3c_formal'],
               eval=['c3_eval', 'c4_store', 'c4_parity', 'c3_store', 'c3_raster', 's3c_formal'],
               diag=['c3_diag', 'c3_eval', 'c4_store', 'c4_parity', 'c3_store', 'c3_raster', 's3c_formal'], render=['rdy_platform', 'adp_binding', 'adp_surf02', 'm3_virtual_camera'],
               analysis=['c4_analysis', 'c4_collect', 'c4_figs'])


def capture() -> dict:
    layers = {l: dict(entries=e, files=sorted(D3.closure(e)), n=0) for l, e in ENTRIES.items()}
    for v in layers.values():
        v['n'] = len(v['files'])
    out = dict(method='c3_deps static closure (top-level + function-level imports, module-name index fallback); runtime loaded subset checked by c4_deps.check',
               layers=layers, union=sorted(set().union(*[set(v['files']) for v in layers.values()])), ambiguous_names=D3.AMBIGUOUS,
               note='analysis layer = package only (never in decode / eval / diag fingerprints)', t_local=K.now_local())
    out['n_union'] = len(out['union'])
    K.save_json(K.DEPS_FILE, out)
    return out


def loaded_project_files() -> list:
    return D3.loaded_project_files()


def check(loaded: list, layers=('decode', 'eval', 'diag')) -> dict:
    d = K.load_json(K.DEPS_FILE); allowed = set().union(*[set(d['layers'][l]['files']) for l in layers])
    infra = {K.rel(K.HERE / m) for m in K.PACKAGE_MODULES} | {K.rel(K.S3C3 / m) for m in ('c3_common.py', 'c3_deps.py')}
    miss = sorted(set(loaded) - allowed - infra)
    return dict(n_loaded=len(loaded), missing_from_closure=miss, ok=not miss)


if __name__ == '__main__':
    if sys.argv[1] == 'capture':
        r = capture(); print(json.dumps({k: (v if k != 'layers' else {l: x['n'] for l, x in v.items()}) for k, v in r.items() if k != 'union'}, indent=1, ensure_ascii=False)[:2000])
