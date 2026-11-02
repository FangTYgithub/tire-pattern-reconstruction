# Define encoding analysis rules and result-column mappings.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import c6_common as K

REQUIRED = ('name', 'evidence', 'evidence_kind', 'record_source', 'key_scheme', 'data_root', 'inputs', 'denominators', 'derived', 'profile', 'structure', 'output')
KINDS = dict(evidence_kind=('seen', 'test', 'formal', 'entry_check'), record_source=('c34_files', 's12_c1', 'synthetic'), key_scheme=('c3', 'c4', 's12'),
             denominators=('supplementary+npz', 'npz', 'row', 'derived'), derived=('compute', 'row', 'none'), profile=('c34', 's12'))


def _abs(p) -> Path:
    return Path(p) if Path(p).is_absolute() else K.ROOT / p


def load(path) -> dict:
    p = _abs(path)
    if not p.exists():
        raise K.Stop('config', f'configuration {p} does not exist (no default batch)')
    cfg = json.loads(p.read_bytes().decode('utf-8'))
    miss = [k for k in REQUIRED if k not in cfg]
    if miss:
        raise K.Stop('config', f'configuration lacks {miss}')
    for k, allowed in KINDS.items():
        if cfg[k] not in allowed:
            raise K.Stop('config', f'{k} = {cfg[k]!r} not in {allowed}')
    out = _abs(cfg['output']).resolve()
    if cfg['evidence_kind'] == 'test' and K.SANDBOX.resolve() not in out.parents:
        raise K.Stop('config', 'test batches write only below the S3-CODE-06 sandbox')
    if cfg['evidence_kind'] != 'test' and K.SANDBOX.resolve() in out.parents:
        raise K.Stop('config', 'research / entry-check batches never write into the sandbox')
    if cfg['record_source'] == 'synthetic' and cfg['evidence_kind'] != 'test':
        raise K.Stop('config', 'synthetic records are test material only')
    if cfg['derived'] == 'compute' and cfg['record_source'] != 'c34_files':
        raise K.Stop('config', 'derived evaluation needs stored decode records')
    cfg['_path'] = K.rel(p); cfg['_sha256'] = K.sha_file(p)
    return cfg


def verify_inputs(cfg: dict, H: K.Hashes) -> dict:
    out, bad = {}, []
    for name, spec in cfg['inputs'].items():
        if spec is None:
            continue
        p = _abs(spec['file']); h = H.sha(p)
        if h is None:
            bad.append(dict(input=name, file=spec['file'], problem='missing')); continue
        if h != spec['sha256']:
            bad.append(dict(input=name, file=spec['file'], problem='sha256 differs', registered=spec['sha256'][:16], actual=h[:16])); continue
        out[name] = dict(file=spec['file'], sha256=h)
    if bad:
        raise K.Stop('input', f'registered inputs do not verify: {bad}')
    return out


def manifest(cfg: dict) -> pd.DataFrame:
    M = pd.read_csv(_abs(cfg['inputs']['manifest']['file']), low_memory=False)
    if 'layout' in M:
        M['layout'] = M.layout.astype(str)
    if cfg.get('manifest_map'):
        M = M.rename(columns={v: k for k, v in cfg['manifest_map'].items()})
    for k, v in (cfg.get('manifest_constants') or {}).items():
        M[k] = v
    return M


def fingerprints(cfg: dict) -> dict:
    src = cfg.get('fingerprints_from', 'freeze')
    if src == 'config':
        return dict(cfg['fingerprints'])
    if src == 'freeze':
        return dict(K.load_json(_abs(cfg['inputs']['freeze']['file']))['fingerprints'])
    if src == 'none':
        return {}
    raise K.Stop('config', f'fingerprints_from {src!r}')


def freeze_status(cfg: dict) -> dict:
    mode = cfg.get('freeze_closure', 'verify'); spec = cfg['inputs'].get('freeze')
    if not spec or mode == 'none':
        return dict(mode='none')
    fz = K.load_json(_abs(spec['file'])); bad = K.K4.verify_closure(fz['closure']) if 'closure' in fz else ['no closure']
    if mode == 'verify' and bad:
        raise K.Stop('freeze', f'freeze closure does not verify: {bad[:5]}')
    return dict(mode=mode, mismatches=len(bad), files=bad[:5], note=cfg.get('freeze_note', ''))


def package_status(cfg: dict) -> dict:
    if not cfg.get('package_freeze'):
        return dict(mode='none')
    p = _abs(cfg['package_freeze'])
    if not p.exists():
        raise K.Stop('freeze', f'package freeze {cfg["package_freeze"]} missing')
    fz = K.load_json(p); bad = K.K4.verify_closure(fz['closure'])
    if bad:
        raise K.Stop('freeze', f'analysis code differs from the package freeze: {bad[:5]}')
    return dict(mode='verify', closure_sha256=fz['closure_sha256'], n_files=fz['n_files'])
