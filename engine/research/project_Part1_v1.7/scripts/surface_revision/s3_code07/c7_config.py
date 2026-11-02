# Define encoding image cases, decoding arms and analysis settings.

from __future__ import annotations

import json

import c6_config as C6
import c7_common as K

REQUIRED = C6.REQUIRED
KINDS = dict(C6.KINDS, evidence_kind=('seen', 'test', 'formal', 'entry_check'))
verify_inputs, manifest, fingerprints, freeze_status, package_status = C6.verify_inputs, C6.manifest, C6.fingerprints, C6.freeze_status, C6.package_status


def _paths(cfg: dict) -> list:
    ps = [cfg['output']] + list((cfg.get('roots') or {}).values())
    rc = cfg.get('runner') or {}
    ps += list((rc.get('roots') or {}).values()) + ([rc['permit']] if rc.get('permit') else [])
    return ps


def load(path) -> dict:
    p = K.abspath(path)
    if not p.exists():
        raise K.Stop('config', f'configuration {p} does not exist (no default batch)')
    cfg = json.loads(p.read_bytes().decode('utf-8'))
    miss = [k for k in REQUIRED if k not in cfg]
    if miss:
        raise K.Stop('config', f'configuration lacks {miss}')
    for k, allowed in KINDS.items():
        if cfg[k] not in allowed:
            raise K.Stop('config', f'{k} = {cfg[k]!r} not in {allowed}')
    for q in _paths(cfg):
        q = K.abspath(q).resolve()
        if K.OUT.resolve() not in q.parents:
            raise K.Stop('config', f'configured write path outside {K.rel(K.OUT)}: {q}')
        in_sb = K.SANDBOX.resolve() in q.parents
        if cfg['evidence_kind'] == 'test' and not in_sb:
            raise K.Stop('config', 'test batches write only below the S3-CODE-07 sandbox')
        if cfg['evidence_kind'] != 'test' and in_sb:
            raise K.Stop('config', 'research batches never write into the sandbox')
    if cfg['record_source'] == 'synthetic' and cfg['evidence_kind'] != 'test':
        raise K.Stop('config', 'synthetic records are test material only')
    if cfg['derived'] == 'compute' and cfg['record_source'] != 'c34_files':
        raise K.Stop('config', 'derived evaluation needs stored decode records')
    cfg['_path'] = K.rel(p); cfg['_sha256'] = K.sha_file(p)
    return cfg
