# Bind study adapters to the surface, imaging, decoding and evaluation interfaces.

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import adp_common as C


CAPABILITIES = {
    'version':          'str or callable() -> str : platform version string (compared with platform.expected_version)',
    'release':          'callable() -> dict : must contain released=True and config_sha256 (the frozen SURF-02 configuration)',
    'surface_record':   'callable(surface_id) -> dict : c, grid / placement / evaluation-domain versions and hashes (enters the forward key)',
    'sensor_record':    'callable() -> dict : camera / projector pose, intrinsics, working distance (enters the forward key)',
    'search_window':    'callable() -> (z0_mm, z1_mm) : the single validated window shared by dots and lines',
    'render':           'callable(surface_id, pattern_f32, peak, sigma, seed, psf_px) -> dict(image=uint8 HxW, meta=dict) : forward image, fixed placement',
    'decode':           'callable(method, image, context) -> dict(pts=(N,3), accepted=(N,), aux=dict) : existing decode paths with the platform window',
    'evaluate':         'callable(surface_id, family, decoded, context) -> dict : per-case evaluation record (see adp_metrics.REQUIRED_EVAL_FIELDS)',
}
REQUIRED_FOR_DRYRUN = ('version', 'release', 'surface_record', 'sensor_record', 'search_window')
REQUIRED_FOR_RUN = tuple(CAPABILITIES)


def binding_template() -> dict:
    return dict(
        schema='S1S5-ADAPT binding v1',
        status='UNBOUND: SURF-02 interface not delivered (fill every null, then run `adp_run.py bind-check`)',
        platform=dict(module=None, module_dir=None, expected_version=None, release_record=None, release_record_sha256=None),
        capabilities={k: None for k in CAPABILITIES},
        capability_contract=CAPABILITIES,
        surface_id_map={k: None for k in C.SURFACE_IDS},
        methods={},
        expected_conventions=C.EXPECTED_CONVENTIONS,
        notes=['Do not point the binding at old runners (sl_lib.work, l01rb_run.work, rd_lib.work): they rebuild old surfaces and apply Z - Z.min().',
               'Fill methods with every decode arm listed in candidates_s1s5.csv column decode_arm (adp_run.py bind-check lists the missing ones).'],
    )


@dataclass
class Bound:
    module: object
    version: str
    release: dict
    caps: dict = field(default_factory=dict)
    surface_id_map: dict = field(default_factory=dict)
    methods: dict = field(default_factory=dict)
    binding_sha256: str = ''


    def sid(self, adapter_surface: str) -> str:
        if adapter_surface in C.LEGACY_SURFACES or adapter_surface not in C.SURFACES:
            raise C.AdapterStop('legacy_geometry_refused', f'surface {adapter_surface!r} is not one of the four new research surfaces')
        s = self.surface_id_map.get(adapter_surface)
        if not s:
            raise C.AdapterStop('platform_unbound', f'no SURF-02 id bound for {adapter_surface}')
        return s

    def call(self, cap: str, *a, **kw):
        f = self.caps.get(cap)
        if f is None:
            raise C.AdapterStop('capability_missing', f'capability {cap} not bound')
        return f(*a, **kw)

    def window(self) -> tuple[float, float]:
        z0, z1 = (float(v) for v in self.call('search_window'))
        if not z1 > z0:
            raise C.AdapterStop('platform_mismatch', f'invalid search window {(z0, z1)}')
        return z0, z1

    def method(self, arm: str) -> str:
        m = self.methods.get(arm)
        if not m:
            raise C.AdapterStop('capability_missing', f'decode arm {arm} has no bound SURF-02 method')
        return m

    def geometry_identity(self, adapter_surface: str) -> dict:

        return dict(platform_version=self.version, platform_config=self.release.get('config_sha256'), surface=self.sid(adapter_surface),
                    surface_record=C.digest(self.call('surface_record', self.sid(adapter_surface))), sensor_record=C.digest(self.call('sensor_record')),
                    window=list(self.window()))


def load_binding(path: Path | None = None) -> dict:
    p = Path(path or C.BINDING_JSON)
    if not p.exists():
        raise C.AdapterStop('platform_unbound', f'binding file missing: {C.rel(p)} (write the template with `adp_run.py bind-template`)')
    return C.load_json(p)


def binding_problems(b: dict, need=REQUIRED_FOR_RUN) -> list[tuple[str, str]]:

    out = []
    pl = b.get('platform') or {}
    for k in ('module', 'expected_version', 'release_record', 'release_record_sha256'):
        if not pl.get(k):
            out.append(('platform_unbound', f'platform.{k} not set'))
    caps = b.get('capabilities') or {}
    for k in need:
        if not caps.get(k):
            out.append(('capability_missing', f'capability {k} not bound'))
    sm = b.get('surface_id_map') or {}
    for k in C.SURFACE_IDS:
        if not sm.get(k):
            out.append(('platform_unbound', f'surface_id_map.{k} not set'))
        elif str(sm[k]) in C.LEGACY_SURFACES:
            out.append(('legacy_geometry_refused', f'surface_id_map.{k} points to the legacy surface {sm[k]!r}'))
    return out


def bind(path: Path | None = None, need=REQUIRED_FOR_RUN, required_methods=()) -> Bound:

    p = Path(path or C.BINDING_JSON)
    b = load_binding(p)
    probs = binding_problems(b, need)
    if probs:
        code = probs[0][0]
        raise C.AdapterStop(code, '; '.join(m for _, m in probs))
    pl = b['platform']
    rr = C.absp(pl['release_record'])
    if not rr.exists():
        raise C.AdapterStop('platform_unreleased', f'release record missing: {pl["release_record"]}')
    if C.sha_file(rr) != pl['release_record_sha256']:
        raise C.AdapterStop('platform_mismatch', 'release record hash differs from the binding')
    if pl.get('module_dir'):
        d = str(C.absp(pl['module_dir']))
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        mod = importlib.import_module(pl['module'])
    except Exception as e:
        raise C.AdapterStop('platform_unbound', f'cannot import {pl["module"]}: {e!r}') from e
    caps = {}
    for k in need:
        name = b['capabilities'][k]
        if not hasattr(mod, name):
            raise C.AdapterStop('capability_missing', f'{pl["module"]} has no attribute {name} (capability {k})')
        caps[k] = getattr(mod, name)
    v = caps['version']() if callable(caps['version']) else caps['version']
    if str(v) != str(pl['expected_version']):
        raise C.AdapterStop('platform_mismatch', f'platform version {v} differs from the bound {pl["expected_version"]}')
    rel = caps['release']() if callable(caps['release']) else caps['release']
    if not isinstance(rel, dict) or rel.get('released') is not True or not rel.get('config_sha256'):
        raise C.AdapterStop('platform_unreleased', f'platform release flag not set: {str(rel)[:300]}')
    caps['version'] = (lambda vv=v: vv)
    caps['release'] = (lambda rr_=rel: rr_)
    missing = [m for m in required_methods if not (b.get('methods') or {}).get(m)]
    if missing:
        raise C.AdapterStop('capability_missing', f'decode arms without a bound method: {missing}')
    bd = Bound(module=mod, version=str(v), release=rel, caps=caps, surface_id_map=dict(b['surface_id_map']), methods=dict(b.get('methods') or {}),
               binding_sha256=C.sha_file(p))
    z0, z1 = bd.window()
    return bd


def status(path: Path | None = None) -> dict:

    p = Path(path or C.BINDING_JSON)
    if not p.exists():
        return dict(state='UNBOUND', reason='binding.json missing', problems=[])
    try:
        b = load_binding(p)
    except Exception as e:
        return dict(state='UNBOUND', reason=repr(e), problems=[])
    probs = binding_problems(b)
    if probs:
        return dict(state='UNBOUND', reason=f'{len(probs)} binding gaps', problems=[m for _, m in probs])
    try:
        bd = bind(p)
    except C.AdapterStop as e:
        return dict(state=e.code.upper(), reason=str(e), problems=[str(e)])
    return dict(state='BOUND', version=bd.version, config_sha256=bd.release.get('config_sha256'), window=list(bd.window()), problems=[])
