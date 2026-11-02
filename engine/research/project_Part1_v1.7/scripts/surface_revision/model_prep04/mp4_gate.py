# Check environment case completeness, pair identities and metric consistency.

from __future__ import annotations

import json

import mp4_common as K

GATE_VERSION = 'mp4_gate.final_status v1 (strict ledger)'
END_STATES = ('ok', 'failed', 'interrupted', 'aborted')
COUNT_FIELDS = ('reserved', 'ok', 'failed', 'interrupted', 'aborted', 'open')
FAIL_FIELDS = ('failed', 'interrupted', 'aborted')
OK_STATUS = dict(prep='PREPARATION_COMPLETE_AWAITING_REVIEW', run='RUN_COMPLETE_AWAITING_REVIEW')


def ledger_rows(path) -> tuple[list, list]:

    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return [], [f'ledger missing: {p.name}']
    rows, probs = [], []
    for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            probs.append(f'ledger line {i}: not JSON')
            continue
        if not isinstance(r, dict):
            probs.append(f'ledger line {i}: not an object')
            continue
        rows.append(r)
    return rows, probs


def ledger_audit(rows: list, caps: dict, expected: dict | None = None) -> dict:

    probs, res, ends = [], {}, {}
    for i, r in enumerate(rows, 1):
        ev = r.get('event')
        if ev == 'reserve':
            rid, grp = r.get('id'), r.get('groups')
            if not isinstance(rid, str) or not rid or not isinstance(grp, list) or not grp or 'key' not in r:
                probs.append(f'record {i}: reserve without id / groups / key')
                continue
            if rid in res:
                probs.append(f'record {i}: duplicate reservation id')
                continue
            unknown = [g for g in grp if g not in caps]
            if unknown:
                probs.append(f'record {i}: unregistered groups {unknown}')
            res[rid] = grp
        elif ev == 'end':
            rid, st = r.get('id'), r.get('status')
            if not isinstance(rid, str) or st not in END_STATES:
                probs.append(f'record {i}: end without id or with illegal status {st!r}')
                continue
            if rid in ends:
                probs.append(f'record {i}: second end for one reservation')
                continue
            ends[rid] = st
        elif ev == 'note':
            if 'kind' not in r:
                probs.append(f'record {i}: note without kind')
        else:
            probs.append(f'record {i}: unknown event {ev!r}')
    orphan = sorted(set(ends) - set(res))
    if orphan:
        probs.append(f'{len(orphan)} end records without a reservation')
    counts = {g: dict.fromkeys(COUNT_FIELDS, 0) for g in caps}
    for rid, grp in res.items():
        st = ends.get(rid, 'open')
        for g in grp:
            d = counts.setdefault(g, dict.fromkeys(COUNT_FIELDS, 0))
            d['reserved'] += 1; d[st] += 1
    unclosed = sorted(rid for rid in res if rid not in ends)
    fails = {g: {f: d[f] for f in FAIL_FIELDS if d[f]} for g, d in counts.items() if any(d[f] for f in FAIL_FIELDS)}
    over = {g: d['reserved'] for g, d in counts.items() if g in caps and d['reserved'] > int(caps[g])}
    if over:
        probs.append(f'caps exceeded {over}')
    short = {}
    for g, n in (expected or {}).items():
        if counts.get(g, {}).get('ok', 0) != int(n) or counts.get(g, {}).get('reserved', 0) != int(n):
            short[g] = dict(expected=int(n), reserved=counts.get(g, {}).get('reserved', 0), ok=counts.get(g, {}).get('ok', 0))
    return dict(counts=counts, problems=probs, unclosed=unclosed, failures=fails, over_caps=over, incomplete_groups=short, n_records=len(rows))


def counts_audit(counts, caps: dict) -> list:

    if not isinstance(counts, dict) or not counts:
        return ['counts missing']
    probs = []
    for g in caps:
        if g not in counts:
            probs.append(f'group {g} missing')
    for g, d in counts.items():
        if not isinstance(d, dict) or any(f not in d for f in COUNT_FIELDS):
            probs.append(f'group {g}: missing fields')
            continue
        vals = [d[f] for f in COUNT_FIELDS]
        if any((not isinstance(v, int)) or isinstance(v, bool) or v < 0 for v in vals):
            probs.append(f'group {g}: counts must be non-negative integers')
            continue
        if d['reserved'] != sum(d[f] for f in COUNT_FIELDS if f != 'reserved'):
            probs.append(f'group {g}: reserved != ok + failed + interrupted + aborted + open')
        if any(d[f] for f in FAIL_FIELDS) or d['open']:
            probs.append(f'group {g}: unexplained failed / interrupted / aborted / open {[(f, d[f]) for f in FAIL_FIELDS + ("open",) if d[f]]}')
    return probs


def binding(parts: dict, layers: tuple) -> str:
    return K.digest({L: parts.get(L) for L in layers})


def check_proof(proof, required: tuple, current_binding: str) -> tuple[bool, list]:
    why = []
    if not isinstance(proof, dict):
        return False, ['no proof']
    rows = proof.get('rows')
    if not isinstance(rows, list) or not rows:
        return False, ['empty or missing check list']
    ids = [r.get('check_id') for r in rows if isinstance(r, dict)]
    if len(ids) != len(rows):
        why.append('malformed check rows')
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        why.append(f'duplicate check ids {dup}')
    miss, extra = sorted(set(required) - set(ids)), sorted(set(ids) - set(required))
    if miss:
        why.append(f'missing required checks {miss}')
    if extra:
        why.append(f'unregistered check ids {extra}')
    failed = [r.get('check_id') for r in rows if isinstance(r, dict) and r.get('passed') is not True]
    if failed:
        why.append(f'failed checks {failed}')
    if proof.get('binding') != current_binding or any(isinstance(r, dict) and r.get('binding') != current_binding for r in rows):
        why.append('proof bound to another version (dependency / configuration / data changed)')
    return (not why), why


def make_proof(rows: list, current_binding: str) -> dict:
    return dict(binding=current_binding, rows=[dict(r, binding=current_binding) for r in rows], t=K.now())


# Combine completion counts and integrity checks into the run status.
def final_status(state: dict, mode: str) -> dict:

    if mode not in OK_STATUS:
        raise K.Stop('usage', f'unknown gate mode {mode!r}')
    hard, review, partial = [], [], []
    if state.get('freeze_problems'):
        hard.append(f"freeze: {state['freeze_problems'][:3]}")
    proofs = state.get('proofs')
    if not isinstance(proofs, dict) or not proofs:
        hard.append('stage proofs missing')
    else:
        for st, v in proofs.items():
            ok, why = v if isinstance(v, (tuple, list)) and len(v) == 2 else (False, ['malformed proof state'])
            if ok is not True:
                hard.append(f'proof {st}: {list(why)[:3]}')
    led = state.get('ledgers')
    if not isinstance(led, dict) or not led:
        hard.append('ledger state missing')
    else:
        for name, a in led.items():
            if not isinstance(a, dict) or 'problems' not in a or 'counts' not in a:
                hard.append(f'ledger {name}: audit missing')
                continue
            if a['problems']:
                hard.append(f"ledger {name}: {a['problems'][:3]}")
            if a.get('unclosed'):
                hard.append(f"ledger {name}: {len(a['unclosed'])} unclosed reservations")
            if a.get('failures'):
                review.append(f"ledger {name}: unexplained failures {a['failures']} (no recovery-record contract; not cleared)")
            if a.get('incomplete_groups'):
                partial.append(f"ledger {name}: incomplete {a['incomplete_groups']}")
    for name, probs in (state.get('counts') or {}).items():
        if probs:
            (review if any('unexplained' in p for p in probs) else hard).append(f'counts {name}: {probs[:3]}')
    calls, exp = state.get('calls'), state.get('calls_expected')
    if not isinstance(calls, dict) or set(calls) != {'render', 'decode', 'evaluation'} or not all(isinstance(v, int) and v >= 0 for v in calls.values()):
        hard.append('scientific call record missing or malformed')
    elif exp is not None and calls != exp:
        hard.append(f'scientific calls {calls} != expected {exp}')
    if state.get('protected_changed'):
        hard.append(f"protected inputs changed: {state['protected_changed'][:3]}")
    comp = state.get('completeness')
    if comp is not None and not comp.get('ok'):
        partial.append(f"completeness: {comp.get('problems', ['missing'])[:3]}")
    if hard:
        st, ex = 'BLOCKED', K.EXIT['technical']
    elif review:
        st, ex = 'NEEDS_REVIEW', K.EXIT['technical']
    elif partial:
        st, ex = 'PARTIAL', K.EXIT['result_integrity']
    else:
        st, ex = OK_STATUS[mode], 0
    return dict(status=st, exit=ex, reasons=hard + review + partial, gate=GATE_VERSION)
