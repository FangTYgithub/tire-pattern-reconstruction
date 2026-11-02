# Dispatch held-out layout generation, execution and analysis commands.

"""S4-REAL-07 entry.

usage: python s4r7_main.py check | a1 | gen | bind | freeze | refreeze "<reason>" | a2 | a3 | drill | subset | rest | analyse | selfcheck | figs | display | assemble | final
A failing stage exits non-zero (never overwritten in a finally block). Imaging only through the registered M3 entry for the frozen new conditions.
"""
from __future__ import annotations

import hashlib
import json
import sys

import s4r7_common as H

K = H.K


def check() -> dict:
    r6 = json.loads(H.reg(H.R6 / 'run.json').read_text(encoding='utf-8'))
    f = r6['final']
    bad_out = [k for k, v in f['outputs'].items() if K.sha_file(K.ROOT / k) != v]
    bad_code = [k for k, v in f['final_code'].items() if K.sha_file(K.ROOT / k) != v]
    bad_in = [k for k, v in r6['inputs_read'].items() if K.sha_file(K.ROOT / k) != v]
    for p in (H.TASK, H.REVIEW, H.R6 / 'report_zh.md', H.R6 / 'manifest.csv', H.R6 / 'results.csv'):
        H.reg(p)
    out = dict(ok=not (bad_out or bad_code or bad_in), s4_real06_outputs=len(f['outputs']), s4_real06_code=len(f['final_code']), s4_real06_inputs=len(r6['inputs_read']),
               changed=bad_out + bad_code + bad_in[:10], t_local=K.now_local())
    H.run_put('check', out); H.flush_reads('check')
    if not out['ok']:
        raise H.Stop('anchor', f'S4-REAL-06 material changed after its final check: {out["changed"][:6]}')
    return out


def main(cmd: str):
    if cmd == 'check':
        return check()
    if cmd == 'a1':
        import s4r7_fix as A
        return A.a1()
    if cmd == 'gen':
        import s4r7_gen as G
        return G.generate()
    if cmd in ('bind', 'freeze'):
        import s4r7_bind as B
        return getattr(B, cmd)()
    if cmd == 'refreeze':
        import s4r7_bind as B
        if len(sys.argv) < 3 or not sys.argv[2].strip():
            raise H.Stop('freeze', 'refreeze needs a reason')
        return B.refreeze(sys.argv[2])
    if cmd in ('a2', 'a3', 'drill', 'subset', 'rest'):
        import s4r7_run as R
        return getattr(R, cmd)()
    if cmd in ('analyse', 'selfcheck'):
        import s4r7_analysis as N
        return getattr(N, cmd)()
    if cmd in ('figs', 'display'):
        import s4r7_figs as F
        return F.make_all() if cmd == 'figs' else F.register_display()
    if cmd in ('assemble', 'final'):
        import s4r7_pack as P
        return getattr(P, cmd)()
    raise SystemExit(__doc__)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    try:
        res = main(cmd)
    except H.Stop as e:
        print(f'STOP {e}'); sys.exit(1)
    print(json.dumps(K.clean(res), ensure_ascii=False)[:6000])
