# Dispatch layout preparation, execution and analysis commands.

"""S4-REAL-06 entry.

usage: python s4r6_main.py check | a1 | a2 | a3 | bind | freeze | a4 | gate_raster | gate_entry | subset | rest | analyse | selfcheck | figs | assemble | final
Imaging only through the registered M3 entry for the 84 planned new pilot images (render budget 90); decode attempts <= 180 by purpose.
A failing stage exits non-zero (the exit status is never overwritten in a finally block).
"""
from __future__ import annotations

import json
import sys

import s4r6_common as H

K = H.K


def check() -> dict:
    a05 = {p: K.sha_file(K.ROOT / p) == h for p, h in H.reference_R05.items()}
    a02 = {p: K.sha_file(K.ROOT / p) == h for p, h in H.reference_R02.items()}
    for p in (H.TASK, H.REVIEW):
        H.reg(p)
    ok = all(a05.values()) and all(a02.values())
    out = dict(ok=bool(ok), s4_real05_at_reference_review=a05, s4_real02_anchors=a02, t_local=K.now_local())
    H.run_put('check', out); H.flush_reads('check')
    if not ok:
        raise H.Stop('anchor', 'anchor check failed')
    return out


def main(cmd: str, args: list) -> object:
    if cmd == 'check':
        return check()
    if cmd in ('a1', 'a2', 'a3'):
        import s4r6_fix as A
        return getattr(A, cmd)()
    if cmd in ('bind', 'freeze'):
        import s4r6_bind as B
        return getattr(B, cmd)()
    if cmd in ('a4', 'gate_raster', 'gate_entry', 'subset', 'rest'):
        import s4r6_run as R
        return getattr(R, cmd)()
    if cmd in ('analyse', 'selfcheck'):
        import s4r6_analysis as N
        return getattr(N, cmd)()
    if cmd == 'figs':
        import s4r6_figs as F
        return F.make_all()
    if cmd in ('assemble', 'final'):
        import s4r6_pack as P
        return getattr(P, cmd)()
    raise SystemExit(__doc__)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    try:
        res = main(cmd, sys.argv[2:])
    except H.Stop as e:
        print(f'STOP {e}'); sys.exit(1)
    print(json.dumps(K.clean(res), ensure_ascii=False)[:6000])
