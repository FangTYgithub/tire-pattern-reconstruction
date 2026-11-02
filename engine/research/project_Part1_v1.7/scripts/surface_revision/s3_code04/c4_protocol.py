# Check registered encoding rules and construct experiment records.

from __future__ import annotations

import c4_common as K


def add(section: str, obj: dict) -> None:
    pr = K.load_json(K.PROTOCOL) if K.PROTOCOL.exists() else dict(task=K.TASK_ID, task_file=K.TASK_FILE.name, task_sha256=K.sha_file(K.TASK_FILE),
                                                                  review_read=K.REVIEW_FILE.name, review_sha256=K.sha_file(K.REVIEW_FILE), created=K.now_local())
    if section in pr:
        raise K.Stop('check', f'protocol section {section} exists; never overwritten')
    pr[section] = dict(obj, t_local=K.now_local()); K.save_json(K.PROTOCOL, pr)


def ref(path) -> dict:
    return dict(file=K.rel(path), sha256=K.sha_file(path))


def checks_add(name: str, obj: dict) -> None:

    c = K.load_json(K.CHECKS_FILE) if K.CHECKS_FILE.exists() else dict(task=K.TASK_ID, created=K.now_local())
    if name in c:
        raise K.Stop('check', f'checks entry {name} exists; never overwritten')
    c[name] = dict(obj, t_local=K.now_local()); K.save_json(K.CHECKS_FILE, c)
