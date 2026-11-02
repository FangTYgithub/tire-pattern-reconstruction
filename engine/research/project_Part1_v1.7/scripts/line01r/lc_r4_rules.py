# Evaluate width-code acceptance rules and identity-risk counts.

GATE = 0.001
EPS_GAIN = 1e-12


def rate(num, den):
    return (float(num) / float(den)) if den else None


def positive_gain(d):
    return bool(d is not None and d == d and float(d) > EPS_GAIN)


def all_or_none(flags):
    flags = [f for f in flags]
    if not flags:
        return None
    return bool(all(flags))


def identity_status(acc_conf, acc_wrong, acc_unconf, new_conf, new_wrong, new_unconf, n_acc, ignore_acc):
    if n_acc == 0:
        return 'no_output'
    if acc_unconf > 0 or new_unconf > 0:
        return 'not_accepted_unconfirmable'
    r_all = rate(acc_wrong, acc_conf)
    if r_all is None or r_all > GATE:
        return 'unsafe'
    n_new = new_conf + new_unconf
    if n_new == 0:
        return 'safe_NA_new' if ignore_acc > 0 else 'no_output'
    r_new = rate(new_wrong, new_conf)
    if r_new is None or r_new > GATE:
        return 'unsafe'
    return 'safe'


def upper_bound_all_unknown_wrong(acc_conf, acc_wrong, acc_unconf):
    return rate(acc_wrong + acc_unconf, acc_conf + acc_unconf)
