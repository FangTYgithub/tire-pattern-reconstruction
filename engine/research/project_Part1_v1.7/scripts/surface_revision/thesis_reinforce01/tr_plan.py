# Plan supplementary evaluations and record their source inputs.

from __future__ import annotations

import pandas as pd

import tr_common as K

SURF = ('B_curv', 'M10_curv', 'M24_curv')
S2_LAYOUTS = ('R6x4', 'R8x4', 'T8x4', 'R8x6')
REPLAY_PER_SURFACE = 2


def model_prep_status() -> dict:
    out = dict(task_book=K.rel(K.MODEL_PREP_TASK), task_book_sha256=K.sha(K.MODEL_PREP_TASK), outputs_dir=K.rel(K.MODEL_PREP_OUT), scripts_dir=K.rel(K.MODEL_PREP_SCRIPTS),
               outputs_exist=K.MODEL_PREP_OUT.exists(), scripts_exist=K.MODEL_PREP_SCRIPTS.exists(), checked=K.now())
    files = sorted(p.name for p in K.MODEL_PREP_OUT.glob('*')) if K.MODEL_PREP_OUT.exists() else []
    out['files'] = files
    out['finding'] = ('no usable MODEL-PREP-01 delivery found at check time (this does not mean the task was not started or failed); '
                      'this round uses the old frozen paths only and does not depend on its new interfaces') if not files else 'deliveries present: see files'
    return out


def views() -> pd.DataFrame:
    V = K.read_csv(K.S12 / 'results' / 'views.csv')
    return V[(V.surface_class == 'main') & (V.family == 'point') & (V.regime == 'resource_cap_11695')].copy()


def a3_views() -> pd.DataFrame:
    V = views()
    s1 = V[(V.section == 'S1') & (V.layout == 'S1_R8x4') & V.size_mm.isin([0.5, 1.0, 1.5]) & V.sigma.isin([40.0, 60.0, 80.0])].assign(part='S1')
    s2 = V[(V.section == 'S2') & V.layout.isin(S2_LAYOUTS) & V.sigma.isin([40.0, 60.0, 80.0])].assign(part='S2')
    lo = V[(V.section == 'S1') & (V.layout == 'S1_R8x4') & V.sigma.isin([0.0, 20.0]) & (V.phase_label == 'sn_phase0') & (V.rep == 0)].assign(part='low_noise')
    A = pd.concat([s1, s2, lo], ignore_index=True)
    if A.input_key.duplicated().any():
        raise K.Stop('refused', 'duplicated input keys in the A3 view list')
    want = dict(S1=243, S2=576, low_noise=18)
    got = A.part.value_counts().to_dict()
    if got != want:
        raise K.Stop('refused', f'A3 view counts {got} != registered {want}')
    return A


def c4_df() -> pd.DataFrame:
    C = K.read_csv(K.C4 / 'results' / 'C1_dots_cases.csv')
    return C[(C.rule == 'DF') & (C.state.notna())]


def a3_plan() -> pd.DataFrame:
    A = a3_views(); D = c4_df()
    have = set(D.input_key)
    A['reuse_class'] = ['reuse_DF' if k in have else 'new_DF' for k in A.input_key]
    A = A.sort_values(['surface', 'part', 'layout', 'size_mm', 'sigma', 'phase_label', 'rep']).reset_index(drop=True)
    rp = []
    for s in SURF:
        a = A[(A.surface == s) & (A.reuse_class == 'reuse_DF')].sort_values('input_key').head(REPLAY_PER_SURFACE)
        b = A[(A.surface == s) & (A.reuse_class == 'new_DF')].sort_values('input_key').head(REPLAY_PER_SURFACE)
        rp += [dict(input_key=k, replay='DF_vs_S3-CODE-04') for k in a.input_key] + [dict(input_key=k, replay='D0_vs_stored_S12') for k in b.input_key]
    A['replay'] = A.input_key.map({r['input_key']: r['replay'] for r in rp})
    return A


def input_hashes(A: pd.DataFrame) -> dict:

    files = set()
    for r in A.itertuples():
        files.add(r.image_record.replace('.json', '.npz')); files.add(r.eval_record); files.add(r.eval_record.replace('.json', '.npz'))
        if r.reuse_class == 'reuse_DF':
            files.add(K.rel(K.C4 / 'cache' / 'C1' / f'{r.input_key}.npz')); files.add(K.rel(K.C4 / 'cache' / 'C1' / f'{r.input_key}.json'))
    out = {}
    for f in sorted(files):
        p = K.ROOT / f
        out[f] = K.sha(p) if p.exists() else None
    return out


RULES = dict(
    a3=dict(scope='S1 243 + S2 576 + low-noise 18 views (837 unique stored images, deduplicated by input_key); D0 = stored S12 record; DF = frozen S3-CODE-03 DF via '
                  's3c_dev.dot_decode(..., "DF", "binding") and s3c_dev.evaluate (same evaluator, domains Omega0 30947 nodes / local r <= 15 mm); no DS, no D0 re-decode',
            replay='per surface first 2 reuse_DF inputs (DF replay vs S3-CODE-04 arrays) and first 2 new_DF inputs (D0 B0 replay vs stored record), sorted input_key; <= 12',
            statistics='paired by input_key (same image); repetitions averaged inside a phase, then phase-equal mean; sigma levels share the base noise field and are never counted '
                       'as independent repetitions; identity = evaluator geometric identity (c / w / u over accepted points), source identity only where S3-CODE-04 holds it',
            cited='S3-CODE-04 sigma 100 / 20 / 0 rows of the same configurations are cited (not recomputed) and kept in the discussion'),
    a4=dict(scope='line rasters of sections 1 / 2 (S12 registered line views, all phases) and section 3 (CODE-07 formal plain / coded line rasters); one registration per '
                  'raster x position; affected = features that are lit in the saved raster but belong to no registered line (not "u > 0")',
            rule='evaluation-side identity annotation only (no truth to the decoder, no new accepted points, no change of ids / points / masks / domains)'),
    a2=dict(scope='registered layouts of sections 1 / 2 and the current section-3 plain / coded layouts, B / M10 / M24, all original phases; expected positions '
                  'deduplicated by (surface, projector feature geometry); window [-16, +4] mm checked, never moved; 3 px tolerance as registered'),
    a1=dict(scope='<= 24 stored images / <= 36 image-method combinations, post-hoc mechanism diagnosis (not validation)',
            selection='per mechanism the condition is fixed from the existing summary tables first; inside the condition the case whose global C_tau is closest to the '
                      'condition median (ties: phase, rep, input_key); one explicit counterexample per family kept; three surfaces covered',
            layers='G0 all expected sampling positions + true height; G1 correct-source subset at expected positions + true height; G2 the same sources at their '
                   'reconstructed XYZ; G3 official valid output. Same domains, interpolation, support rule, tau; d 6 mm main, 5 / 7 mm sensitivity'),
)


def register() -> dict:
    mp = model_prep_status()
    A = a3_plan(); h = input_hashes(A)
    missing = [f for f, v in h.items() if v is None]
    if missing:
        raise K.Stop('refused', f'{len(missing)} registered input files missing', detail=missing[:5])
    counts = dict(views=len(A), by_part=A.part.value_counts().to_dict(), by_class=A.reuse_class.value_counts().to_dict(),
                  by_part_class=A.groupby(['part', 'reuse_class']).size().rename('n').reset_index().to_dict('records'),
                  replay=A.replay.value_counts().to_dict(), decodes_planned=int((A.reuse_class == 'new_DF').sum() + A.replay.notna().sum()))
    if counts['decodes_planned'] > K.CAPS['decode_total']:
        raise K.Stop('budget', f"planned decodes {counts['decodes_planned']} > {K.CAPS['decode_total']}")
    plan = dict(t=K.now(), model_prep01=mp, caps=K.CAPS, rules=RULES, a3_counts=counts,
                a3_keys=A[['input_key', 'part', 'surface', 'layout', 'size_mm', 'phase_label', 'sigma', 'rep', 'reuse_class', 'replay']].to_dict('records'),
                inputs_sha256=h, n_input_files=len(h),
                reuse_sources=dict(S12=K.rel(K.S12 / 'results' / 'views.csv'), S12_sha256=K.sha(K.S12 / 'results' / 'views.csv'),
                                   C4_DF=K.rel(K.C4 / 'results' / 'C1_dots_cases.csv'), C4_DF_sha256=K.sha(K.C4 / 'results' / 'C1_dots_cases.csv')))
    K.run_put('plan', plan)
    return dict(counts=counts, n_input_files=len(h), model_prep01=mp['finding'])


if __name__ == '__main__':
    import json
    print(json.dumps(K.clean(register()), ensure_ascii=False, indent=1, default=str))
