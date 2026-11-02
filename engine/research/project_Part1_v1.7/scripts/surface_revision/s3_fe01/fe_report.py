# Summarize front-end comparisons and their implications for parameter rankings.

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fe_common as K

R = K.RESULTS
SURF = {"B_curv": "EF", "M10_curv": "M10", "M24_curv": "M24"}


def pct(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(100 * float(x), 2)


def section31() -> pd.DataFrame:
    A = pd.read_csv(R / "A_s12_cells.csv", low_memory=False); B = pd.read_csv(R / "B_size02_cells.csv", low_memory=False)
    cols = ["regime", "surface", "size_mm", "sigma", "rule", "n_images", "n_phases", "glob_Ctau_mean", "glob_Ctau_min", "glob_Ctau_max", "glob_recovery_mean", "glob_pt_rmse_mean", "glob_surf_rmse_mean", "id_wrong", "id_unconf", "n_no_output"]
    a = A[A.layout == "S1_R8x4"][cols].assign(source="S12-RESOURCE-01 (A)")
    b = B[cols].assign(source="S1-SIZE-02 (B)")
    U = pd.concat([a, b]).drop_duplicates(["regime", "surface", "size_mm", "sigma", "rule"]).sort_values(["regime", "rule", "size_mm", "surface", "sigma"])
    U["surface"] = U.surface.map(SURF)
    for c in ("glob_Ctau_mean", "glob_Ctau_min", "glob_Ctau_max", "glob_recovery_mean"):
        U[c] = U[c].map(pct)
    K.save_csv(U, R / "R_31_fuenf_radien_cells.csv")

    W = U.pivot_table(index=["regime", "rule", "size_mm"], columns="sigma", values="glob_Ctau_mean", aggfunc="mean").round(2).reset_index()
    K.save_csv(W, R / "R_31_fuenf_radien_dreiflaechen_mittel.csv")
    return W


def section32() -> pd.DataFrame:
    A = pd.read_csv(R / "A_s12_cells.csv", low_memory=False)
    s2 = A[(A.layout != "S1_R8x4") & (A.regime == "resource_cap_11695")].copy(); s2["surface"] = s2.surface.map(SURF)
    for c in ("glob_Ctau_mean", "glob_Ctau_min", "glob_Ctau_max", "glob_recovery_mean"):
        s2[c] = s2[c].map(pct)
    K.save_csv(s2, R / "R_32_layouts_cells.csv")
    W = s2.pivot_table(index=["rule", "layout"], columns="sigma", values="glob_Ctau_mean", aggfunc="mean").round(2).reset_index()
    K.save_csv(W, R / "R_32_layouts_dreiflaechen_mittel.csv")
    rank = []
    for sg, g in s2.groupby("sigma"):
        for rule, h in g.groupby("rule"):
            t = h.groupby("layout").glob_Ctau_mean.mean().sort_values(ascending=False)
            rank.append(dict(sigma=sg, rule=rule, order=" > ".join(f"{k} {v:.1f}" for k, v in t.items())))
    K.save_csv(pd.DataFrame(rank), R / "R_32_layouts_rangfolge.csv")
    return W


def df_losses() -> dict:

    P = pd.read_csv(R / "A_s12_paired.csv", low_memory=False)
    P["worse"] = P.d_glob_Ctau < -1e-9; P["better"] = P.d_glob_Ctau > 1e-9
    g = P.groupby(["layout", "sigma"]).agg(n=("worse", "size"), n_worse=("worse", "sum"), n_better=("better", "sum"), mean_d_pp=("d_glob_Ctau", lambda x: 100 * x.mean()),
                                            min_d_pp=("d_glob_Ctau", lambda x: 100 * x.min()), mean_floor_removed=("n_candidates_D0", "mean")).reset_index()
    g["mean_d_pp"] = g.mean_d_pp.round(2); g["min_d_pp"] = g.min_d_pp.round(2)
    K.save_csv(g, R / "R_A_DF_minus_D0_by_layout_sigma.csv")
    worst = P.sort_values("d_glob_Ctau").head(12)[["input_key", "layout", "size_mm", "surface", "sigma", "glob_Ctau_D0", "glob_Ctau_DF", "n_accepted_D0", "n_accepted_DF", "n_candidates_D0", "n_candidates_DF"]]
    K.save_csv(worst, R / "R_A_DF_schlechter_beispiele.csv")
    tot = dict(n=int(len(P)), worse=int(P.worse.sum()), better=int(P.better.sum()), equal=int((~P.worse & ~P.better).sum()),
               worse_by_sigma=P[P.worse].groupby("sigma").size().to_dict(), worse_by_layout=P[P.worse].groupby("layout").size().to_dict(),
               r8x25_sigma40=g[(g.layout == "R8x2.5") & (g.sigma == 40.0)].to_dict("records"))
    return tot


def section34() -> dict | None:
    p = R / "C_s4_cases.csv"
    if not p.exists():
        return None
    C = pd.read_csv(p, low_memory=False); C["surface"] = C.surface.map(SURF)
    out = {}

    six = C[C.dataset == "S4-REAL-02"].copy()
    rows = []
    for rule in ("D0", "DF"):
        d = six[six.rule == rule]
        s0 = d[d.strategy == "S0"].set_index(["surface", "sigma", "phase"]).glob_Ctau
        for strat in ("S1", "S2", "S3"):
            x = d[d.strategy == strat]
            for (sf, sg), g in x.groupby(["surface", "sigma"]):
                blk = []
                for b, h in g.groupby("block"):
                    diffs = [float(r.glob_Ctau) - float(s0.get((sf, sg, r.phase), np.nan)) for r in h.itertuples()]
                    diffs = [v for v in diffs if not np.isnan(v)]
                    if diffs:
                        blk.append(float(np.median(diffs)))
                if blk:
                    rows.append(dict(rule=rule, strategy=strat, surface=sf, sigma=sg, n_blocks=len(blk), median_pp=round(100 * float(np.median(blk)), 2), min_pp=round(100 * min(blk), 2), max_pp=round(100 * max(blk), 2), n_pos=int(sum(v > 0 for v in blk))))
    T = pd.DataFrame(rows); K.save_csv(T, R / "R_34_sechs_sigma_block_differenzen.csv")

    lev = six.groupby(["rule", "strategy", "surface", "sigma"]).glob_Ctau.median().map(pct).reset_index()
    K.save_csv(lev, R / "R_34_sechs_sigma_niveaus.csv")

    new = C[C.dataset == "S4-REAL-07"].copy(); rows = []
    for rule in ("D0", "DF"):
        d = new[new.rule == rule]
        s0 = d[d.strategy == "S0"].groupby(["surface", "sigma", "phase"]).glob_Ctau.mean()
        for strat in sorted(set(d.strategy.dropna()) - {"S0"}):
            x = d[d.strategy == strat]
            for (sf, sg), g in x.groupby(["surface", "sigma"]):
                blk = []
                for b, h in g.groupby("block"):
                    vals = [float(r.glob_Ctau) - float(s0.get((sf, sg, r.phase), np.nan)) for r in h.itertuples()]
                    vals = [v for v in vals if not np.isnan(v)]
                    if vals:
                        blk.append(float(np.mean(vals)))
                if blk:
                    rows.append(dict(rule=rule, strategy=strat, surface=sf, sigma=sg, n_blocks=len(blk), median_pp=round(100 * float(np.median(blk)), 2), min_pp=round(100 * min(blk), 2), max_pp=round(100 * max(blk), 2), n_pos=int(sum(v > 0 for v in blk))))
    N = pd.DataFrame(rows); K.save_csv(N, R / "R_34_neue_layouts_block_differenzen.csv")
    Pd = pd.read_csv(R / "C_s4_paired.csv", low_memory=False)
    out = dict(six_noise_images=int(len(six) // 2), new_layout_images=int(len(new) // 2),
               df_vs_d0_by_sigma={float(k): dict(n=int(len(g)), mean_pp=round(100 * g.d_glob_Ctau.mean(), 2), worse=int((g.d_glob_Ctau < -1e-9).sum()), better=int((g.d_glob_Ctau > 1e-9).sum())) for k, g in Pd.groupby("sigma")})
    return out


def main() -> None:
    w31 = section31(); w32 = section32(); loss = df_losses(); s34 = section34()
    A = K.load_json(R / "A_s12_summary.json"); B = K.load_json(R / "B_size02_summary.json"); C = K.load_json(R / "C_s4_summary.json") if (R / "C_s4_summary.json").exists() else None
    summ = dict(task=K.TASK_ID, t=K.now(), A=A, B=B, C=C, df_losses=loss, s34=s34)
    K.save_json(R / "R_summary.json", summ)
    print(json.dumps(K.clean(dict(A=dict(ok=A["ok"], xcheck=A["crosscheck_s3_code04"]), B=dict(ok=B["ok"], old_df_equal=B["old_df_equal"]), C=(dict(ok=C["ok"], by=C["by_dataset"]) if C else None),
                                   df_losses={k: v for k, v in loss.items() if k != "r8x25_sigma40"}, r8x25_sigma40=loss["r8x25_sigma40"], s34=s34)), ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
