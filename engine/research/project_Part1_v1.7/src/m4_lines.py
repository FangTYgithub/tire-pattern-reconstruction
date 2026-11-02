# Detect continuous line centres, number their segments and reconstruct 3D samples. The P4 settings
# are supplied by the study adapters.

from __future__ import annotations

import math

import cv2
import numpy as np

from m4_reconstruction import (_resolve_sensor_cfg, _detect_vertical_line_centres, _m3_cfg, BINARY_THRESHOLD, BORDER_MARGIN_PX,
                               MEASUREMENT_VOLUME_MM, EPI_T_MARGIN)

STATUS = {0: "accepted", 1: "no_candidate", 2: "segment_unknown", 3: "inconsistent_with_segment", 4: "triangulation_invalid",
          5: "near_horizontal", 6: "short_chain", 7: "border", 8: "guard_chain_short", 9: "guard_width", 10: "guard_end"}
DETECTORS = ("rowcentroid", "rowcentroid_sg", "steger", "rowcentroid_ext")
LINES_VERSION = "m4_lines_v1.102"
GUARD_RULES = ("chain_then_width_v1", "width_then_split_then_chain_v2", "end_then_width_then_split_then_chain_v3")
WIDTH_MODELS = ("band_v1", "row_section_v2")

NOISE_K_DEFAULT = 3.0  # Multiplier on estimated background noise for noise-based thresholding.
NOISE_Q = 0.8413  # Background quantile used by the noise estimator.
NOISE_MIN_BG_FRAC = 0.05  # Minimum fraction of background pixels required for estimation.
NOISE_SMOOTH_PX = 7  # Background-estimation smoothing window [px].
NOISE_DILATE_PX = 9  # Dilation size for excluding illuminated pixels [px].
STEGER_NOISE_COEF = {1.0: 0.745, 1.6: 0.167, 2.4: 0.048}
VOTE_FRAC = 0.9  # Required leading segment-vote fraction; study value: 0.90.
STEGER_THR_FRAC = 0.3  # Fraction of the predicted line response used in the default threshold.
STEGER_NX_MIN = 0.3  # Minimum absolute horizontal normal component for accepted line centres.
STEGER_SUBPX_TOL = 0.55  # Maximum subpixel centre correction [px]; fixed study value: 0.55.
SG_MIN_ROWS = 5  # Minimum chain length [rows] for smoothing.
UNKNOWN_ID = -1_000_000


# Resolve line orientation, spacing and registered line identifiers from pattern metadata.
def line_family(spec: dict) -> dict:

    phi = math.radians(float(spec["angle_deg"]))
    return dict(phi_deg=float(spec["angle_deg"]), n=np.array([math.cos(phi), math.sin(phi)]), t=np.array([-math.sin(phi), math.cos(phi)]),
                s_px=float(spec["spacing_px"]), w_px=float(spec["width_px"]), phase=float(spec["phase_frac"]),
                c=np.asarray(spec["centre_uv"], np.float64), ids=np.asarray(spec["ids"], np.int64))


def line_q(fam: dict, k) -> np.ndarray:

    return (np.asarray(k, np.float64) + fam["phase"]) * fam["s_px"]


# Associate illuminated projector pixels with their line centres.
def raster_centrelines(pattern: np.ndarray, fam: dict) -> dict:

    pat = np.asarray(pattern, np.float64)
    H, W = pat.shape
    lit = pat > 0.5 * pat.max() if pat.max() > 0 else np.zeros_like(pat, bool)
    vv, uu = np.nonzero(lit)
    n, c = fam["n"], fam["c"]
    q = (uu - c[0]) * n[0] + (vv - c[1]) * n[1]
    r = q / fam["s_px"] - fam["phase"]
    k = np.round(r).astype(np.int64)
    ids = fam["ids"]
    dense = np.full((len(ids), H), np.nan)
    out = {}
    for j, kk in enumerate(ids):
        sel = k == kk
        if not sel.any():
            out[int(kk)] = dict(rows=np.zeros(0, np.int64), u=np.zeros(0), q_offset_px=None, row_dev_std_px=None, row_dev_max_px=None)
            continue
        vs, us = vv[sel], uu[sel]
        cnt = np.bincount(vs, minlength=H); su = np.bincount(vs, weights=us, minlength=H)
        rows = np.flatnonzero(cnt)
        u_mean = su[rows] / cnt[rows]
        dense[j, rows] = u_mean
        q_rows = (u_mean - c[0]) * n[0] + (rows - c[1]) * n[1] - float(line_q(fam, kk))
        out[int(kk)] = dict(rows=rows, u=u_mean, q_offset_px=float(np.mean(q[sel]) - float(line_q(fam, kk))), row_dev_std_px=float(np.std(q_rows)), row_dev_max_px=float(np.abs(q_rows).max()))
    out["_u_of_row"] = dense
    return out


def lines_from_pattern(pattern: np.ndarray) -> list[dict]:

    pat = np.asarray(pattern, np.float64)
    b = (pat > 0.5 * pat.max()).astype(np.uint8) if pat.max() > 0 else np.zeros_like(pat, np.uint8)
    n, lbl = cv2.connectedComponents(b, connectivity=8)
    out = []
    for i in range(1, n):
        vv, uu = np.nonzero(lbl == i)
        w = pat[vv, uu]
        cu, cv = float(np.average(uu, weights=w)), float(np.average(vv, weights=w))
        cov = np.cov(np.stack([uu - cu, vv - cv]), aweights=w)
        val, vec = np.linalg.eigh(cov)
        t = vec[:, 1]
        if t[1] < 0:
            t = -t
        out.append(dict(label=i, n_px=int(len(uu)), centroid_uv=(cu, cv), direction_uv=(float(t[0]), float(t[1])), angle_deg=float(math.degrees(math.atan2(-t[0], t[1])))))
    return out


def _geometry(cfg: dict):
    from m3_virtual_camera import build_intrinsics, build_extrinsics
    m3 = _m3_cfg(cfg)
    K = build_intrinsics(m3); proj, cam = build_extrinsics(m3)
    return K, proj, cam, m3


def _dirs(uv: np.ndarray, K: np.ndarray, ext: dict) -> np.ndarray:
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    d = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))], axis=1)
    return (ext["R_inv"] @ d.T).T


def _project(P: np.ndarray, K: np.ndarray, ext: dict, W: int, H: int):
    from m3_virtual_camera import project_to_camera
    return project_to_camera(np.asarray(P, np.float64).reshape(-1, 3), K, ext, W, H)


# Map camera rays to projector tracks across the depth search interval.
def epipolar_in_projector(cam_uv: np.ndarray, cfg: dict, z0: float, z1: float):

    K, proj, cam, m3 = _geometry(cfg)
    W, H = m3["image_width_px"], m3["image_height_px"]
    d = _dirs(cam_uv, K, cam); o = cam["center"]

    def at(z):
        t = (z - o[2]) / d[:, 2]
        return o + t[:, None] * d
    P0, _ = _project(at(z0), K, proj, W, H)
    P1, _ = _project(at(z1), K, proj, W, H)
    return np.asarray(P0, np.float64), np.asarray(P1, np.float64)


def triangulate64(cam_uv: np.ndarray, proj_uv: np.ndarray, cfg: dict):

    K, proj, cam, m3 = _geometry(cfg)
    d_c = _dirs(cam_uv, K, cam); d_p = _dirs(proj_uv, K, proj)
    d_c /= np.linalg.norm(d_c, axis=1, keepdims=True); d_p /= np.linalg.norm(d_p, axis=1, keepdims=True)
    O_c, O_p = cam["center"], proj["center"]
    w = O_c - O_p
    a = np.einsum("ni,ni->n", d_c, d_p); b1 = d_c @ w; b2 = d_p @ w
    denom = a * a - 1.0
    valid = np.abs(denom) > 1e-10
    s = np.where(valid, (a * b1 - b2) / np.where(valid, denom, 1.0), 0.0)
    t = np.where(valid, a * s - b1, 0.0)
    valid &= (t > 0) & (s > 0)
    P = O_c + t[:, None] * d_c; Q = O_p + s[:, None] * d_p
    pts = 0.5 * (P + Q)
    pts[~valid] = np.nan
    return pts, np.linalg.norm(P - Q, axis=1), valid


# Intersect camera rays with the planes of the assigned projector lines.
def light_plane_points(cam_uv: np.ndarray, line_id: np.ndarray, fam: dict, cfg: dict) -> np.ndarray:

    K, proj, cam, m3 = _geometry(cfg)
    c, n, t = fam["c"], fam["n"], fam["t"]
    d_c = _dirs(cam_uv, K, cam); o_c = cam["center"]; o_p = proj["center"]
    out = np.full((len(cam_uv), 3), np.nan)
    for k in np.unique(line_id):
        sel = line_id == k
        q = float(line_q(fam, k))
        p_a = c + q * n - 200.0 * t; p_b = c + q * n + 200.0 * t
        da, db = _dirs(p_a[None, :], K, proj)[0], _dirs(p_b[None, :], K, proj)[0]
        N = np.cross(da, db); N /= np.linalg.norm(N)
        den = d_c[sel] @ N
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = ((o_p - o_c) @ N) / den
        pts = o_c + lam[:, None] * d_c[sel]
        pts[~np.isfinite(lam) | (lam <= 0)] = np.nan
        out[sel] = pts
    return out


def binarise(camera_image: np.ndarray):

    if camera_image.dtype != np.uint8:
        raise TypeError("camera_image must be uint8")
    if BINARY_THRESHOLD == 0:
        thr, binary = cv2.threshold(camera_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thr, binary = cv2.threshold(camera_image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    return float(thr), binary


# Estimate the background noise scale from unilluminated image regions.
def estimate_noise_sigma(camera_image: np.ndarray) -> tuple:

    sm = cv2.blur(camera_image, (NOISE_SMOOTH_PX, NOISE_SMOOTH_PX))
    _, b = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg = cv2.dilate(b, np.ones((NOISE_DILATE_PX, NOISE_DILATE_PX), np.uint8)) == 0
    v = camera_image[bg].astype(np.float64)
    frac = float(v.size / camera_image.size)
    if frac < NOISE_MIN_BG_FRAC:
        return None, "insufficient_background", frac, None
    zf = float((v == 0).mean())
    sig = float(np.quantile(v, NOISE_Q))
    if zf < 0.30 and sig > 0:
        return sig, "background_not_clipped_zero_mean", frac, zf
    return sig, "ok", frac, zf


def binarise_rule(camera_image: np.ndarray, rule: str = "otsu", noise_k: float = NOISE_K_DEFAULT):

    thr_o, binary = binarise(camera_image)
    info = dict(binarise_rule=rule, otsu_threshold=float(thr_o), threshold=float(thr_o), noise_sigma_hat=None, noise_status="not_estimated")
    if rule == "otsu":
        return float(thr_o), binary, info
    if rule != "otsu_noise_floor":
        raise ValueError(f"unknown binarise rule '{rule}'")
    sig, status, frac, zf = estimate_noise_sigma(camera_image)
    info.update(noise_sigma_hat=sig, noise_status=status, noise_bg_frac=frac, noise_bg_zero_frac=zf, noise_k=float(noise_k))
    if sig is None:
        info["threshold_source"] = "otsu_fallback"
        return float(thr_o), binary, info
    thr = max(float(thr_o), float(noise_k) * sig)
    info.update(threshold=thr, threshold_source=("noise_floor" if thr > thr_o else "otsu"))
    binary = ((camera_image.astype(np.float64) > thr).astype(np.uint8) * 255)
    return thr, binary, info


def detect_rowcentroid_ext(camera_image: np.ndarray, binary: np.ndarray):

    from m4_reconstruction import _line_labels, _group_consecutive, EXCLUDE_BORDER_BLOBS
    H, W = camera_image.shape
    n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(binary)
    valid_labels, n_specks = _line_labels(label_map, n_labels)
    if not valid_labels:
        return np.empty((0, 2), np.float32), np.zeros(0, np.int64), np.zeros(0), np.zeros(0), {"n_vlines": 0, "n_border_excluded": 0, "n_line_specks": n_specks}
    clean_binary = np.where(np.isin(label_map, list(valid_labels)), 255, 0).astype(binary.dtype)
    feats, comps, widths, peaks = [], [], [], []
    n_border = 0
    for v in range(H):
        row_binary = clean_binary[v, :]; row_image = camera_image[v, :].astype(np.float64)
        bright_cols = np.where(row_binary > 0)[0]
        if len(bright_cols) == 0:
            continue
        for group in _group_consecutive(bright_cols):
            if EXCLUDE_BORDER_BLOBS and (group[0] <= BORDER_MARGIN_PX or group[-1] >= W - 1 - BORDER_MARGIN_PX):
                n_border += 1
                continue
            weights = row_image[group]
            if weights.sum() < 1e-6:
                continue
            feats.append([float(np.average(group, weights=weights)), float(v)]); comps.append(int(label_map[v, int(group[0])]))
            widths.append(float(len(group))); peaks.append(float(weights.max()))
    arr = np.array(feats, np.float32) if feats else np.empty((0, 2), np.float32)
    return arr, np.asarray(comps, np.int64), np.asarray(widths), np.asarray(peaks), {"n_vlines": len(valid_labels), "n_border_excluded": n_border, "n_line_specks": n_specks}


# Predict the observed band-width range from line width, blur and depth.
def expected_band_width_px(w_px: float, psf_px: float, wd_mm: float, z0_mm: float, z1_mm: float) -> tuple[float, float]:

    ms = [(wd_mm + z0_mm) / wd_mm, (wd_mm + z1_mm) / wd_mm]
    vals = [math.sqrt((w_px * m) ** 2 + (2.3548 * psf_px) ** 2) for m in ms]
    return min(vals), max(vals)


# Estimate row-section widths over the sampled measurement geometry.
def expected_row_width_px_geom(fam: dict, cfg: dict, psf_px: float, z0_mm: float, z1_mm: float, n_grid: int = 5, eps_px: float = 2.0) -> tuple[float, float, dict]:

    K, proj, cam, m3 = _geometry(cfg)
    W, H = m3["image_width_px"], m3["image_height_px"]
    cs = np.linspace(0.1, 0.9, n_grid) * (W - 1); rs = np.linspace(0.1, 0.9, n_grid) * (H - 1)
    uu, vv = np.meshgrid(cs, rs); cam_uv = np.column_stack([uu.ravel(), vv.ravel()])
    n, t = fam["n"], fam["t"]
    d_c = _dirs(cam_uv, K, cam); o_c = cam["center"]; o_p = proj["center"]
    runs, gs, nxs = [], [], []
    for z in (float(z0_mm), 0.5 * (float(z0_mm) + float(z1_mm)), float(z1_mm)):
        tt = (z - o_c[2]) / d_c[:, 2]; X = o_c + tt[:, None] * d_c
        Qp, _ = _project(X, K, proj, W, H); Qp = np.asarray(Qp, np.float64)

        def to_cam(Q, _z=z):
            d = _dirs(Q, K, proj); s = (_z - o_p[2]) / d[:, 2]; P = o_p + s[:, None] * d
            uv, _ = _project(P, K, cam, W, H)
            return np.asarray(uv, np.float64)
        dn = (to_cam(Qp + eps_px * n) - to_cam(Qp - eps_px * n)) / (2.0 * eps_px)
        dt = (to_cam(Qp + eps_px * t) - to_cam(Qp - eps_px * t)) / (2.0 * eps_px)
        tl = np.linalg.norm(dt, axis=1); tdir = dt / np.where(tl[:, None] > 0, tl[:, None], 1.0)
        ncam = np.column_stack([tdir[:, 1], -tdir[:, 0]])
        g = np.abs((dn * ncam).sum(axis=1)); nx = np.abs(ncam[:, 0])
        runs.append(np.sqrt((fam["w_px"] * g) ** 2 + (2.3548 * psf_px) ** 2) / np.maximum(nx, 1e-6)); gs.append(g); nxs.append(nx)
    runs = np.concatenate(runs); gs = np.concatenate(gs); nxs = np.concatenate(nxs)
    info = dict(model="row_section_v2", w_px=float(fam["w_px"]), psf_px=float(psf_px), z_mm=[float(z0_mm), float(z1_mm)], normal_scale_min=float(gs.min()), normal_scale_max=float(gs.max()),
                n_cam_x_min=float(nxs.min()), n_cam_x_max=float(nxs.max()), run_min=float(runs.min()), run_max=float(runs.max()), run_median=float(np.median(runs)))
    return float(runs.min()), float(runs.max()), info


def split_segments_at_gaps(feat: np.ndarray, chain: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:

    seg = np.asarray(chain, np.int64).copy(); valid = np.asarray(valid, bool)
    seg[~valid] = -1
    next_id = int(chain.max()) + 1 if len(chain) else 0
    n_split = 0
    for cid in np.unique(chain[(chain >= 0) & ~valid]):
        sel = np.flatnonzero((chain == cid) & valid)
        if len(sel) == 0:
            continue
        o = sel[np.argsort(feat[sel, 1])]; rows = np.round(feat[o, 1]).astype(np.int64)
        brk = np.flatnonzero(np.diff(rows) > 1)
        if len(brk) == 0:
            continue
        n_split += 1
        for part in np.split(o, brk + 1)[1:]:
            seg[part] = next_id; next_id += 1
    return seg, n_split


def end_guard(feat: np.ndarray, chain: np.ndarray, k_rows: int, du_max: float) -> tuple[np.ndarray, dict]:

    N = len(feat); reason = np.zeros(N, np.int64); k = int(k_rows)
    info = dict(end_guard_rows=k, n_chain_ends=0, n_terminal_ends=0, n_gap_ends=0, n_guard_end=0)
    if k <= 0 or N == 0:
        return reason, info
    u = feat[:, 0].astype(np.float64); v = np.round(feat[:, 1]).astype(np.int64)
    by_row: dict = {}
    for i in range(N):
        by_row.setdefault(int(v[i]), []).append(i)

    def continuation(v_end: int, u_end: float, slope: float, direction: int) -> bool:
        for d in range(1, k + 1):
            vv = v_end + direction * d; cand = by_row.get(vv)
            if not cand:
                continue
            ue = u_end + slope * direction * d
            if np.any(np.abs(u[cand] - ue) <= du_max + 1e-9):
                return True
        return False

    for cid in np.unique(chain[chain >= 0]):
        sel = np.flatnonzero(chain == cid)
        o = sel[np.argsort(v[sel], kind="stable")]; vv = v[o]; uu = u[o]
        n = len(o); m = min(5, n)
        slope_top = float(np.median(np.diff(uu[:m]) / np.maximum(np.diff(vv[:m]), 1))) if m >= 2 else 0.0
        slope_bot = float(np.median(np.diff(uu[-m:]) / np.maximum(np.diff(vv[-m:]), 1))) if m >= 2 else 0.0
        info["n_chain_ends"] += 2
        if continuation(int(vv[0]), float(uu[0]), slope_top, -1):
            info["n_gap_ends"] += 1
        else:
            info["n_terminal_ends"] += 1; reason[o[vv <= vv[0] + k - 1]] = 10
        if continuation(int(vv[-1]), float(uu[-1]), slope_bot, +1):
            info["n_gap_ends"] += 1
        else:
            info["n_terminal_ends"] += 1; reason[o[vv >= vv[-1] - k + 1]] = 10
    info["n_guard_end"] = int((reason == 10).sum())
    return reason, info


# Remove short chains and samples outside the permitted width range.
def apply_guards(feat: np.ndarray, chain: np.ndarray, widths: np.ndarray | None, min_chain_px: float, width_window: tuple | None) -> tuple[np.ndarray, dict]:

    N = len(feat); reason = np.zeros(N, np.int64)
    if min_chain_px and min_chain_px > 0:
        for cid in np.unique(chain[chain >= 0]):
            sel = np.flatnonzero(chain == cid)
            if len(sel) == 1:
                L = 0.0
            else:
                o = sel[np.argsort(feat[sel, 1])]
                d = np.diff(feat[o].astype(np.float64), axis=0)
                L = float(np.sqrt((d ** 2).sum(axis=1)).sum())
            if L < float(min_chain_px):
                reason[sel] = 8
    if width_window is not None and widths is not None and len(widths) == N:
        lo, hi = float(width_window[0]), float(width_window[1])
        bad = (widths < lo) | (widths > hi)
        reason[bad & (reason == 0)] = 9
    return reason, dict(n_guard_chain_short=int((reason == 8).sum()), n_guard_width=int((reason == 9).sum()), min_chain_px=float(min_chain_px or 0.0), width_window=(None if width_window is None else [float(width_window[0]), float(width_window[1])]))


# Convert estimated image noise into the threshold for the chosen filter scale.
def steger_noise_threshold(sigma_hat: float | None, sigma_f: float, coef: dict | None = None) -> float:

    if sigma_hat is None or sigma_hat <= 0:
        return 0.0
    table = STEGER_NOISE_COEF if coef is None else {float(k): float(v) for k, v in coef.items()}
    ks = sorted(table); cs = [table[k] for k in ks]
    c = float(np.interp(float(sigma_f), ks, cs))
    return c * float(sigma_hat)


# Locate line centres from intensity-weighted row profiles.
def detect_rowcentroid(camera_image: np.ndarray):

    thr, binary = binarise(camera_image)
    feats, info = _detect_vertical_line_centres(camera_image, binary)
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(binary)
    comp = np.zeros(len(feats), np.int64)
    if len(feats):
        uu = np.clip(np.round(feats[:, 0]).astype(int), 0, camera_image.shape[1] - 1)
        vv = np.clip(np.round(feats[:, 1]).astype(int), 0, camera_image.shape[0] - 1)
        comp = lbl[vv, uu].astype(np.int64)
    info = dict(info, otsu_threshold=thr, n_components_binary=int(n_lbl - 1))
    return feats.astype(np.float32), comp, info


# Link neighbouring centre samples into continuous line chains.
def chain_features(feat: np.ndarray, comp: np.ndarray, du_max: float) -> np.ndarray:

    N = len(feat)
    chain = np.full(N, -1, np.int64)
    next_id = 0
    for c in np.unique(comp):
        sel = np.flatnonzero(comp == c)
        rows = np.round(feat[sel, 1]).astype(int); us = feat[sel, 0].astype(np.float64)
        o = np.lexsort((us, rows)); sel, rows, us = sel[o], rows[o], us[o]
        active: list = []
        i = 0
        while i < len(sel):
            v = int(rows[i]); j = i
            while j < len(sel) and rows[j] == v:
                j += 1
            active = [a for a in active if a[1] == v - 1]
            used = [False] * len(active); updates = []; new = []
            for m in range(i, j):
                u = float(us[m]); best = -1; bestd = du_max
                for a_i, a in enumerate(active):
                    if not used[a_i] and abs(a[2] - u) <= bestd:
                        best, bestd = a_i, abs(a[2] - u)
                if best >= 0:
                    chain[sel[m]] = active[best][0]; used[best] = True; updates.append((best, v, u))
                else:
                    chain[sel[m]] = next_id; new.append([next_id, v, u]); next_id += 1
            for a_i, v_, u_ in updates:
                active[a_i] = [active[a_i][0], v_, u_]
            active = [a for a_i, a in enumerate(active) if used[a_i]] + new
            i = j
    return chain


# Smooth chain coordinates with a local Savitzky-Golay filter.
def smooth_chains(feat: np.ndarray, chain: np.ndarray, h: int, min_rows: int = SG_MIN_ROWS) -> tuple[np.ndarray, int]:

    from scipy.signal import savgol_filter
    out = feat.astype(np.float64).copy(); n_sm = 0
    for cid in np.unique(chain[chain >= 0]):
        sel = np.flatnonzero(chain == cid)
        if len(sel) < min_rows:
            continue
        sel = sel[np.argsort(feat[sel, 1])]
        win = min(2 * int(h) + 1, len(sel) if len(sel) % 2 == 1 else len(sel) - 1)
        if win < 5:
            continue
        out[sel, 0] = savgol_filter(feat[sel, 0].astype(np.float64), win, 2, mode="interp")
        n_sm += len(sel)
    return out.astype(np.float32), n_sm


# Calculate the expected second-derivative response of a blurred line.
def steger_design_response(w_px: float, sigma: float, psf_px: float = 1.0, height: float = 1.0) -> float:

    a = w_px / 2.0; se = math.sqrt(sigma ** 2 + psf_px ** 2)
    return float(height * 2.0 * a * math.exp(-0.5 * (a / se) ** 2) / math.sqrt(2.0 * math.pi) / se ** 3)


# Locate subpixel line centres from Gaussian image derivatives.
def detect_steger(camera_image: np.ndarray, sigma: float, threshold: float, nx_min: float = STEGER_NX_MIN):

    from scipy.ndimage import gaussian_filter
    f = camera_image.astype(np.float64)
    rx = gaussian_filter(f, sigma, order=(0, 1)); ry = gaussian_filter(f, sigma, order=(1, 0))
    rxx = gaussian_filter(f, sigma, order=(0, 2)); ryy = gaussian_filter(f, sigma, order=(2, 0)); rxy = gaussian_filter(f, sigma, order=(1, 1))
    disc = np.sqrt(((rxx - ryy) / 2.0) ** 2 + rxy ** 2)
    lam = (rxx + ryy) / 2.0 - disc
    ex = rxy.copy(); ey = lam - rxx
    small = np.abs(rxy) < 1e-12
    ex[small] = np.where(rxx[small] <= ryy[small], 1.0, 0.0); ey[small] = np.where(rxx[small] <= ryy[small], 0.0, 1.0)
    nrm = np.hypot(ex, ey); nrm[nrm == 0] = 1.0
    nx_, ny_ = ex / nrm, ey / nrm
    denom = rxx * nx_ ** 2 + 2.0 * rxy * nx_ * ny_ + ryy * ny_ ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -(rx * nx_ + ry * ny_) / denom
    ridge = (lam <= -float(threshold)) & np.isfinite(t) & (np.abs(t * nx_) <= STEGER_SUBPX_TOL) & (np.abs(t * ny_) <= STEGER_SUBPX_TOL)
    H, W = f.shape
    vv, uu = np.nonzero(ridge)
    nxs = nx_[vv, uu]; tt = t[vv, uu]
    ok = np.abs(nxs) >= nx_min
    u_row = uu[ok] + tt[ok] / nxs[ok]
    n_lbl, lbl = cv2.connectedComponents(ridge.astype(np.uint8), connectivity=8)
    comp = lbl[vv[ok], uu[ok]].astype(np.int64)
    key = comp * 100000 + vv[ok].astype(np.int64)
    if len(key):
        uniq, inv = np.unique(key, return_inverse=True)
        sums = np.bincount(inv, weights=u_row); cnt = np.bincount(inv)
        feat = np.column_stack([sums / cnt, (uniq % 100000).astype(np.float64)])
        comp_out = (uniq // 100000).astype(np.int64)
    else:
        feat = np.zeros((0, 2)); comp_out = np.zeros(0, np.int64)
    border = (feat[:, 0] <= BORDER_MARGIN_PX) | (feat[:, 0] >= W - 1 - BORDER_MARGIN_PX)
    info = dict(n_ridge_px=int(ridge.sum()), n_near_horizontal=int((~ok).sum()), n_ridge_components=int(n_lbl - 1), n_border_excluded=int(border.sum()),
                steger_sigma_px=float(sigma), steger_threshold=float(threshold), n_vlines=int(n_lbl - 1))
    return feat[~border].astype(np.float32), comp_out[~border], info


# Construct geometrically admissible projector-line candidates for each camera sample.
def line_candidates(cam_uv: np.ndarray, fam: dict, cfg: dict, centrelines: dict | None = None) -> dict:

    cfg = _resolve_sensor_cfg(cfg)
    vol = float(cfg.get("measurement_volume_mm", MEASUREMENT_VOLUME_MM))


    z0 = float(cfg.get("search_z_min_mm", 0.0) or 0.0)
    margin = float(cfg["search_margin_mm"]) if cfg.get("search_margin_mm") is not None else float(cfg.get("depth_margin_frac", EPI_T_MARGIN)) * vol
    cam_uv = np.asarray(cam_uv, np.float64).reshape(-1, 2)
    ids = fam["ids"]; N, K = len(cam_uv), len(ids)
    if N == 0:
        return dict(Q=np.zeros((0, K, 2)), Q_design=np.zeros((0, K, 2)), z=np.zeros((0, K)), admissible=np.zeros((0, K), bool), mu=np.zeros((0, K)), dq=np.zeros(0), ids=ids,
                    sin_alpha=np.zeros(0), search_volume_mm=vol, depth_margin_mm=margin, window_z0_mm=z0, window_z1_mm=z0 + vol, centre_source="raster" if centrelines is not None else "analytic")
    P0, P1 = epipolar_in_projector(cam_uv, cfg, z0, z0 + vol)
    n, c, t = fam["n"], fam["c"], fam["t"]
    q0 = (P0 - c) @ n; q1 = (P1 - c) @ n; dq = q1 - q0
    e = P1 - P0; e_len = np.linalg.norm(e, axis=1); e_unit = e / np.where(e_len[:, None] > 0, e_len[:, None], 1.0)
    sin_alpha = np.abs(e_unit[:, 0] * t[1] - e_unit[:, 1] * t[0])
    qk = line_q(fam, ids)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = (qk[None, :] - q0[:, None]) / dq[:, None]
    Q_design = P0[:, None, :] + mu[..., None] * e[:, None, :]
    W, H = _m3_cfg(cfg)["image_width_px"], _m3_cfg(cfg)["image_height_px"]
    Q = Q_design.copy()
    has_centre = np.ones((N, K), bool)
    if centrelines is not None:
        dense = centrelines["_u_of_row"]


        vd = np.nan_to_num(Q_design[..., 1], nan=-10.0)
        in_rows = np.isfinite(Q_design[..., 1]) & (vd >= 0.0) & (vd <= H - 1.0)
        kk = np.arange(K)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            m = ((P1[:, 1] - P0[:, 1]) / (P1[:, 0] - P0[:, 0]))[:, None]
        P0u, P0v = P0[:, 0][:, None], P0[:, 1][:, None]
        v0 = np.clip(np.floor(vd).astype(np.int64), 0, H - 2)
        u_lit = np.full((N, K), np.nan); v_lit = np.full((N, K), np.nan)
        for _ in range(2):
            ua, ub = dense[kk, v0], dense[kk, v0 + 1]
            both = np.isfinite(ua) & np.isfinite(ub)
            a = np.where(np.isfinite(ua), ua, ub)
            b = np.where(both, ub - ua, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                u_lit = (a + b * (P0v - v0 - P0u * m)) / (1.0 - b * m)
            v_lit = P0v + (u_lit - P0u) * m
            v0 = np.clip(np.floor(np.nan_to_num(v_lit, nan=-10.0)).astype(np.int64), 0, H - 2)
        u_lit = np.where(in_rows, u_lit, np.nan)
        has_centre = np.isfinite(u_lit) & np.isfinite(v_lit)
        Q = np.where(has_centre[..., None], np.stack([u_lit, v_lit], axis=-1), np.nan)
    cam_rep = np.repeat(cam_uv, K, axis=0)
    pts, resid, valid = triangulate64(cam_rep, np.nan_to_num(Q.reshape(-1, 2), nan=0.0), cfg)
    z = np.where(has_centre, pts[:, 2].reshape(N, K), np.nan)
    inside = has_centre & (Q[..., 0] >= -0.5) & (Q[..., 0] <= W - 0.5) & (Q[..., 1] >= -0.5) & (Q[..., 1] <= H - 0.5)
    adm = np.isfinite(mu) & inside & valid.reshape(N, K) & np.isfinite(z) & (z >= z0 - margin) & (z <= z0 + vol + margin)
    return dict(Q=Q, Q_design=Q_design, z=z, admissible=adm, mu=mu, dq=dq, ids=ids, sin_alpha=sin_alpha, search_volume_mm=vol, depth_margin_mm=margin, window_z0_mm=z0, window_z1_mm=z0 + vol,
                centre_source="raster" if centrelines is not None else "analytic")


# Assign a projector line to each segment by candidate votes.
def number_segments(cand: dict, chain: np.ndarray, vote_frac: float = VOTE_FRAC, valid: np.ndarray | None = None):

    adm = cand["admissible"]; ids = cand["ids"]
    N, K = adm.shape
    n_cand = adm.sum(axis=1)
    line_id = np.full(N, UNKNOWN_ID, np.int64); status = np.full(N, 1, np.int64)
    table = []
    for cid in np.unique(chain):
        sel = np.flatnonzero(chain == cid)
        if valid is not None:
            sel = sel[np.asarray(valid, bool)[sel]]
        if cid < 0 or len(sel) == 0:
            continue
        votes = adm[sel].sum(axis=0) if K else np.zeros(0, int)
        n = len(sel)
        if K == 0 or votes.max() == 0:
            status[sel] = 1
            table.append(dict(segment=int(cid), n=int(n), decision="no_candidate", line_id=None, candidates=[], v_min=float(np.min(sel)), v_max=float(np.max(sel))))
            continue
        order = np.argsort(-votes)
        top = int(votes[order[0]]); second = int(votes[order[1]]) if K > 1 else 0
        if top >= vote_frac * n and second <= (1.0 - vote_frac) * n:
            k = int(ids[order[0]]); line_id[sel] = k
            ok = adm[sel, order[0]]
            status[sel[ok]] = 0; status[sel[~ok]] = 3
            decision = "numbered"
        else:
            status[sel] = 2; decision = "unknown"
        table.append(dict(segment=int(cid), n=int(n), decision=decision, line_id=(int(ids[order[0]]) if decision == "numbered" else None),
                          candidates=[(int(ids[j]), int(votes[j])) for j in order[:4] if votes[j] > 0], top=top, second=second))
    return line_id, status, n_cand, table


# Recover 3D samples from accepted segment identities.
def reconstruct_numbered(cam_uv: np.ndarray, line_id: np.ndarray, status: np.ndarray, cand: dict, fam: dict, cfg: dict):

    N = len(cam_uv)
    pts = np.full((N, 3), np.nan); proj = np.full((N, 2), np.nan); resid = np.full(N, np.nan); plane_diff = np.full(N, np.nan)
    acc = np.flatnonzero(status == 0)
    if len(acc) == 0:
        return pts, proj, status, dict(n_accepted=0, ray_residual_max_mm=None, plane_check_max_mm=None)
    col = {int(k): j for j, k in enumerate(cand["ids"])}
    jj = np.array([col[int(k)] for k in line_id[acc]])
    Q = cand["Q"][acc, jj]
    Qd = cand.get("Q_design", cand["Q"])[acc, jj]
    cam_acc = np.asarray(cam_uv, np.float64)[acc]
    P, r, valid = triangulate64(cam_acc, Q, cfg)
    Pd, _, valid_d = triangulate64(cam_acc, Qd, cfg)
    PL = light_plane_points(cam_acc, line_id[acc], fam, cfg)
    bad = ~valid
    status = status.copy(); status[acc[bad]] = 4
    good = acc[valid]
    pts[good] = P[valid]; proj[good] = Q[valid]; resid[good] = r[valid]
    ok_d = valid & valid_d
    plane_diff[acc[ok_d]] = np.linalg.norm(Pd[ok_d] - PL[ok_d], axis=1)
    raster_dz = np.full(N, np.nan); raster_dz[acc[ok_d]] = P[ok_d, 2] - Pd[ok_d, 2]
    checks = dict(n_accepted=int(len(good)), ray_residual_max_mm=float(np.nanmax(resid)) if len(good) else None,
                  plane_check_max_mm=float(np.nanmax(plane_diff)) if np.isfinite(plane_diff).any() else None,
                  raster_depth_shift_mm=dict(mean=float(np.nanmean(raster_dz)) if np.isfinite(raster_dz).any() else None,
                                             max_abs=float(np.nanmax(np.abs(raster_dz))) if np.isfinite(raster_dz).any() else None))
    return pts, proj, status, checks


# Derive detector defaults from line geometry and the sensor configuration.
def default_params(fam: dict, cfg: dict) -> dict:

    phi = math.radians(fam["phi_deg"])
    return dict(du_max_px=2.0 + abs(math.tan(phi)), sg_window_rows=int(round(fam["s_px"] / 8.0)), steger_sigma_px=max(1.0, fam["w_px"] / (2.0 * math.sqrt(3.0))),
                steger_threshold_frac=STEGER_THR_FRAC, vote_frac=VOTE_FRAC, pattern_peak=float(cfg.get("pattern_peak", 1.0)), psf_sigma_px=float(cfg.get("psf_sigma_px", 1.0)))


# Run line detection, chain checks, candidate voting and reconstruction with the supplied parameters.
def run_olines(camera_image: np.ndarray, spec: dict, cfg: dict | None = None, detector: str = "rowcentroid", pattern: np.ndarray | None = None, **params) -> dict:

    cfg = _resolve_sensor_cfg(cfg)
    fam = line_family(spec)
    centrelines = raster_centrelines(pattern, fam) if pattern is not None else None
    p = {**default_params(fam, cfg), **{k: v for k, v in params.items() if v is not None}}
    p.setdefault("binarise_rule", "otsu"); p.setdefault("noise_k", NOISE_K_DEFAULT); p.setdefault("guard_min_chain_px", 0.0); p.setdefault("guard_width_window", None)
    p.setdefault("guard_width_mult", (0.5, 2.0)); p.setdefault("steger_threshold_rule", "design")
    p.setdefault("vote_valid_only", True)
    p.setdefault("steger_noise_coef", None)
    p.setdefault("steger_noise_coef_id", None)
    p.setdefault("guard_end_rows", None)
    if detector not in DETECTORS:
        raise ValueError(f"unknown line detector '{detector}' (use one of {DETECTORS})")
    widths = None; guard_info = {}
    if detector == "rowcentroid":
        feat, comp, dinfo = detect_rowcentroid(camera_image)
        chain = chain_features(feat, comp, p["du_max_px"])
    elif detector == "rowcentroid_ext":
        thr, binary, binfo = binarise_rule(camera_image, p["binarise_rule"], p["noise_k"])
        feat, comp, widths, peaks, dinfo = detect_rowcentroid_ext(camera_image, binary)
        dinfo = dict(dinfo, **binfo, run_width_median=float(np.median(widths)) if len(widths) else None)
        chain = chain_features(feat, comp, p["du_max_px"])
    elif detector == "rowcentroid_sg":
        feat0, comp, dinfo = detect_rowcentroid(camera_image)
        chain = chain_features(feat0, comp, p["du_max_px"])
        feat, n_sm = smooth_chains(feat0, chain, int(p["sg_window_rows"]))
        dinfo = dict(dinfo, sg_window_rows=int(p["sg_window_rows"]), n_smoothed=int(n_sm))
    else:
        thr_design = p["steger_threshold_frac"] * steger_design_response(fam["w_px"], p["steger_sigma_px"], p["psf_sigma_px"], 255.0 * p["pattern_peak"])
        sig_hat, nstatus = None, "not_estimated"
        thr = thr_design
        if p["steger_threshold_rule"] == "design_or_noise":
            sig_hat, nstatus, _, _ = estimate_noise_sigma(camera_image)
            thr = max(thr_design, steger_noise_threshold(sig_hat, p["steger_sigma_px"], p["steger_noise_coef"]))
        feat, comp, dinfo = detect_steger(camera_image, p["steger_sigma_px"], thr)
        dinfo = dict(dinfo, steger_threshold_rule=p["steger_threshold_rule"], steger_threshold_design=float(thr_design), noise_sigma_hat=sig_hat, noise_status=nstatus,
                     steger_noise_coef=("registered_fp1e-4" if p["steger_noise_coef"] is None else {str(k): float(v) for k, v in p["steger_noise_coef"].items()}),
                     steger_noise_coef_id=(p["steger_noise_coef_id"] or ("registered_fp1e-4" if p["steger_noise_coef"] is None else "custom")), steger_scale_effective=float(p["steger_sigma_px"]),
                     steger_noise_coef_effective=(float(np.interp(float(p["steger_sigma_px"]), sorted(STEGER_NOISE_COEF), [STEGER_NOISE_COEF[k] for k in sorted(STEGER_NOISE_COEF)])) if p["steger_noise_coef"] is None
                                                  else float(np.interp(float(p["steger_sigma_px"]), sorted(float(k) for k in p["steger_noise_coef"]), [float(p["steger_noise_coef"][k]) for k in sorted(p["steger_noise_coef"], key=float)]))))
        chain = chain_features(feat, comp, p["du_max_px"])

    ww = p["guard_width_window"]; width_model = None
    if isinstance(ww, str) and ww in ("auto", "auto_geom"):
        lo_m, hi_m = p["guard_width_mult"]
        z0 = float(cfg.get("search_z_min_mm", 0.0) or 0.0); vol = float(cfg.get("measurement_volume_mm", MEASUREMENT_VOLUME_MM))
        if ww == "auto":
            wmin, wmax = expected_band_width_px(fam["w_px"], p["psf_sigma_px"], float(cfg["working_dist_mm"]), z0, z0 + vol)
            width_model = dict(model="band_v1", run_min=float(wmin), run_max=float(wmax), z_mm=[z0, z0 + vol])
        else:
            wmin, wmax, width_model = expected_row_width_px_geom(fam, cfg, p["psf_sigma_px"], z0, z0 + vol)
        ww = (lo_m * wmin, hi_m * wmax)
    min_chain = float(p["guard_min_chain_px"] or 0.0)
    ge = p["guard_end_rows"]; end_info = {}
    if ge is not None and not p["vote_valid_only"]:
        raise ValueError("guard_end_rows requires vote_valid_only=True")
    if p["vote_valid_only"]:


        guard_reason = np.zeros(len(feat), np.int64)
        if ge is not None:
            k_end = int(4.0 * float(p["steger_sigma_px"] if detector == "steger" else p["psf_sigma_px"]) + 0.5) if ge == "support" else int(ge)
            r10, end_info = end_guard(feat, chain, k_end, float(p["du_max_px"]))
            guard_reason[r10 == 10] = 10
        if ww is not None and widths is not None and len(widths) == len(feat):
            lo, hi = float(ww[0]), float(ww[1]); guard_reason[((widths < lo) | (widths > hi)) & (guard_reason == 0)] = 9
        segments, n_split = split_segments_at_gaps(feat, chain, guard_reason == 0)
        r8, _ = apply_guards(feat, segments, None, min_chain, None)
        guard_reason = np.where((guard_reason == 0) & (r8 == 8), 8, guard_reason)
        segments = np.where(guard_reason == 0, segments, -1)
        guard_info = dict(n_guard_chain_short=int((guard_reason == 8).sum()), n_guard_width=int((guard_reason == 9).sum()), min_chain_px=min_chain,
                          width_window=(None if ww is None else [float(ww[0]), float(ww[1])]), n_chains_split=int(n_split), guard_rule=("end_then_width_then_split_then_chain_v3" if ge is not None else "width_then_split_then_chain_v2"), vote_rule="valid_only",
                          **({"end_guard": str(ge), **end_info} if ge is not None else {}))
    else:
        guard_reason, guard_info = apply_guards(feat, chain, widths, min_chain, ww)
        segments = chain
        guard_info = dict(guard_info, n_chains_split=0, guard_rule="chain_then_width_v1", vote_rule="all_features")
    guard_info["width_model"] = width_model
    cand = line_candidates(feat, fam, cfg, centrelines=centrelines)
    line_id, status, n_cand, table = number_segments(cand, segments, p["vote_frac"], valid=(guard_reason == 0) if p["vote_valid_only"] else None)
    status = np.where(guard_reason > 0, guard_reason, status)
    pts, proj, status, checks = reconstruct_numbered(feat, line_id, status, cand, fam, cfg)
    acc = status == 0
    counts = {STATUS[k]: int((status == k).sum()) for k in STATUS}
    raster_shift = np.full(len(feat), np.nan)
    if acc.any():
        col = {int(k): j for j, k in enumerate(cand["ids"])}
        jj = np.array([col[int(k)] for k in line_id[acc]])
        raster_shift[acc] = cand["Q"][np.flatnonzero(acc), jj, 0] - cand["Q_design"][np.flatnonzero(acc), jj, 0]
    checks = dict(checks, centre_source=cand["centre_source"], raster_shift_px=dict(mean=float(np.nanmean(raster_shift)) if acc.any() else None, max_abs=float(np.nanmax(np.abs(raster_shift))) if acc.any() else None))
    return dict(detector=detector, params=p, features=feat.astype(np.float32), comp=comp.astype(np.int64), chain=chain, segment_id=segments, line_id=line_id, status=status, run_width_px=widths,
                admissible=cand["admissible"], candidate_ids=cand["ids"], guard_reason=guard_reason,
                n_candidates=n_cand.astype(np.int64), points=pts, proj_uv=proj, accepted=acc, segments=table, detect_info=dict(dinfo, **guard_info), checks=checks, raster_shift_px=raster_shift,
                counts=dict(n_detected=int(len(feat)), n_chains=int(len(np.unique(chain[chain >= 0]))), n_segments_numbered=sum(1 for s in table if s["decision"] == "numbered"),
                            n_segments_unknown=sum(1 for s in table if s["decision"] == "unknown"), n_segments_no_candidate=sum(1 for s in table if s["decision"] == "no_candidate"),
                            n_ambiguous_features=int((n_cand >= 2).sum()), **counts),
                sin_alpha=dict(min=float(cand["sin_alpha"].min()) if len(feat) else None, mean=float(cand["sin_alpha"].mean()) if len(feat) else None),
                search_volume_mm=cand["search_volume_mm"], depth_margin_mm=cand["depth_margin_mm"], window_z0_mm=cand["window_z0_mm"], window_z1_mm=cand["window_z1_mm"])


def reconstruct_olines(camera_image: np.ndarray, pattern: np.ndarray, sensor_cfg: dict | None = None, method: str = "epipolar"):

    cfg = _resolve_sensor_cfg(sensor_cfg)
    spec = cfg.get("olines")
    if spec is None:
        raise ValueError("decoding 'olines' needs sensor_cfg['olines'] = metadata['olines'] of the generated pattern")
    if (method or "epipolar").lower() != "epipolar":
        raise ValueError("olines are decoded by the epipolar-intersection path only (method='epipolar')")
    r = run_olines(camera_image, spec, cfg, detector=str(cfg.get("line_detector", "rowcentroid")), pattern=pattern,
                   sg_window_rows=cfg.get("sg_window_rows"), steger_sigma_px=cfg.get("steger_sigma_px"), steger_threshold_frac=cfg.get("steger_threshold_frac"),
                   du_max_px=cfg.get("du_max_px"), vote_frac=cfg.get("vote_frac"), binarise_rule=cfg.get("binarise_rule"), noise_k=cfg.get("noise_k"),
                   guard_min_chain_px=cfg.get("guard_min_chain_px"), guard_width_window=cfg.get("guard_width_window"), guard_width_mult=cfg.get("guard_width_mult"),
                   steger_threshold_rule=cfg.get("steger_threshold_rule"), vote_valid_only=cfg.get("vote_valid_only"), steger_noise_coef=cfg.get("steger_noise_coef"), steger_noise_coef_id=cfg.get("steger_noise_coef_id"), guard_end_rows=cfg.get("guard_end_rows"))
    acc = r["accepted"]
    pts = r["points"][acc].astype(np.float32)
    info = dict(pattern_type="olines", correspondence_method="epipolar_line_intersection", line_detector=r["detector"], **r["detect_info"], **r["counts"],
                n_points_3d=int(len(pts)), n_matched=int(acc.sum()), n_rejected=int((~acc).sum()), n_ambiguous=int(r["counts"]["n_ambiguous_features"]),
                ambiguity_ratio=(float(r["counts"]["n_ambiguous_features"] / len(acc)) if len(acc) else float("nan")),
                rejection_ratio=(float((~acc).sum() / len(acc)) if len(acc) else float("nan")), search_depth_mm=r["search_volume_mm"],
                ambiguous_mask=np.zeros(int(acc.sum()), bool), line_id=r["line_id"][acc], feature_status=r["status"], features=r["features"], chain=r["chain"],
                plane_check_max_mm=r["checks"]["plane_check_max_mm"], ray_residual_max_mm=r["checks"]["ray_residual_max_mm"],
                sin_alpha_min=r["sin_alpha"]["min"], sin_alpha_mean=r["sin_alpha"]["mean"], centre_source=r["checks"]["centre_source"],
                raster_shift_mean_px=r["checks"]["raster_shift_px"]["mean"], raster_shift_max_px=r["checks"]["raster_shift_px"]["max_abs"],
                window_z0_mm=r["window_z0_mm"], window_z1_mm=r["window_z1_mm"], depth_margin_mm=r["depth_margin_mm"])
    return pts, info
