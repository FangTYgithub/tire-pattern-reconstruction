# Shared print sizes, German labels and figure layout checks.

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.text as mtext

CM = 1 / 2.54
TEXTWIDTH_CM = 15.0
MIN_PT = 9.0


def setup():
    plt.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.default": "it",
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "legend.title_fontsize": 10, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.2, "lines.markersize": 3.5, "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100,
    })


def num(v, nd=None):
    """German number string: 0.75 -> '0,75'; nd fixes the decimals."""
    if nd is None:
        s = f"{v:g}"
    else:
        s = f"{v:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def comma_axis(ax, axis="both"):
    f = FuncFormatter(lambda x, pos: num(round(x, 6)))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


def figure(w_cm, h_cm):
    return plt.figure(figsize=(w_cm * CM, h_cm * CM))


def qa(fig, name):
    """All visible texts inside the canvas and not smaller than MIN_PT; returns a record."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    out, small, sizes = [], [], []
    # tick labels outside the view limits exist as objects but are not drawn (e.g. on log axes): skip them
    all_ticks, drawn = set(), set()
    for ax in fig.axes:
        for axis in ax._axis_map.values():
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks |= {tk.label1, tk.label2}
            for tk in axis._update_ticks():
                drawn |= {tk.label1, tk.label2}
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t in all_ticks and t not in drawn:
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width == 0 and bb.height == 0:
            continue
        sizes.append(t.get_fontsize())
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            out.append(t.get_text()[:40])
        if t.get_fontsize() < MIN_PT - 1e-6:
            small.append((t.get_text()[:30], t.get_fontsize()))
    w_cm, h_cm = fig.get_size_inches() / CM
    rec = {"figure": name, "width_cm": round(float(w_cm), 2), "height_cm": round(float(h_cm), 2),
           "min_font_pt": min(sizes) if sizes else None, "texts_outside_canvas": out, "texts_below_9pt": small}
    assert w_cm <= TEXTWIDTH_CM + 1e-6, f"{name}: wider than text width"
    assert not out, f"{name}: text outside canvas: {out}"
    assert not small, f"{name}: text below {MIN_PT} pt: {small}"
    return rec


def pdf_glyph_check(path, name, tol_pt=0.01):
    """Check that exported glyph boxes stay inside the PDF page."""
    import pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        pg = pdf.pages[0]; W, H = pg.width, pg.height
        bad = sorted({c["text"] for c in pg.chars
                      if c["x0"] < -tol_pt or c["top"] < -tol_pt or c["x1"] > W + tol_pt or c["bottom"] > H + tol_pt})
    assert not bad, f"{name}: glyphs beyond the PDF page: {bad}"


def save(fig, base, name, records):
    """Fixed-size export: PDF (vector, rasterised artists embedded) + PNG preview."""
    records.append(qa(fig, name))
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=200)
    plt.close(fig)
    pdf_glyph_check(str(base) + ".pdf", name)
