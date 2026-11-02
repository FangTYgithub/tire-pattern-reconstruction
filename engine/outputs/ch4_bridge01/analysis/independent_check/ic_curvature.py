# Measure surface curvature and the directional sampling gaps of the point layouts.

from pathlib import Path
import numpy as np
p = Path(__file__).resolve().parents[4] / 'research/project_Part1_v1.7/outputs/surface_revision/s1_real01/domain/global_domain_step0p50.npz'
z = np.load(p)
x, y = z['x'], z['y']
xs, ys = np.unique(np.round(x, 6)), np.unique(np.round(y, 6))
ix = np.searchsorted(xs, np.round(x, 6)); iy = np.searchsorted(ys, np.round(y, 6))
for name, key in [('EF', 'z_B_curv'), ('M10', 'z_A_curv'), ('M24', 'z_C_curv')]:
    Z = np.full((len(xs), len(ys)), np.nan); Z[ix, iy] = z[key]
    h = 0.5
    zxx = (Z[2:, 1:-1] - 2 * Z[1:-1, 1:-1] + Z[:-2, 1:-1]) / h ** 2
    zyy = (Z[1:-1, 2:] - 2 * Z[1:-1, 1:-1] + Z[1:-1, :-2]) / h ** 2
    ok = np.isfinite(zxx) & np.isfinite(zyy)
    ax, ay = np.abs(zxx[ok]), np.abs(zyy[ok])
    out = [f'{name}: |z_xx| median {np.median(ax):.4f} p95 {np.percentile(ax, 95):.4f} /mm; |z_yy| median {np.median(ay):.4f} p95 {np.percentile(ay, 95):.4f} /mm']
    for gap in (4, 6, 8):
        out.append(f'  gap {gap} mm: share with kappa*h^2/8 > 0.15 mm  along x {np.mean(ax * gap ** 2 / 8 > .15):.3f}, along y {np.mean(ay * gap ** 2 / 8 > .15):.3f}')
    print('\n'.join(out))
