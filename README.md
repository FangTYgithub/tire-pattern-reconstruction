# Single-shot reconstruction experiments

Python 3.13. Source, digitised profiles, experiment settings, phases and seeds are included. Generated results are not bundled.

## Run

```powershell
python -m pip install -r requirements.lock.txt
python -B run.py verify
python -B run.py list
python -B run.py plan S12 --where family=point --where sigma=40
python -B run.py run S12 --where family=point --where sigma=40 --limit 2 --output outputs/demo
```

Use `--all` instead of `--limit 2` for the full selected set. Multiple `--where` filters are combined. Results are stored in `results.sqlite`; an unchanged run can resume in the same directory.

Optional outputs: `--save-images`, `--save-clouds`, `--save-details` (all numerical detection, correspondence and evaluation arrays). Use a new output directory when adding these options to an already completed run.

## Experiments

| Entry | Content | Image conditions |
|---|---|---:|
| S12 | Size, spacing and resource comparisons | 13088 |
| SIZE02 | Point radii 0.75 / 1.25 mm | 576 |
| CODE07 | Coding and same-image class controls | 6336 |
| LAYOUT02 | Ordered displacement; 4752 strategy views | 2520 |
| LAYOUT07 | Layout, phase and CLIP comparisons | 510 |
| JOINT05 | Size × spacing | 3960 |
| CH4-BLUR | Blur | 7920 |
| CH4-LIGHT | Illumination | 9504 |
| CH4-BRIDGE | Additional candidates and joint comparisons | 6600 |
| CH4-VALID | New-phase validation | 22440 |
| TR5 | Size and spacing validation | 480 |
| SUBRAY-CH2 | 24 conditions, 1×1 / 3×3 rays | 48 |
| SAMPLING-S12 | Truth sampling; no imaging | 699 |
| S12-IDENTITY / CODE-IDENTITY | Complete illuminated-line identity counts | 8096 / 3168 |
| LAYOUT-DIAG | Sampling, correct-point and reconstructed-point evaluation | 105 |
| MP05 | Environment and D0 / DF comparisons | 2450 |
| PSF-DIAG | Detection-scale and threshold controls | 392 |
| MP06C | Range, spacing and reversed-gradient comparisons | 4136 |
| MP06C-HOLDOUT / MP06E-HOLDOUT / BRIDGE-HOLDOUT | New-phase checks | 216 / 288 / 264 |
| SUBRAY-CH4 / SUBRAY-MP05 | 3×3 rays; 1×1 cases in parent experiments | 36 / 96 |

Entries overlap. S12 includes supplementary surfaces and brightness controls. Ordinary points retain D0 / DF; lines use P4. Coding and layout entries retain their registered comparison arms.

## Tables and figures

```powershell
python -B run.py export --database outputs/demo/results.sqlite --output outputs/tables
python -B run.py select-ch4 CH4-VALID --database outputs/validation/results.sqlite --output outputs/selection
python -B tools/chapter3_tables.py --data-root D:/numerical_results --output outputs/chapter3_tables
python -B plots/kap3_redraw.py --data-root D:/numerical_results --output outputs/chapter3_figures --only Abb_3_01
```

| Output | Input |
|---|---|
| Case metrics, sensitivity and diagnostics | Public-entry SQLite database |
| BLUR / LIGHT / VALID selection tables | Complete corresponding SQLite matrix |
| Tables 3.3, 3.4 and 3.13 | Result tables in the original study schema |
| Final Chapter 2–4 figures | Result tables and selected arrays in the original study schema |
| Other study summaries | Study-specific analysis modules listed in `provenance/thesis_inventory.json` |

The plot scripts read CSV/NPZ study outputs, not SQLite directly. `--data-root` contains `research/project_Part1_v1.7/` and `outputs/`; regenerate the required inputs before plotting. `--verify-reference` additionally checks archived input hashes. The reference hashes identify the thesis inputs; byte hashes may differ for independently regenerated files.

Final figure scripts: `plots/kap2_redraw.py`, `kap3_redraw.py`, `kap3b_redraw.py`, `kap4_redraw.py`. Supplementary plots are in `chapter3_figures.py` and `layout_figures.py`. All use an explicit output directory.

## Layout

- `engine/`: scientific modules and numerical inputs.
- `tools/`: public adapters and exports.
- `plots/`: data-driven figures.
- `provenance/`: experiment, figure/table and integrity mappings.
- `tests/`: numerical references and targeted checks.
- `docs/VALIDATION.json`: checks for this delivery.

Study IDs, schema versions and hashes preserve result compatibility. Public entries use the packaged inputs; older study runners also contain protocol and release interfaces. Generated outputs are excluded from submission and Git.

All packaged paths and public commands use English names. Original surface labels remain as UTF-8 metadata in the frozen surface configuration and two corresponding modules. They require no Chinese system locale. Legacy protocol-document paths are not used by public commands.
