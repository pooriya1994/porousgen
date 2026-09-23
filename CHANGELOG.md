# Changelog

All notable changes to PorousGen. Versions follow semantic versioning; every
release is tagged `vX.Y.Z` on GitHub and archived on Zenodo.

## [1.1.0] — 2026-09

Revision in response to the SoftwareX review of the v1.0 article.

### Fixed
- **Windows CI failure: `UnicodeEncodeError` on non-ASCII console output.**
  Verbose generator output (φ, →, —, box-drawing characters) crashed on
  Windows, where the default stdout/pipe encoding is the system codepage
  (e.g. cp1252), not UTF-8 — reproducible losslessly with `PYTHONIOENCODING=
  cp1252:strict`. All runtime `print()`/report output is now plain ASCII
  (`φ` → `phi`, `→` → `->`, etc.); Unicode remains only in source-level
  docstrings and comments, which are unaffected since Python reads `.py`
  files as UTF-8 regardless of platform. `write_info_file` and the metrics
  JSON writer now open their output with `encoding="utf-8"` explicitly, and
  the CLI reconfigures stdout/stderr to UTF-8 with `errors="replace"` as a
  defence-in-depth fallback for any future or third-party non-ASCII output.
- **Gyroid exact porosity.** v1.0 estimated the Gyroid threshold on a
  separate fixed-size sample (256³ / 512²) and applied it to the requested
  grid, so the fluid-voxel count could miss the target by thousands of
  voxels. The Gyroid field is now evaluated on the requested grid and
  exactly `round((1-φ)·N)` voxels are made solid by count selection.
  Discretised Gyroid fields contain large sets of *equal* values (e.g.
  1,296 voxels share the cut-off value on a 48³ grid with 3 cells), which is
  why no scalar threshold can hit the count; ties are now drawn with a
  seeded random choice.
- **Closed-cell (cellular) boundary convention.** v1.0 placed Voronoi seeds
  non-periodically but detected cell walls with `np.roll`, i.e. across
  opposite faces, which created spurious walls on the domain faces. Seeds,
  wall detection, dilation and distance field now follow one explicit
  convention (`periodic=False` or `True`).
- **Overlapping spheres.** v1.0 added spheres one by one until the porosity
  crossed the target; the overshoot could exceed one sphere (up to 1,097
  voxels in the test campaign) and each sphere cost a full-grid pass. The
  model is now a Boolean model with exact porosity (see Changed).
- **Blob (Gaussian field)** now uses the shared count-selection engine, as
  the article describes, instead of a quantile threshold.
- **STL export** wrote vertex coordinates in voxel-index units and left the
  surface open at the domain faces. Coordinates are now physical
  (`domain_size`) and the mesh is watertight by default (`closed=True`).
- Importing PorousGen no longer silences all Python warnings globally
  (v1.0 called `warnings.filterwarnings("ignore")` at import).
- `docs/catalog.png` regenerated with all eight architectures by the new,
  fully parameterised `reproducibility/catalogue.py`.

### Added
- `periodic=True/False` on granular, fibrous, cellular, open foam, blob and
  overlapping spheres. All periodic distances use the minimum-image
  convention (periodic KD-tree, wrap-padded exact EDT, periodic fibre
  images, periodic RSA overlap test). Gyroid: intrinsically periodic when
  the box holds whole unit cells; per-axis periodicity is reported and a
  warning is raised otherwise. Fractured: non-periodic only (documented).
- Exact-porosity mode for **fibrous** (`porosity=`; one effective fibre
  radius, fibre shape unchanged) and **fractured** (`porosity=`; one
  effective aperture). Seven of the eight generators now offer exact
  porosity; granular (hard-sphere RSA) remains indirect.
- `porousgen.metrics`: connectivity and percolating porosity, isolated-pore
  fraction, solid clusters, Euler characteristic, specific surface area
  (mesh and voxel estimators), local-thickness pore-size and
  strut/wall-thickness distributions, under-resolution flag.
- `GENERATOR_INFO` capability table; `return_info=True` on all generators
  (realised parameters: effective radius/aperture, particles placed,
  periodic axes, ...); `read_palabos()`.
- CLI: `--periodic`, `--metrics`, optional exact `--porosity` for fibrous
  and fractured, `--length` for fibres, `porousgen analyze <file.dat>`,
  `porousgen list`. The gyroid command now passes `--seed`.
- Info file: exact integer fluid-voxel count, realised generation details,
  explicit `.dat` format description, optional metrics section.
- Test suite (`tests/`), committed reference outputs (`tests/reference/`,
  `tools/make_reference.py`), GitHub Actions CI on Linux/Windows/macOS ×
  Python 3.9/3.12, benchmark script (`benchmarks/`), reproducibility
  material for the article (`reproducibility/`), `RELEASING.md`.

### Changed
- **Tie-breaking in count selection is spatially coherent.** Voxels at exactly
  the cut-off value form a shell of which only part is needed. v1.0 chose
  that part at random, which left isolated single voxels over every wall
  surface; on a closed-cell geometry (96³, φ = 0.85) this produced 4,711
  one-voxel protrusions and 18.5 % spurious extra wall area. Ties are now
  ranked by a smooth, seeded, low-frequency field with integer wave numbers
  (hence exactly periodic on the grid), so the extra layer appears as
  coherent patches: 67 protrusions at the same exact porosity.
  `tie_break="random"` remains available in `_core.select_k_smallest`.
- **Default boundary treatment is now non-periodic for every generator with
  a `periodic` switch.** For the blob generator this changes the default:
  v1.0 blobs were periodic (wrap-filtered). Non-periodic blobs are drawn on
  a padded box and cropped, so the field is statistically stationary up to
  the faces (mirror padding is avoided because it doubles the field
  variance at the faces).
- Overlapping spheres: the number of spheres follows the Boolean-model
  relation φ = exp(−λ·v); in non-periodic mode centres are sampled in the box
  grown by 1.5 radii so spheres centred just outside also cut into it; one
  effective radius (reported, typically within ~1 % of the nominal radius)
  makes the porosity exact.
- Memory: no generator builds an (N × ndim) float64 coordinate array any
  more (chunked KD-tree queries, broadcast coordinates, per-fibre bounding
  boxes). The `.dat` writer is vectorised (byte-identical output).

### Reproducing v1.0 geometries
| Generator | v1.1 call reproducing v1.0 |
|---|---|
| granular, fibrous (count mode), fractured (count mode) | default call — identical |
| blob | `periodic=True` — identical up to ≤1 voxel (v1.0 quantile rounding) |
| gyroid | exact count as v1.0 intended; differs only by the choice among tied voxels |
| open foam | default call — differs only by the choice among tied voxels |
| cellular | not reproducible (v1.0 mixed conventions — see Fixed) |
| overlapping spheres | not reproducible (algorithm replaced — see Fixed) |

To reproduce v1.0 exactly, install the archived v1.0.0 release.

## [1.0.0] — 2026-07-04
Initial release.
