# Reproducing the article

Everything below assumes the archived release cited in the article is
installed (`pip install porousgen==1.1.0`, or `pip install .` at tag `v1.1.0`).
`porousgen --version` must print the cited version.

## Geometry figures

| Article item | How to regenerate |
|---|---|
| Catalogue of the eight architectures (2-D and 3-D) | `python reproducibility/catalogue.py --outdir figures` |
| Open-cell foam porosity sweep (Section 3) | `bash reproducibility/porosity_sweep.sh` — runs the three commands printed in the text, then renders the figure from the resulting `.dat` files |
| Equal-porosity geometries for the lattice-Boltzmann comparison | `python reproducibility/lbm_validation_geometries.py --porosity 0.80 --resolution 128 --size 0.002` |
| Runtime and memory table | `python benchmarks/benchmark.py --sizes 64 128 256 512` |
| Exact-porosity validation | `python -m pytest tests/test_exact_porosity.py -v` |

## Lattice-Boltzmann simulations

> **TO BE COMPLETED BY THE AUTHORS BEFORE RELEASE.** Every field marked
> `[...]` must be filled with the values actually used. Nothing here may be
> left as a placeholder in the tagged release.

### Solver
- Palabos version: `[release tag or commit hash, e.g. vX.Y / abcdef0]`
- Solver source: `[path in this folder or public URL + commit]`
- Build: compiler `[...]`, MPI `[...]`, flags `[...]`
- Hardware: CPU `[model, cores]`, RAM `[GB]`, OS `[...]`, number of MPI processes `[...]`
- GPU: `[not used / device]`

### Geometry input
- For each LBM figure: the exact `porousgen` command (or Python call with
  seed) that produced its `.dat` file, e.g.
  `porousgen granular --resolution [..] --radius [..] --particles [..] --seed [..] --out [..]`
- The `_info.txt` of every geometry used (copy into `lbm/<case>/`).

### Model and parameters
- Lattice: `[D2Q9 / D3Q19 / ...]`, collision operator: `[BGK / MRT / TRT ...]`
- Relaxation time(s) `tau = [...]`; lattice viscosity `[...]`
- Physical-to-lattice conversion: dx = `[...]` m, dt = `[...]` s
- Driving: `[pressure difference / body force]` of `[...]` (lattice and physical units)
- Solid walls: `[half-way bounce-back / ...]`
- Inlet: `[...]`; outlet: `[...]`; lateral boundaries: `[periodic / walls / ...]`
- Initial condition: `[...]`
- Convergence criterion: `[e.g. relative change of mean velocity < 1e-8 over N steps]`
- Output and post-processing (how permeability / plotted fields were computed): `[...]`
- Wall-clock time per case: `[...]`

### Palabos GPU compatibility
- `.dat` files are read with the same whitespace-delimited `>>` loop in the
  CPU and GPU (accelerated) versions of Palabos: `[confirm and state the GPU
  version tested, or state that it was not tested]`.

Place the solver input files (XML parameter files), the solver source (or a
pinned link to it) and a `run.sh` per case under `reproducibility/lbm/`.
