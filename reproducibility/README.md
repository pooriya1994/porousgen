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
### Solver
- Palabos version: [VERSION]
- Hardware: [HARDWARE]

### Domain and grid
- Domain: 6.6667 x 6.6667 x 6.6667 mm; grid 150x150x150 (3,375,000 voxels); dx = 44.44 um

### Geometry
- Granular: `porousgen granular [ARGS] --seed [N]` -> measured porosity 0.8952
- Fibrous:  `porousgen fibrous  [ARGS] --seed [N]` -> measured porosity 0.8765

### Flow
- Pressure-driven, two-clock scheme; deltaP = 10 [units/relationship to 778.7 Pa TBC]
- nu = 1.1086e-6 m^2/s, rho = 3.2034 kg/m^3 (carrier vapour)
- Lateral walls: solid, non-periodic

### Species transport
- Inlet: Dirichlet, C_inlet = 1589.2 mol/m^3
- Outlet: [TBC, likely zero-gradient]
- D_gas = 8.1767e-4 m^2/s, D_solid = 8.27e-5 m^2/s

### Adsorption
- Langmuir; N_max = 37015.157 mol/m^3; k_a = 2e5, k_d = 1.56197e7 [units TBC]

### Thermal
- Coupled, exothermic; wall = adiabatic
- dH_ads = [-50000 in file / -42000 elsewhere in your work -- TBC]
- Ea = 21000 J/mol; T_amb = T_ref = 297.15 K; R = 8.314 J/mol/K
- rho_fluid/cp_fluid/k_fluid = 787 / 2530 / 0.203 [confirm final]
- rho_solid/cp_solid/k_solid = 397.74 / 1000 / 0.2 [confirm final; rho_solid bed vs grain TBC]

### Numerics
- Two relaxation clocks; omega_F ~ 1.3718 (tau_F ~ 0.729), nu_lattice ~ 0.076331
- dt: [TBC -- possibly dt_flow / dt_sorption separately]

### Run length
- Max iterations: flow 10,000; sorption 2,000,001
- Convergence: relative change of <N> < 5%
- Output: stats/VTK/image every 1,000 iterations; checkpoint every 5,000
- Fig. 7 snapshot: [TBC]; total run length / wall-clock: [TBC]