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

## Lattice-Boltzmann simulations (Figs. 6–7)

> Two items below are still genuinely open (marked `[TBC]`) — the specific
> `.dat`-generating commands/seeds for the granular and fibrous geometries,
> and the Fig. 7 snapshot time / total run length / wall-clock time. Fill
> these in, or soften the corresponding manuscript sentences to avoid
> needing exact figures, before the tagged release.

### Solver
- Palabos version: v2.3.0 (latest tagged release; https://gitlab.com/unigespc/palabos —
  development has continued on the `master` branch since June 2021 without a
  newer version tag)
- Hardware: AMD Ryzen 7 7435HS (8 cores / 16 threads, 3.1–4.5 GHz), 40 GB DDR5
  RAM, single workstation. NVIDIA GeForce RTX 2050 present but not used for
  the computation (Palabos is CPU/MPI-based).
- Build: compiler `[...]`, MPI `[...]`, flags `[...]`
- Number of MPI processes: `[...]`

### Domain and grid
- Domain: 6.6667 × 6.6667 × 6.6667 mm; grid 150×150×150 (3,375,000 voxels); dx = 44.44 μm

### Geometry input
- Granular: `porousgen granular [ARGS] --seed [N]` → measured porosity 0.8952 **[TBC — exact args/seed]**
- Fibrous:  `porousgen fibrous  [ARGS] --seed [N]` → measured porosity 0.8765 **[TBC — exact args/seed]**
- The `_info.txt` of each geometry used should be copied into `lbm/<case>/`.

### Flow
- Pressure-driven, two-clock scheme; ΔP = 778.7 Pa (physical); the raw
  `deltaP = 10` field in the parameter XML is the two-clock scheme's internal
  input representation, not the physical pressure.
- ν = 1.1086×10⁻⁶ m²/s, ρ = 3.2034 kg/m³ (carrier vapour)
- Lateral walls: solid, non-periodic

### Species transport
- Inlet: Dirichlet, C_inlet = 1589.2 mol/m³
- Outlet: zero-gradient
- D_gas = 8.1767×10⁻⁴ m²/s, D_solid = 8.27×10⁻⁵ m²/s

### Adsorption
- Langmuir isotherm; N_max = 37,015.157 mol/m³
- k_a = 2×10⁵, k_d = 1.56197×10⁷ **[TBC — units/rate-law form]**

### Thermal
- Coupled, exothermic; wall thermal BC = adiabatic (insulated)
- ΔH_ads = −42,000 J/mol (confirmed; consistent with Ea via ΔH/2, and with
  the GAC/copper-foam paper's value — supersedes −50,000 in an earlier
  parameter-file draft)
- Ea (ads. and des.) = 21,000 J/mol
- T_amb = T_ref = 297.15 K; R = 8.314 J/mol/K
- ρ_fluid, cp_fluid, k_fluid = 787 kg/m³, 2530 J/kg/K, 0.203 W/m/K
- ρ_solid, cp_solid, k_solid = 397.74 kg/m³, 1000 J/kg/K, 0.2 W/m/K
  (ρ_solid is bed/bulk density; k_solid is an interim value pending a full
  literature check — both confirmed as the values actually used)

### Numerics
- Two relaxation clocks (flow, sorption), auto-scaled diffusion
- ω_F (flow relaxation) ≈ 1.3718 (τ_F ≈ 0.729), ν_lattice ≈ 0.076331
  (from the solver's own startup banner)
- dt = 4.026×10⁻⁷ s

### Run length and convergence
- Max iterations: flow 10,000; sorption 2,000,001
- Convergence criterion: relative change of ⟨N⟩ < 5%
- Output cadence: statistics, VTK and image every 1,000 iterations; checkpoint every 5,000
- Fig. 7 snapshot (iteration/physical time): **[TBC]**
- Total run length / wall-clock time: **[TBC]**

### Palabos GPU compatibility
`.dat` files are read with the same whitespace-delimited `>>` loop whether the
downstream code builds a CPU `MultiBlockLattice` or a GPU-accelerated
`AcceleratedLattice` — the two containers share the same user-facing
interface (Latt & Coreixas, *Multi-GPU acceleration of PALABOS fluid solver
using C++ standard parallelism*, Computer Physics Communications 323 (2026)
110094), and that paper's own validation includes a comparable porous-media
case (Berea sandstone, a 400³ voxel digital rock). All runs reported in this
article were executed on CPU; GPU execution has not been tested directly by
us, and compatibility here is inferred from the shared interface and the
cited paper's own benchmark, not independently verified.

---

Place the solver input files (XML parameter files), the solver source (or a
pinned link to it) and a `run.sh` per case under `reproducibility/lbm/`.
