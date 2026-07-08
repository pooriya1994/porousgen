# PorousGen

**LBM-ready porous-media geometry generator** — eight canonical architectures,
exact porosity control, and one-call export of everything a lattice-Boltzmann
simulation needs.

Developed at the **Applied Multi-phase Fluid Dynamics Laboratory**,
Iran University of Science and Technology (IUST), Tehran, Iran.

| Authors | |
|---|---|
| **Pooriya Ghorbani** (developer) | Ph.D. Student, AMFD Lab, IUST — p_ghorbani@mecheng.iust.ac.ir |
| **Majid Siavashi** (corresponding) | Associate Professor, AMFD Lab, IUST — msiavashi@iust.ac.ir |

![catalog](docs/catalog.png)

## Geometry catalogue

| # | Type | Algorithm | Porosity control |
|---|------|-----------|------------------|
| 1 | Granular | Random Sequential Addition packing | via count/radius |
| 2 | Fibrous | random cylinders | via count/radius |
| 3 | Cellular (closed-cell) | Voronoi **face** network | **exact** (`target_porosity`) |
| 4 | Open-cell foam | Voronoi **strut** network | **exact** |
| 5 | Fractured | solid matrix + planar fractures | via count/width |
| 6 | Gyroid (TPMS) | implicit surface | **exact** |
| 7 | Stochastic blobs | Gaussian random field | **exact** |
| 8 | Overlapping spheres | Boolean model | near-exact |

Every generator returns a voxel grid (`0 = fluid, 1 = solid`, the PALABOS
convention). `export_all()` writes the full simulation bundle:

```
<name>.dat        PALABOS voxel geometry
<name>.stl        binary surface mesh (3-D)
<name>.png        rendered preview
<name>_info.txt   dimensions, porosity, ready-to-paste <numDomain>/<domain> XML
```

## Install (Windows x64 / Linux / macOS)

```bash
pip install .
```

Standalone Windows executable (no Python required): see
[WINDOWS_BUILD.md](WINDOWS_BUILD.md).

## 30-second start

```bash
porousgen foam --resolution 128 --dim 3 --porosity 0.90 --cells 6 --out my_foam
porousgen blob --resolution 256 --dim 2 --porosity 0.60 --corr 0.08 --out soil
```

or from Python:

```python
from porousgen import generate_open_foam, export_all
g = generate_open_foam((128,)*3, porosity=0.90, n_cells=6)
export_all(g, "my_foam", domain_size=(2e-3,)*3)
```

A browser-based parameter builder is included at `ui/porous_media_ui.html`
(open locally; it composes the CLI command / Python snippet for you).

## Cite

See [CITATION.cff](CITATION.cff). License: MIT.
