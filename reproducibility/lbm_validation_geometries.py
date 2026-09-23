"""
Geometries for the lattice-Boltzmann validation requested in review:
exact-porosity architectures generated at the SAME prescribed porosity and
used directly (no post-processing) as solver input.

    python reproducibility/lbm_validation_geometries.py \
        --porosity 0.80 --resolution 128 --size 0.002 [--periodic] [--all]

For every architecture this writes the full export bundle (.dat, .stl, .png,
_info.txt, _metrics.json) into lbm_validation/, and a table
lbm_validation/summary.md comparing exact voxel counts, percolating
porosity, specific surface and pore/strut sizes. Permeabilities computed by
the solver should be reported next to this table: at equal porosity, the
differences between architectures are carried by surface area and pore size.

Note: the closed-cell (cellular) architecture, added with --all, is sealed
by its walls - its percolating porosity is zero and it cannot carry a
through-flow. It is included only as a connectivity illustration; do not
send it to a permeability run.
"""
import argparse
import json
import os

import porousgen as pg

CORE = {
    "open_foam": lambda s, L, phi, per: pg.generate_open_foam(s, L, porosity=phi, n_cells=5, seed=1,
                                                             periodic=per, return_info=True, verbose=False),
    "gyroid":    lambda s, L, phi, per: pg.generate_ordered_gyroid(s, L, n_cells=3, target_porosity=phi,
                                                                  return_info=True, verbose=False),
    "gaussian_field": lambda s, L, phi, per: pg.generate_blob(s, L, porosity=phi, correlation_length=0.06 * L[0],
                                                              seed=1, periodic=per, return_info=True, verbose=False),
}
EXTRA = {
    "cellular":  lambda s, L, phi, per: pg.generate_cellular(s, L, n_seeds=30, target_porosity=phi, seed=1,
                                                            periodic=per, return_info=True, verbose=False),
    "overlapping_spheres": lambda s, L, phi, per: pg.generate_overlapping_spheres(
        s, L, porosity=phi, radius=0.08 * L[0], seed=1, periodic=per, return_info=True, verbose=False),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--porosity", type=float, default=0.80)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--size", type=float, default=0.002, help="cubic box edge [m]")
    ap.add_argument("--periodic", action="store_true",
                    help="periodic RVEs (for periodic lattice-Boltzmann boundaries)")
    ap.add_argument("--all", action="store_true", help="also cellular and overlapping spheres")
    ap.add_argument("--outdir", default="lbm_validation")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    shape, L = (a.resolution,) * 3, (a.size,) * 3
    gens = dict(CORE, **(EXTRA if a.all else {}))
    rows = []
    for name, fn in gens.items():
        g, info = fn(shape, L, a.porosity, a.periodic)
        params = dict(type=name, resolution=a.resolution, size=a.size,
                      porosity=a.porosity, periodic=a.periodic)
        out = pg.export_all(g, os.path.join(a.outdir, name), L, params, info=info,
                            metrics=True, verbose=False)
        m = json.load(open(out["metrics"]))
        rows.append((name, m))
        print(f"  {name:20s} N_fluid={m['n_fluid']:,}  porosity={m['porosity']:.6f}  "
              f"percolating={m['percolating_porosity']:.4f}")
    h = a.size / a.resolution
    L_ = ["| architecture | fluid voxels | porosity | percolating porosity | S_v mesh [1/m] | "
          "pore d50 [um] | strut d50 [um] | strut < 3 vox |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in rows:
        L_.append(f"| {name} | {m['n_fluid']:,} | {m['porosity']:.6f} | {m['percolating_porosity']:.4f} | "
                  f"{m['specific_surface_mesh']:.4g} | {m['pore_size']['d50'] * 1e6:.1f} | "
                  f"{m['solid_thickness']['d50'] * 1e6:.1f} | {m['solid_thickness']['fraction_below_3vox']:.2f} |")
    if any(m["percolating_porosity"] == 0 for _, m in rows):
        L_.append("")
        L_.append("Architectures with zero percolating porosity cannot carry a through-flow "
                  "(closed cells); exclude them from permeability runs.")
    L_.append("")
    L_.append(f"Grid {a.resolution}^3, voxel {h * 1e6:.2f} um, box {a.size * 1e3:.3f} mm, "
              f"target porosity {a.porosity}, periodic={a.periodic}, PorousGen {pg.__version__}.")
    open(os.path.join(a.outdir, "summary.md"), "w").write("\n".join(L_) + "\n")
    print("\n".join(L_))
