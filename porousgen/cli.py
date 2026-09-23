"""PorousGen command-line interface.

Examples
--------
  porousgen foam     --resolution 128 --porosity 0.90 --cells 6 --out my_foam
  porousgen foam     --resolution 128 --porosity 0.90 --periodic --metrics --out rve
  porousgen fibrous  --resolution 96  --fibers 80 --porosity 0.85 --out mat
  porousgen blob     --resolution 256 --dim 2 --porosity 0.60 --corr 0.08 --out soil
  porousgen gyroid   --resolution 96  --porosity 0.70 --cells 3 --out tpms
  porousgen analyze  my_foam.dat --size 0.01          # metrics of any .dat file
  porousgen list                                      # capability table

Every generation run writes <out>.dat, <out>.stl (3-D), <out>.png and
<out>_info.txt; --metrics adds <out>_metrics.json and a metrics section.
The command is lower-case `porousgen` (case-sensitive on Linux/macOS).
If the console script is not on PATH, `python -m porousgen.cli ...` works
identically.
"""
import argparse
import json
import sys

from . import generator as G

# Windows does not default stdout/stderr to UTF-8 (the console/pipe encoding
# follows the system codepage). All PorousGen output is ASCII by design (see
# CHANGELOG v1.1.0), so this is defence in depth against any future or
# third-party message that isn't: replace instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _table() -> str:
    rows = ["  generator             porosity   periodic",
            "  --------------------  ---------  ---------"]
    for name, v in G.GENERATOR_INFO.items():
        rows.append(f"  {name:<20}  {v['porosity']:<9}  {v['periodic']}")
    rows += ["",
             "  porosity: exact = always exact; optional = exact when --porosity",
             "            is given; indirect = set by counts/radii",
             "  periodic: switch = --periodic flag (default off); intrinsic =",
             "            periodic when the box holds whole unit cells; no = never"]
    return "\n".join(rows)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="porousgen",
        description=f"Generate LBM-ready porous-media geometries (PorousGen v{G.__version__}).")
    p.add_argument("--version", action="version", version=G.__version__)
    sub = p.add_subparsers(dest="geom", required=True)

    def common(sp, porosity=None, optional_porosity=False, periodic=True):
        sp.add_argument("--resolution", type=int, default=96,
                        help="voxels per axis (default 96)")
        sp.add_argument("--dim", type=int, choices=(2, 3), default=3)
        sp.add_argument("--size", type=float, default=1.0,
                        help="physical domain edge length (default 1.0)")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--out", required=True, help="output basename")
        sp.add_argument("--no-stl", action="store_true")
        sp.add_argument("--metrics", action="store_true",
                        help="compute structural/topological metrics")
        if periodic:
            sp.add_argument("--periodic", action="store_true",
                            help="periodic (tileable) geometry")
        if porosity is not None:
            sp.add_argument("--porosity", type=float, default=porosity,
                            help=f"target porosity, exact (default {porosity})")
        elif optional_porosity:
            sp.add_argument("--porosity", type=float, default=None,
                            help="if given, porosity is honoured exactly")

    sp = sub.add_parser("granular", help="RSA sphere/disc packing")
    common(sp); sp.add_argument("--radius", type=float, default=0.05)
    sp.add_argument("--particles", type=int, default=120)

    sp = sub.add_parser("fibrous", help="random capsule fibres")
    common(sp, optional_porosity=True)
    sp.add_argument("--radius", type=float, default=0.02)
    sp.add_argument("--length", type=float, default=0.65)
    sp.add_argument("--fibers", type=int, default=120)

    sp = sub.add_parser("cellular", help="Voronoi closed-cell walls")
    common(sp, porosity=0.80); sp.add_argument("--cells", type=int, default=35,
        help="number of Voronoi seed points")

    sp = sub.add_parser("foam", help="Voronoi open-cell strut foam")
    common(sp, porosity=0.90); sp.add_argument("--cells", type=int, default=6,
        help="foam cells per axis")

    sp = sub.add_parser("fractured", help="solid matrix + planar fractures")
    common(sp, optional_porosity=True, periodic=False)
    sp.add_argument("--fractures", type=int, default=8)
    sp.add_argument("--width", type=float, default=0.02)

    sp = sub.add_parser("gyroid", help="TPMS ordered structure (intrinsically periodic)")
    common(sp, porosity=0.70, periodic=False)
    sp.add_argument("--cells", type=int, default=3)

    sp = sub.add_parser("blob", help="Gaussian-random-field blobs")
    common(sp, porosity=0.60); sp.add_argument("--corr", type=float,
        default=0.08, help="correlation (blob) length")

    sp = sub.add_parser("spheres", help="overlapping-spheres Boolean model")
    common(sp, porosity=0.55); sp.add_argument("--radius", type=float, default=0.07)

    sp = sub.add_parser("analyze", help="metrics of an existing .dat voxel file")
    sp.add_argument("file")
    sp.add_argument("--size", type=float, nargs="+", default=[1.0],
                    help="physical box size: one value (cube) or one per axis")
    sp.add_argument("--flow-axis", type=int, default=0)
    sp.add_argument("--json", default=None, help="also write metrics as JSON")

    sub.add_parser("list", help="print which generators are exact / periodic")

    a = p.parse_args(argv)

    if a.geom == "list":
        print(_table())
        return 0

    if a.geom == "analyze":
        from .metrics import compute_metrics, format_metrics
        g = G.read_palabos(a.file)
        size = a.size * g.ndim if len(a.size) == 1 else a.size
        m = compute_metrics(g, domain_size=tuple(size), flow_axis=a.flow_axis)
        print(f"{a.file}: shape {g.shape}")
        print(format_metrics(m))
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(m, fh, indent=2)
            print(f"\nwritten {a.json}")
        return 0

    shape = tuple([a.resolution] * a.dim)
    dom = tuple([a.size] * a.dim)
    per = getattr(a, "periodic", False)
    params = dict(type=a.geom, resolution=a.resolution, dim=a.dim,
                  size=a.size, seed=a.seed)
    if hasattr(a, "periodic"):
        params["periodic"] = per
    kw = dict(seed=a.seed, return_info=True)

    if a.geom == "granular":
        g, info = G.generate_granular(shape, dom, radius=a.radius,
                                      n_particles=a.particles, periodic=per, **kw)
        params.update(radius=a.radius, particles=a.particles)
    elif a.geom == "fibrous":
        g, info = G.generate_fibrous(shape, dom, fiber_radius=a.radius,
                                     fiber_length=a.length, n_fibers=a.fibers,
                                     porosity=a.porosity, periodic=per, **kw)
        params.update(radius=a.radius, length=a.length, fibers=a.fibers,
                      porosity=a.porosity)
    elif a.geom == "cellular":
        g, info = G.generate_cellular(shape, dom, n_seeds=a.cells,
                                      target_porosity=a.porosity, periodic=per, **kw)
        params.update(cells=a.cells, porosity=a.porosity)
    elif a.geom == "foam":
        g, info = G.generate_open_foam(shape, dom, porosity=a.porosity,
                                       n_cells=a.cells, periodic=per, **kw)
        params.update(cells=a.cells, porosity=a.porosity)
    elif a.geom == "fractured":
        g, info = G.generate_consolidated(shape, dom, n_fractures=a.fractures,
                                          fracture_width=a.width,
                                          porosity=a.porosity, **kw)
        params.update(fractures=a.fractures, width=a.width, porosity=a.porosity)
    elif a.geom == "gyroid":
        g, info = G.generate_ordered_gyroid(shape, dom, n_cells=a.cells,
                                            target_porosity=a.porosity, **kw)
        params.update(cells=a.cells, porosity=a.porosity)
    elif a.geom == "blob":
        g, info = G.generate_blob(shape, dom, porosity=a.porosity,
                                  correlation_length=a.corr, periodic=per, **kw)
        params.update(corr=a.corr, porosity=a.porosity)
    elif a.geom == "spheres":
        g, info = G.generate_overlapping_spheres(shape, dom, porosity=a.porosity,
                                                 radius=a.radius, periodic=per, **kw)
        params.update(radius=a.radius, porosity=a.porosity)
    else:
        p.error("unknown geometry")

    paths = G.export_all(g, a.out, dom, params, stl=not a.no_stl,
                         info=info, metrics=a.metrics)
    print("\nDone. Files written:")
    for k, v in paths.items():
        print(f"  {k:<7} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
