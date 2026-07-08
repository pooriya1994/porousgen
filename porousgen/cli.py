"""PorousGen command-line interface.

Examples
--------
  porousgen foam    --resolution 128 --dim 3 --porosity 0.90 --cells 6 --out my_foam
  porousgen blob    --resolution 256 --dim 2 --porosity 0.60 --corr 0.08 --out soil
  porousgen gyroid  --resolution 96  --porosity 0.70 --cells 3 --out tpms
Every run writes <out>.dat, <out>.stl (3-D), <out>.png and <out>_info.txt.
"""
import argparse, sys
from . import generator as G


def main(argv=None):
    p = argparse.ArgumentParser(prog="porousgen",
        description="Generate LBM-ready porous-media geometries "
                    f"(PorousGen v{G.__version__}).")
    p.add_argument("--version", action="version", version=G.__version__)
    sub = p.add_subparsers(dest="geom", required=True)

    def common(sp, porosity=None):
        sp.add_argument("--resolution", type=int, default=96,
                        help="voxels per axis (default 96)")
        sp.add_argument("--dim", type=int, choices=(2, 3), default=3)
        sp.add_argument("--size", type=float, default=1.0,
                        help="physical domain edge length (default 1.0)")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--out", required=True, help="output basename")
        sp.add_argument("--no-stl", action="store_true")
        if porosity is not None:
            sp.add_argument("--porosity", type=float, default=porosity,
                            help=f"target porosity (default {porosity})")

    sp = sub.add_parser("granular", help="RSA sphere/disc packing")
    common(sp); sp.add_argument("--radius", type=float, default=0.05)
    sp.add_argument("--particles", type=int, default=120)

    sp = sub.add_parser("fibrous", help="random cylinders")
    common(sp); sp.add_argument("--radius", type=float, default=0.02)
    sp.add_argument("--fibers", type=int, default=120)

    sp = sub.add_parser("cellular", help="Voronoi closed-cell walls")
    common(sp, porosity=0.80); sp.add_argument("--cells", type=int, default=35,
        help="number of Voronoi seed points")

    sp = sub.add_parser("foam", help="Voronoi open-cell strut foam")
    common(sp, porosity=0.90); sp.add_argument("--cells", type=int, default=6,
        help="foam cells per axis")

    sp = sub.add_parser("fractured", help="solid matrix + planar fractures")
    common(sp); sp.add_argument("--fractures", type=int, default=8)
    sp.add_argument("--width", type=float, default=0.02)

    sp = sub.add_parser("gyroid", help="TPMS ordered structure")
    common(sp, porosity=0.70); sp.add_argument("--cells", type=int, default=3)

    sp = sub.add_parser("blob", help="Gaussian-random-field blobs")
    common(sp, porosity=0.60); sp.add_argument("--corr", type=float,
        default=0.08, help="correlation (blob) length")

    sp = sub.add_parser("spheres", help="overlapping-spheres Boolean model")
    common(sp, porosity=0.55); sp.add_argument("--radius", type=float, default=0.07)

    a = p.parse_args(argv)
    shape = tuple([a.resolution] * a.dim)
    dom   = tuple([a.size] * a.dim)
    params = dict(type=a.geom, resolution=a.resolution, dim=a.dim,
                  size=a.size, seed=a.seed)

    if a.geom == "granular":
        g = G.generate_granular(shape, dom, radius=a.radius,
                                n_particles=a.particles, seed=a.seed)
        params.update(radius=a.radius, particles=a.particles)
    elif a.geom == "fibrous":
        g = G.generate_fibrous(shape, dom, fiber_radius=a.radius,
                               n_fibers=a.fibers, seed=a.seed)
        params.update(radius=a.radius, fibers=a.fibers)
    elif a.geom == "cellular":
        g = G.generate_cellular(shape, dom, n_seeds=a.cells,
                                target_porosity=a.porosity, seed=a.seed)
        params.update(cells=a.cells, porosity=a.porosity)
    elif a.geom == "foam":
        g = G.generate_open_foam(shape, dom, porosity=a.porosity,
                                 n_cells=a.cells, seed=a.seed)
        params.update(cells=a.cells, porosity=a.porosity)
    elif a.geom == "fractured":
        g = G.generate_consolidated(shape, dom, n_fractures=a.fractures,
                                    fracture_width=a.width, seed=a.seed)
        params.update(fractures=a.fractures, width=a.width)
    elif a.geom == "gyroid":
        g = G.generate_ordered_gyroid(shape, dom, n_cells=a.cells,
                                      target_porosity=a.porosity)
        params.update(cells=a.cells, porosity=a.porosity)
    elif a.geom == "blob":
        g = G.generate_blob(shape, dom, porosity=a.porosity,
                            correlation_length=a.corr, seed=a.seed)
        params.update(corr=a.corr, porosity=a.porosity)
    elif a.geom == "spheres":
        g = G.generate_overlapping_spheres(shape, dom, porosity=a.porosity,
                                           radius=a.radius, seed=a.seed)
        params.update(radius=a.radius, porosity=a.porosity)
    else:
        p.error("unknown geometry")

    paths = G.export_all(g, a.out, dom, params, stl=not a.no_stl)
    print("\nDone. Files written:")
    for k, v in paths.items():
        print(f"  {k:<5} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
