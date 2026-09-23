"""
Regenerate the reference outputs in tests/reference/.

    python tools/make_reference.py

Writes reference_grids.npz (bit-packed voxel grids) and manifest.json
(generator call, shape, exact fluid-voxel count, SHA-256 of the grid bytes,
and the software versions that produced them). tests/test_reference.py
regenerates every case and compares against these files.
Only run this deliberately, when a change to the geometry is intended, and
record the reason in CHANGELOG.md.
"""
import hashlib
import json
import os
import platform
import sys

import numpy as np
import scipy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import porousgen as pg  # noqa: E402

CASES = {
    "granular_np":   ("generate_granular",            dict(shape=(32, 32, 32), radius=0.1, n_particles=25, seed=1)),
    "granular_per":  ("generate_granular",            dict(shape=(32, 32, 32), radius=0.1, n_particles=25, seed=1, periodic=True)),
    "fibrous_np":    ("generate_fibrous",             dict(shape=(32, 32, 32), n_fibers=20, fiber_radius=0.04, seed=1)),
    "fibrous_exact": ("generate_fibrous",             dict(shape=(32, 32, 32), n_fibers=20, porosity=0.8, seed=1, periodic=True)),
    "cellular_np":   ("generate_cellular",            dict(shape=(32, 32, 32), n_seeds=15, target_porosity=0.8, seed=1)),
    "cellular_per":  ("generate_cellular",            dict(shape=(32, 32, 32), n_seeds=15, target_porosity=0.8, seed=1, periodic=True)),
    "cellular_wall": ("generate_cellular",            dict(shape=(32, 32, 32), n_seeds=15, wall_thickness=0.04, seed=1)),
    "fractured":     ("generate_consolidated",        dict(shape=(32, 32, 32), n_fractures=6, seed=1)),
    "fractured_ex":  ("generate_consolidated",        dict(shape=(32, 32, 32), n_fractures=6, porosity=0.3, seed=1)),
    "gyroid":        ("generate_ordered_gyroid",      dict(shape=(32, 32, 32), n_cells=2, target_porosity=0.7)),
    "gyroid_2d":     ("generate_ordered_gyroid",      dict(shape=(64, 64), n_cells=3, target_porosity=0.6)),
    "foam_np":       ("generate_open_foam",           dict(shape=(32, 32, 32), porosity=0.9, n_cells=3, seed=1)),
    "foam_per":      ("generate_open_foam",           dict(shape=(32, 32, 32), porosity=0.9, n_cells=3, seed=1, periodic=True)),
    "blob_np":       ("generate_blob",                dict(shape=(32, 32, 32), porosity=0.6, correlation_length=0.08, seed=1)),
    "blob_per":      ("generate_blob",                dict(shape=(32, 32, 32), porosity=0.6, correlation_length=0.08, seed=1, periodic=True)),
    "spheres_np":    ("generate_overlapping_spheres", dict(shape=(32, 32, 32), porosity=0.55, radius=0.1, seed=1)),
    "spheres_per":   ("generate_overlapping_spheres", dict(shape=(32, 32, 32), porosity=0.55, radius=0.1, seed=1, periodic=True)),
    "foam_2d":       ("generate_open_foam",           dict(shape=(96, 96), porosity=0.85, n_cells=4, seed=1)),
}


def build(case):
    fn, kw = CASES[case]
    return getattr(pg, fn)(verbose=False, **kw)


def sha256(g):
    return hashlib.sha256(np.ascontiguousarray(g, dtype=np.int8).tobytes()).hexdigest()


if __name__ == "__main__":
    here = os.path.join(os.path.dirname(__file__), "..", "tests", "reference")
    os.makedirs(here, exist_ok=True)
    grids, manifest = {}, {}
    for name, (fn, kw) in CASES.items():
        g = build(name)
        grids[name] = np.packbits(g.astype(bool).ravel())
        manifest[name] = dict(function=fn, kwargs={k: list(v) if isinstance(v, tuple) else v
                                                   for k, v in kw.items()},
                              shape=list(g.shape), n_fluid=int((g == 0).sum()),
                              sha256=sha256(g))
        print(f"  {name:14s} {str(g.shape):14s} N_fluid={manifest[name]['n_fluid']}")
    np.savez_compressed(os.path.join(here, "reference_grids.npz"), **grids)
    meta = dict(porousgen=pg.__version__, python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform())
    with open(os.path.join(here, "manifest.json"), "w") as fh:
        json.dump(dict(generated_with=meta, cases=manifest), fh, indent=2)
    print("written", here)
