"""
Regenerate the geometry catalogue figures of the article (2-D and 3-D
realisations of all eight architectures) and docs/catalog.png.

    python reproducibility/catalogue.py [--outdir figures]

Every parameter is written out below; with the tagged release installed the
figures are reproducible voxel for voxel.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import porousgen as pg

L2, L3 = (1.0, 1.0), (1.0, 1.0, 1.0)

CATALOGUE = [
    # label, function, kwargs for 2-D, kwargs for 3-D
    ("Granular", "generate_granular",
     dict(radius=0.035, n_particles=120, seed=42), dict(radius=0.08, n_particles=90, seed=42)),
    ("Fibrous", "generate_fibrous",
     dict(fiber_radius=0.008, fiber_length=0.6, n_fibers=70, seed=7),
     dict(fiber_radius=0.03, fiber_length=0.7, n_fibers=90, seed=7)),
    ("Cellular (closed)", "generate_cellular",
     dict(n_seeds=40, target_porosity=0.85, seed=13), dict(n_seeds=35, target_porosity=0.85, seed=13)),
    ("Fractured", "generate_consolidated",
     dict(n_fractures=10, fracture_width=0.02, seed=21), dict(n_fractures=12, fracture_width=0.03, seed=21)),
    ("Gyroid", "generate_ordered_gyroid",
     dict(n_cells=3, target_porosity=0.60), dict(n_cells=3, target_porosity=0.60)),
    ("Open-cell foam", "generate_open_foam",
     dict(porosity=0.85, n_cells=6, seed=42), dict(porosity=0.85, n_cells=5, seed=42)),
    ("Gaussian field", "generate_blob",
     dict(porosity=0.60, correlation_length=0.025, seed=7), dict(porosity=0.60, correlation_length=0.04, seed=7)),
    ("Overlapping spheres", "generate_overlapping_spheres",
     dict(porosity=0.55, radius=0.04, seed=11), dict(porosity=0.55, radius=0.07, seed=11)),
]


def build(n2=256, n3=96):
    g2, g3 = {}, {}
    for label, fn, k2, k3 in CATALOGUE:
        g2[label] = getattr(pg, fn)((n2, n2), L2, verbose=False, **k2)
        g3[label] = getattr(pg, fn)((n3,) * 3, L3, verbose=False, **k3)
    return g2, g3


def panel_figure(grids, title, path, section=None):
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.2), constrained_layout=True)
    for ax, (label, g) in zip(axes.ravel(), grids.items()):
        img = g if g.ndim == 2 else g[:, :, g.shape[2] // 2]
        ax.imshow(img.T, origin="lower", cmap="gray_r", interpolation="nearest")
        if label == "Open-cell foam" and g.ndim == 2:
            label = "Open-cell foam\n(2-D: strut cross-sections)"
        ax.set_title(f"{label}\n\u03c6 = {(g == 0).mean():.4f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    g2, g3 = build()
    panel_figure(g2, "PorousGen architectures, 2-D realisations (256\u00b2)",
                 os.path.join(a.outdir, "catalogue_2d.png"))
    panel_figure(g3, "PorousGen architectures, 3-D realisations (96\u00b3), mid-plane z-sections",
                 os.path.join(a.outdir, "catalogue_3d_sections.png"))
    docs = os.path.join(os.path.dirname(__file__), "..", "docs")
    os.makedirs(docs, exist_ok=True)
    panel_figure(g2, "PorousGen v" + pg.__version__ + " - the eight architectures (2-D)",
                 os.path.join(docs, "catalog.png"))
    with open(os.path.join(a.outdir, "catalogue_porosities.txt"), "w") as fh:
        for d, grids in (("2-D", g2), ("3-D", g3)):
            for label, g in grids.items():
                fh.write(f"{d}  {label:<22} shape={g.shape}  N_fluid={(g == 0).sum()}  "
                         f"porosity={(g == 0).mean():.6f}\n")
    print("written", a.outdir, "and docs/catalog.png")
