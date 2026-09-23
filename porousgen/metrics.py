"""
PorousGen structural and topological metrics (v1.1).

Exact global porosity says nothing about whether a generated medium is
physically meaningful: pores may be isolated, solid may float, struts may
be one voxel thin. ``compute_metrics`` reports the quantities needed to
judge that, for any voxel grid (0 = fluid, 1 = solid):

  porosity and exact voxel counts
  connectivity  - fluid clusters; porosity of the fluid that percolates
                  from the inlet face to the outlet face along the flow axis
                  (the part an LBM flow can actually use); inlet-accessible
                  porosity; isolated (non-percolating) fluid fraction
                - solid clusters; fraction of solid in the largest cluster
                  (floating solid fragments)
  topology      - Euler characteristic of the fluid phase (and per unit
                  volume); in 3-D chi = components - tunnels + cavities
  surface area  - specific surface area S_v by two estimators:
                  "voxel" (count of exposed voxel faces; biased high by
                  ~3/2 in 3-D and 4/pi in 2-D for isotropic surfaces) and
                  "mesh" (marching-cubes / marching-squares length or area)
  sizes         - local-thickness distributions: the diameter of the
                  largest sphere (disc) that fits in the phase and covers
                  each voxel (Hildebrand & Ruegsegger 1997). Fluid phase ->
                  pore-size distribution; solid phase -> strut / wall
                  thickness distribution. Volume-weighted d10/d50/d90/mean.

All lengths are reported in physical units (domain_size) and in voxels.
Connectivity uses face adjacency by default (6-neighbour in 3-D,
4-neighbour in 2-D), the conservative choice for flow; pass
``connectivity=ndim`` for full (26 / 8) adjacency.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

FLUID, SOLID = 0, 1


# ─────────────────────────────────────────────────────────────────────────────
#  Connectivity
# ─────────────────────────────────────────────────────────────────────────────
def _structure(ndim, connectivity):
    return ndi.generate_binary_structure(ndim, connectivity)


def percolating_mask(fluid: np.ndarray, flow_axis: int = 0,
                     connectivity: int = 1) -> np.ndarray:
    """Fluid voxels in clusters that touch BOTH the inlet (index 0) and the
    outlet (index -1) faces normal to ``flow_axis``."""
    lab, n = ndi.label(fluid, structure=_structure(fluid.ndim, connectivity))
    if n == 0:
        return np.zeros_like(fluid, dtype=bool)
    inlet = np.unique(np.take(lab, 0, axis=flow_axis))
    outlet = np.unique(np.take(lab, -1, axis=flow_axis))
    through = np.intersect1d(inlet[inlet > 0], outlet[outlet > 0])
    return np.isin(lab, through)


def _connectivity_block(grid, flow_axis, connectivity):
    fluid = grid == FLUID
    solid = ~fluid
    N = grid.size
    st = _structure(grid.ndim, connectivity)
    flab, nf = ndi.label(fluid, structure=st)
    slab, ns = ndi.label(solid, structure=st)
    inlet = np.unique(np.take(flab, 0, axis=flow_axis))
    inlet = inlet[inlet > 0]
    perc = percolating_mask(fluid, flow_axis, connectivity)
    n_fluid = int(fluid.sum())
    n_solid = N - n_fluid
    largest_solid = int(np.bincount(slab.ravel())[1:].max()) if ns else 0
    return dict(
        n_fluid_clusters=int(nf),
        n_solid_clusters=int(ns),
        percolating_porosity=float(perc.sum()) / N,
        inlet_accessible_porosity=float(np.isin(flab, inlet).sum()) / N,
        isolated_fluid_fraction=(1.0 - perc.sum() / n_fluid) if n_fluid else 0.0,
        largest_solid_cluster_fraction=(largest_solid / n_solid) if n_solid else 0.0,
        percolates=bool(perc.any()),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Topology and surface area
# ─────────────────────────────────────────────────────────────────────────────
def euler_characteristic(phase: np.ndarray, connectivity: int = 1) -> int:
    """Euler characteristic of a binary phase (scikit-image)."""
    from skimage.measure import euler_number
    return int(euler_number(phase, connectivity=connectivity))


def surface_area_voxel(grid: np.ndarray, h) -> float:
    """Total exposed fluid/solid voxel-face area (interior faces only)."""
    area = 0.0
    for ax in range(grid.ndim):
        n_faces = int(np.count_nonzero(np.diff(grid, axis=ax)))
        face = float(np.prod([h[i] for i in range(grid.ndim) if i != ax]))
        area += n_faces * face
    return area


def surface_area_mesh(grid: np.ndarray, h, smooth_sigma: float = 0.0) -> float:
    """
    Interface area (3-D, marching cubes) or length (2-D, marching squares)
    through the fluid/solid voxel boundaries; domain faces excluded.

    Accuracy (validated in tests/test_metrics.py): on the unsmoothed binary
    field the estimate is ~8-10 % high for smooth surfaces resolved by
    >~10 voxels, and unreliable for features of 1-2 voxels (as is any
    voxel-based estimator). ``smooth_sigma`` > 0 applies a Gaussian
    pre-filter that removes most of the staircase bias on well-resolved
    surfaces but shrinks - and at sigma ~ 1 voxel can completely erase -
    features only 2-3 voxels thick, so it is off by default.
    """
    g = (grid == SOLID).astype(np.float32)
    if g.min() == g.max():
        return 0.0
    if smooth_sigma > 0:
        g = ndi.gaussian_filter(g, smooth_sigma)
        if g.max() < 0.5 or g.min() > 0.5:
            return 0.0
    if grid.ndim == 3:
        from skimage.measure import marching_cubes, mesh_surface_area
        v, f, _, _ = marching_cubes(g, level=0.5, spacing=tuple(h))
        return float(mesh_surface_area(v, f))
    from skimage.measure import find_contours
    total = 0.0
    for c in find_contours(g, 0.5):
        d = np.diff(c * np.asarray(h), axis=0)
        total += float(np.sqrt((d ** 2).sum(axis=1)).sum())
    return total


# ─────────────────────────────────────────────────────────────────────────────
#  Local thickness (pore- and strut-size distributions)
# ─────────────────────────────────────────────────────────────────────────────
def local_thickness(phase: np.ndarray, n_sizes: int = 25) -> np.ndarray:
    """
    Local thickness map in VOXEL units (a diameter): for each voxel of
    ``phase``, the diameter of the largest ball that lies inside the phase
    and contains the voxel. Discretised on ``n_sizes`` radii spaced
    geometrically between 1 voxel and the maximum inscribed radius.
    """
    dt = ndi.distance_transform_edt(phase)
    out = np.zeros(phase.shape, dtype=np.float32)
    rmax = float(dt.max())
    if rmax <= 0:
        return out
    radii = np.unique(np.geomspace(1.0, max(rmax, 1.0), n_sizes))[::-1]
    for r in radii:
        seeds = dt >= r
        if not seeds.any():
            continue
        covered = ndi.distance_transform_edt(~seeds) <= r
        sel = covered & phase & (out == 0)
        out[sel] = 2.0 * r
    out[phase & (out == 0)] = 2.0 * min(1.0, rmax)   # sub-voxel features
    return out


def _size_stats(lt: np.ndarray, phase: np.ndarray, h_mean: float, n_bins=20):
    vals = lt[phase].astype(float)
    if vals.size == 0:
        return None
    q = np.percentile(vals, [10, 50, 90])
    hist, edges = np.histogram(vals, bins=n_bins)
    return dict(
        fraction_below_3vox=float((vals < 3.0).mean()),
        mean_vox=float(vals.mean()), d10_vox=float(q[0]), d50_vox=float(q[1]),
        d90_vox=float(q[2]), max_vox=float(vals.max()),
        mean=float(vals.mean() * h_mean), d10=float(q[0] * h_mean),
        d50=float(q[1] * h_mean), d90=float(q[2] * h_mean),
        histogram_volume_fraction=(hist / hist.sum()).round(6).tolist(),
        histogram_edges_vox=edges.round(4).tolist(),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(grid: np.ndarray, domain_size=None, flow_axis: int = 0,
                    connectivity: int = 1, sizes: bool = True,
                    n_sizes: int = 25, surface: str = "both",
                    mesh_smoothing: float = 0.0) -> dict:
    """
    Structural / topological metrics of a voxel grid (0 fluid, 1 solid).

    Parameters
    ----------
    domain_size  : physical box size; defaults to 1.0 per axis
    flow_axis    : axis along which percolation is tested (0 = x)
    connectivity : 1 = face adjacency (default), ndim = full adjacency
    sizes        : compute local-thickness distributions (the costliest step)
    surface      : "voxel", "mesh" or "both"
    mesh_smoothing : Gaussian sigma (voxels) for the mesh estimator; 0 = off

    Size distributions are discretised on voxel-centred balls: expect an
    uncertainty of about one voxel in each reported diameter. The field
    ``fraction_below_3vox`` flags under-resolution: pore channels or struts
    only 1-2 voxels across are poorly represented by any voxel method,
    including bounce-back lattice-Boltzmann walls.
    """
    grid = np.asarray(grid)
    ndim = grid.ndim
    if domain_size is None:
        domain_size = tuple(1.0 for _ in grid.shape)
    h = [domain_size[i] / grid.shape[i] for i in range(ndim)]
    h_mean = float(np.mean(h))
    V = float(np.prod(domain_size))
    fluid = grid == FLUID
    m = dict(
        shape=list(grid.shape), domain_size=list(domain_size),
        voxel_size=h, n_voxels=int(grid.size),
        n_fluid=int(fluid.sum()), n_solid=int(grid.size - fluid.sum()),
        porosity=float(fluid.mean()), flow_axis=flow_axis,
        connectivity=connectivity,
    )
    m.update(_connectivity_block(grid, flow_axis, connectivity))
    try:
        chi = euler_characteristic(fluid, connectivity=connectivity)
        m["euler_fluid"] = chi
        m["euler_fluid_per_volume"] = chi / V
    except Exception as exc:                      # skimage missing / too old
        m["euler_fluid"] = None
        m["euler_note"] = str(exc)
    if surface in ("voxel", "both"):
        m["specific_surface_voxel"] = surface_area_voxel(grid, h) / V
    if surface in ("mesh", "both"):
        try:
            m["specific_surface_mesh"] = surface_area_mesh(grid, h, mesh_smoothing) / V
        except Exception as exc:
            m["specific_surface_mesh"] = None
            m["surface_note"] = str(exc)
    if sizes:
        m["pore_size"] = _size_stats(local_thickness(fluid, n_sizes), fluid, h_mean)
        m["solid_thickness"] = _size_stats(local_thickness(~fluid, n_sizes), ~fluid, h_mean)
    return m


def format_metrics(m: dict) -> str:
    """Human-readable block for the info file / console."""
    L = []
    A = L.append
    A("  -- Structure / topology --")
    A(f"  fluid voxels       : {m['n_fluid']:,} of {m['n_voxels']:,}")
    A(f"  porosity           : {m['porosity']:.6f}")
    ax = "xyz"[m['flow_axis']]
    A(f"  percolating porosity ({ax}-flow) : {m['percolating_porosity']:.6f}"
      f"   {'(percolates)' if m['percolates'] else '(DOES NOT PERCOLATE)'}")
    A(f"  inlet-accessible porosity      : {m['inlet_accessible_porosity']:.6f}")
    A(f"  isolated fluid fraction        : {m['isolated_fluid_fraction']:.6f}")
    A(f"  fluid / solid clusters         : {m['n_fluid_clusters']} / {m['n_solid_clusters']}")
    A(f"  solid in largest cluster       : {m['largest_solid_cluster_fraction']:.6f}")
    if m.get("euler_fluid") is not None:
        A(f"  Euler characteristic (fluid)   : {m['euler_fluid']}")
    if m.get("specific_surface_mesh") is not None:
        A(f"  specific surface, mesh  [1/L]  : {m['specific_surface_mesh']:.6g}")
    if "specific_surface_voxel" in m:
        A(f"  specific surface, voxel [1/L]  : {m['specific_surface_voxel']:.6g}"
          "   (voxel estimator, biased high)")
    for key, name in (("pore_size", "pore size (local thickness)"),
                      ("solid_thickness", "strut/wall thickness")):
        st = m.get(key)
        if st:
            A(f"  {name}: d10/d50/d90 = {st['d10']:.4g} / {st['d50']:.4g} / "
              f"{st['d90']:.4g}  (= {st['d10_vox']:.1f} / {st['d50_vox']:.1f} / "
              f"{st['d90_vox']:.1f} vox), mean {st['mean']:.4g}")
            A(f"    fraction thinner than 3 voxels: {st['fraction_below_3vox']:.4f}"
              + ("   <- under-resolved, consider a finer grid"
                 if st['fraction_below_3vox'] > 0.25 else ""))
    return "\n".join(L)
