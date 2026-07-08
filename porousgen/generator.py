#!/usr/bin/env python3
"""
================================================================================
  PorousGen — Porous Media Geometry Generator for Lattice-Boltzmann Simulation
  Eight canonical architectures · LBM-ready exports (.dat/.stl/.png/info)
================================================================================

Authors
-------
  Pooriya Ghorbani (developer)
      Ph.D. Student, Applied Multi-phase Fluid Dynamics Laboratory,
      Iran University of Science and Technology, Tehran, Iran
      Email: p_ghorbani@mecheng.iust.ac.ir

  Majid Siavashi (corresponding author)
      Associate Professor, Applied Multi-Phase Fluid Dynamics Laboratory,
      Iran University of Science and Technology, Tehran, Iran
      Email: msiavashi@iust.ac.ir
      Postal address: School of Mechanical Engineering, Iran University of
      Science and Technology, Narmak, Tehran, Iran. Postal Code: 1684613114.
      Tel: +98 21 77240391   Fax: +98 21 77240488

Geometry catalogue
------------------
  1  Granular            RSA sphere/disc packing      radius, n_particles
  2  Fibrous             random cylinders             fiber_radius, n_fibers
  3  Cellular (walls)    Voronoi face network         n_seeds, wall_thickness
                                                      or target_porosity
  4  Open-cell foam      Voronoi EDGE (strut) network target POROSITY, n_cells
  5  Consolidated        solid + planar fractures     n_fractures, frac_width
  6  Ordered (Gyroid)    TPMS implicit surface        n_cells, porosity
  7  Stochastic blobs    Gaussian random field        target POROSITY, corr_len
  8  Overlapping spheres Boolean model                target POROSITY, radius

Every generator returns a voxel grid with the PALABOS convention
    0 = fluid (pore)      1 = solid
and `export_all()` writes the full simulation-ready bundle:
    <name>.dat  +  <name>.stl  +  <name>.png  +  <name>_info.txt

Dependencies:  pip install numpy scipy matplotlib scikit-image
================================================================================
"""

__version__  = "1.0.0"
__author__   = "Pooriya Ghorbani, Majid Siavashi"
__email__    = "p_ghorbani@mecheng.iust.ac.ir"
__license__  = "MIT"
__citation__ = ("Ghorbani P., Siavashi M., PorousGen: an LBM-ready porous-media "
                "geometry generator, Applied Multi-phase Fluid Dynamics Lab, "
                "Iran University of Science and Technology (2026).")


# ─── stdlib ──────────────────────────────────────────────────────────────────
import os
import struct
import time
import warnings
warnings.filterwarnings("ignore")

# ─── third-party (required) ───────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation, label
# Force-register the 3-D ("projection='3d'") axes plugin. matplotlib enables
# 3-D plotting via a projection-registry side effect of importing this
# submodule, NOT a direct call anywhere in this file. PyInstaller's static
# import scanner can miss that registration, which then raises
# "Unknown projection: 3d" only at runtime, inside the frozen .exe, only when
# a SMALL 3-D grid is rendered (large grids fall back to 2-D cross-sections
# and never hit this path — so the bug can hide during casual testing).
# Importing it explicitly here guarantees PyInstaller bundles and registers it.
import mpl_toolkits.mplot3d  # noqa: F401

# ─── third-party (optional mesh export) ───────────────────────────────────────
try:
    from skimage.measure import marching_cubes as _mc
    _HAS_MARCHING_CUBES = True
except ImportError:
    _HAS_MARCHING_CUBES = False

# Optional drop-in replacements for richer mesh I/O
try:
    import trimesh as _trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 0  ─  GLOBAL CONSTANTS & COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════

FLUID = 0   # pore-space voxel  (PALABOS fluid)
SOLID = 1   # solid-material voxel (PALABOS solid / boundary)

# Publication-quality colormap:  dark = fluid/void · bright = solid/grain
#   Matches the look of X-ray CT micro-tomography images.
_CT_CMAP = LinearSegmentedColormap.from_list(
    "ct_scan",
    [(0, "#0a0f1e"),    # deep navy  – fluid
     (1, "#e8e8ec")],   # light grey – solid
    N=256,
)

# Dark-background figure palette
_FIG_BG    = "#0d1117"
_PANEL_BG  = "#161b22"
_TEXT_CLR  = "#e6edf3"
_ACCENT    = "#58a6ff"
_SUB_CLR   = "#8b949e"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1  ─  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_porosity(grid: np.ndarray) -> float:
    """
    Compute the porosity φ = N_fluid / N_total.

    φ = 1.0  →  completely empty domain
    φ = 0.0  →  completely filled domain
    """
    return float(np.sum(grid == FLUID)) / grid.size




def _solidify_to_porosity(dist: np.ndarray, porosity: float,
                          seed: int = 0) -> np.ndarray:
    """
    EXACT porosity by construction: mark the k = round((1-φ)·N) voxels with
    the SMALLEST distance-to-skeleton as solid. Ties (e.g. the zero-distance
    skeleton itself) are broken randomly, so targets both above and below the
    raw skeleton fraction are honoured — the skeleton is thinned or thickened
    as needed. Quantile thresholding fails on tie plateaus; this cannot.
    """
    n = dist.size
    k = int(round((1.0 - porosity) * n))
    k = min(max(k, 0), n)
    rng = np.random.default_rng(seed)
    keys = dist.ravel() + rng.uniform(0, 1e-9, n)   # random tie-break
    order = np.argpartition(keys, max(k - 1, 0))[:k]
    solid = np.zeros(n, dtype=bool)
    solid[order] = True
    return solid.reshape(dist.shape)


def _voxel_centres(shape: tuple, domain_size: tuple) -> list:
    """
    Return a list of 1-D arrays, one per spatial dimension, holding the
    physical coordinates of voxel CENTRES.

    Voxels are evenly spaced; the first centre is at Δx/2 from the boundary.
    This avoids placing solid particles/walls exactly on domain edges.

    Example (shape=(4,), domain_size=(1.0,)):
        → [0.125, 0.375, 0.625, 0.875]
    """
    return [
        np.linspace(domain_size[i] / (2 * shape[i]),
                    domain_size[i] - domain_size[i] / (2 * shape[i]),
                    shape[i])
        for i in range(len(shape))
    ]


def _meshgrid(shape: tuple, domain_size: tuple) -> list:
    """
    Build a full coordinate meshgrid (indexing='ij') for physical voxel centres.

    Returns a list of ndarray objects each of the given `shape`.
    """
    coords_1d = _voxel_centres(shape, domain_size)
    return np.meshgrid(*coords_1d, indexing="ij")


def _flat_pts(mesh_arrays: list) -> np.ndarray:
    """
    Flatten a meshgrid list into an (N_voxels, n_dim) coordinate array.
    Row k = physical (x, y, [z]) coordinates of voxel k.
    """
    return np.column_stack([m.ravel() for m in mesh_arrays])


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2  ─  PALABOS .dat EXPORT  (and optional STL mesh export)
# ══════════════════════════════════════════════════════════════════════════════

def export_palabos(
    grid:     np.ndarray,
    filename: str,
    verbose:  bool = True,
) -> None:
    """
    Write a voxel grid to a PALABOS-compatible text (.dat) file.

    PALABOS C++ reading pattern (3-D):
    ───────────────────────────────────
        for (plint iX = 0; iX < nx; ++iX) {
            geometryFile >> *slice;   // reads ny × nz integers
            copy(*slice, …, geometry, Box3D(iX,iX, 0,ny-1, 0,nz-1));
        }

    The >> operator accepts any whitespace as separator, so we write:

    2-D  (nx, ny):
        nx rows, each containing ny tab-separated integers.
        No blank lines.

    3-D  (nx, ny, nz):
        For each x-slice iX = 0 … nx-1:
            ny rows of nz tab-separated integers.
        A single blank line separates consecutive slices.
        No blank line after the final slice.

    This layout exactly matches the BidPack.dat reference file format
    used in our earlier MicroCT preprocessing pipeline.

    Parameters
    ----------
    grid     : np.ndarray  dtype int8  shape (nx,ny) or (nx,ny,nz)
    filename : str         output file path  (.dat)
    verbose  : bool        print shape, porosity, and file size
    """
    g = grid.astype(np.int8)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as fh:
        if g.ndim == 2:
            # ── 2-D layout ────────────────────────────────────────────────────
            for ix in range(g.shape[0]):
                fh.write("\t".join(map(str, g[ix, :].tolist())))
                fh.write("\n")
        else:
            # ── 3-D layout ─────────────────────────────────────────────────────
            nx, ny, nz = g.shape
            for ix in range(nx):
                for iy in range(ny):
                    fh.write("\t".join(map(str, g[ix, iy, :].tolist())))
                    fh.write("\n")
                if ix < nx - 1:
                    fh.write("\n")   # blank line between x-slices

    if verbose:
        phi  = compute_porosity(g)
        size = os.path.getsize(filename) / 1024
        print(f"  [PALABOS]  {filename!r:40s}  "
              f"shape={g.shape}  φ={phi:.4f}  ({size:.0f} KB)")


def export_stl(
    grid:     np.ndarray,
    filename: str,
    verbose:  bool = True,
) -> None:
    """
    Export a 3-D voxel grid as a binary STL surface mesh.

    Algorithm
    ---------
    1.  Marching-cubes (scikit-image) extracts the triangulated iso-surface
        at level 0.5, which sits at the fluid / solid boundary.
    2.  The triangle vertices and normals are packed into a binary STL file
        using Python's struct module — no external STL library required.

    If trimesh is installed the mesh is also cleaned (duplicate vertices,
    degenerate faces removed) before writing.

    Falls back gracefully when scikit-image is unavailable.

    Parameters
    ----------
    grid     : np.ndarray  shape (nx, ny, nz)   3-D only
    filename : str         output path  (.stl)
    """
    if not _HAS_MARCHING_CUBES:
        print("  [STL] Skipped — install scikit-image to enable mesh export.")
        return
    if grid.ndim != 3:
        print("  [STL] Skipped — STL export requires a 3-D grid.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    # ── Marching cubes ────────────────────────────────────────────────────────
    # The iso-level 0.5 sits exactly at the fluid/solid interface.
    # `step_size=1` uses every voxel; increase for coarser but faster meshes.
    verts, faces, normals, _ = _mc(
        grid.astype(np.float32), level=0.5, step_size=1, allow_degenerate=False)

    # ── Optional: clean mesh via trimesh ─────────────────────────────────────
    if _HAS_TRIMESH:
        mesh = _trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        mesh.export(filename)
        if verbose:
            print(f"  [STL-trimesh]  {filename!r}  "
                  f"({len(mesh.faces)} triangles)")
        return

    # ── Minimal binary STL writer (no extra dependencies) ────────────────────
    #   Binary STL format:
    #     80-byte ASCII header
    #     uint32  : number of triangles
    #     For each triangle (50 bytes):
    #         3 × float32 : normal vector
    #         3 × 3 float32 : vertex coordinates
    #         uint16 : attribute byte count (0)
    n_tri = len(faces)
    with open(filename, "wb") as fh:
        # 80-byte header
        fh.write(b"PALABOS voxel-to-STL export" + b" " * 53)
        fh.write(struct.pack("<I", n_tri))
        for tri_idx, face in enumerate(faces):
            # normals from marching_cubes are per-vertex; compute per-face normal
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            fn = np.cross(v1 - v0, v2 - v0)
            fn_len = np.linalg.norm(fn)
            fn = fn / fn_len if fn_len > 1e-12 else np.array([0., 0., 1.])
            fh.write(struct.pack("<fff", float(fn[0]), float(fn[1]), float(fn[2])))
            for v_idx in face:
                fh.write(struct.pack("<fff", *verts[v_idx].tolist()))
            fh.write(struct.pack("<H", 0))

    if verbose:
        size = os.path.getsize(filename) / 1024
        print(f"  [STL]  {filename!r}  ({n_tri} triangles  {size:.0f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3  ─  GEOMETRY 1 : GRANULAR  (Random Sequential Addition packing)
# ══════════════════════════════════════════════════════════════════════════════

def generate_granular(
    shape:            tuple  = (96, 96, 96),
    domain_size:      tuple  = None,
    radius:           float  = 0.08,
    radius_min:       float  = None,
    radius_max:       float  = None,
    size_distribution: str   = "uniform",
    n_particles:      int    = 70,
    overlap_tol:      float  = 0.01,
    max_attempts:     int    = 300_000,
    seed:             int    = 42,
    verbose:          bool   = True,
) -> np.ndarray:
    """
    GRANULAR POROUS MEDIUM — Random Sequential Addition (RSA) packing.
    Supports both MONODISPERSE (single radius) and POLYDISPERSE
    (a distribution of particle sizes — e.g. real GAC grain-size spread)
    packings.

    ┌────────────────────────────────────────────────────────────────────────┐
    │                      RSA Algorithm                                    │
    │                                                                        │
    │  Repeat until n_particles placed OR max_attempts exhausted:           │
    │    1. Draw a particle radius r_i:                                     │
    │         monodisperse  →  r_i = radius           (constant)            │
    │         polydisperse  →  r_i ~ Uniform(r_min, r_max)                 │
    │                       or  r_i ~ LogNormal(μ, σ), clipped to range     │
    │                          (μ, σ chosen so the range spans ±2σ in       │
    │                           log-space — realistic for sieved granular   │
    │                           media, where size spread is right-skewed)   │
    │    2. Draw a random candidate centre  c ~ Uniform(domain)             │
    │       (rejecting candidates that would place the sphere outside       │
    │        the domain walls, since r_i now varies per-particle)           │
    │    3. Compute d_min(i,j) = min distance from c to all accepted        │
    │       centres j, each with its own accepted radius r_j                │
    │    4. Accept  if  d_min ≥ (r_i + r_j)·(1 - overlap_tol)  ∀j           │
    │         (hard-sphere condition generalised to unequal radii;          │
    │          overlap_tol allows slight controlled overlap)                │
    │                                                                        │
    │  Rasterisation (per-particle bounding box — required for             │
    │  polydisperse packings; a single global KD-tree radius threshold      │
    │  is only valid when every particle shares the same radius):           │
    │    For each accepted particle (centre c_i, radius r_i):               │
    │      • Compute the voxel index bounding box covering [c_i ± r_i]      │
    │      • Build a local coordinate sub-grid only inside that box         │
    │      • Mark SOLID  where  |voxel − c_i| ≤ r_i                        │
    │    This is O(n_particles × (2r/voxel_size)^ndim), far cheaper than    │
    │    scanning the full domain per particle.                             │
    └────────────────────────────────────────────────────────────────────────┘

    Known RSA packing limits (theoretical maximum, monodisperse):
        2-D disks   → φ_solid ≈ 0.547  (Rényi parking constant)
        3-D spheres → φ_solid ≈ 0.382  (Rényi 1958)
    Polydisperse packings can exceed these limits (small particles fill
    the gaps between large ones), which is one motivation for using them
    to match a real material's measured grain-size distribution.

    Parameters
    ----------
    shape             : (nx, ny) [2-D] or (nx, ny, nz) [3-D]
    domain_size       : physical extent [m] per axis; defaults to (1.0, …)
                         For REV (Representative Elementary Volume) studies,
                         increase domain_size while holding radius / radius_min
                         / radius_max / particle numerical density fixed, and
                         track porosity until it converges — that convergence
                         point is your REV size.
    radius            : sphere / disk radius [m]  (MONODISPERSE mode; ignored
                         if radius_min and radius_max are both given)
    radius_min        : minimum particle radius [m]  (POLYDISPERSE mode)
    radius_max        : maximum particle radius [m]  (POLYDISPERSE mode)
    size_distribution : 'uniform' or 'lognormal'  (only used in polydisperse
                         mode). 'lognormal' better matches sieved/crushed
                         granular media (e.g. GAC), which are typically
                         right-skewed toward smaller particles.
    n_particles       : target number of particles to place
    overlap_tol       : fractional overlap allowed  (0 = perfect hard-sphere)
    max_attempts      : RSA trial budget before giving up
    seed              : RNG seed for reproducibility

    Returns
    -------
    np.ndarray  int8   shape = `shape`    0 = fluid   1 = solid
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)

    rng = np.random.default_rng(seed)

    polydisperse = (radius_min is not None and radius_max is not None
                     and radius_max > radius_min)

    if polydisperse:
        # Log-normal parameters chosen so [radius_min, radius_max] spans
        # roughly ±2σ around the geometric mean in log-space.
        log_lo, log_hi = np.log(radius_min), np.log(radius_max)
        mu_log    = 0.5 * (log_lo + log_hi)
        sigma_log = max((log_hi - log_lo) / 4.0, 1e-6)

    def _draw_radius() -> float:
        if not polydisperse:
            return radius
        if size_distribution == "lognormal":
            r = float(rng.lognormal(mu_log, sigma_log))
            return float(np.clip(r, radius_min, radius_max))
        return float(rng.uniform(radius_min, radius_max))   # 'uniform'

    # ── Phase 1: RSA centre + radius placement ─────────────────────────────────
    centres = []   # list of np.ndarray, physical coordinates
    radii   = []   # list of float, accepted radius per particle

    attempt = -1
    for attempt in range(max_attempts):
        r_i = _draw_radius()

        # Candidate drawn at least r_i from each wall so it fits inside
        lo = np.full(ndim, r_i)
        hi = np.array([domain_size[k] - r_i for k in range(ndim)])

        if np.any(hi <= lo):
            continue   # this particle (possibly large) doesn't fit — try another draw

        c = rng.uniform(lo, hi)

        # Hard-sphere overlap check, generalised to unequal radii
        if centres:
            arr      = np.asarray(centres)             # (N, ndim)
            rarr     = np.asarray(radii)                # (N,)
            diffs    = arr - c[np.newaxis, :]
            dists    = np.sqrt((diffs**2).sum(axis=1))
            min_gaps = (rarr + r_i) * (1.0 - overlap_tol)
            if np.any(dists < min_gaps):
                continue   # reject: overlaps an existing particle

        centres.append(c.copy())
        radii.append(r_i)
        if len(centres) >= n_particles:
            break

    if verbose:
        if polydisperse:
            print(f"  Granular (polydisperse, {size_distribution}): "
                  f"r ∈ [{radius_min:.3f}, {radius_max:.3f}] m, "
                  f"placed {len(centres)}/{n_particles} in {attempt + 1} RSA trials")
        else:
            print(f"  Granular (monodisperse): r = {radius:.3f} m, "
                  f"placed {len(centres)}/{n_particles} in {attempt + 1} RSA trials")

    # ── Phase 2: rasterise spheres onto voxel grid (per-particle bounding box) ──
    grid = np.full(shape, FLUID, dtype=np.int8)

    if centres:
        voxel_size = np.array([domain_size[k] / shape[k] for k in range(ndim)])
        coords_1d  = _voxel_centres(shape, domain_size)

        for c, r in zip(centres, radii):
            # Voxel-index bounding box covering [c - r, c + r], padded by 1
            idx_lo, idx_hi = [], []
            for k in range(ndim):
                lo_k = max(0,         int(np.floor((c[k] - r) / voxel_size[k])) - 1)
                hi_k = min(shape[k],  int(np.ceil ((c[k] + r) / voxel_size[k])) + 1)
                idx_lo.append(lo_k)
                idx_hi.append(hi_k)

            sub_coords = [coords_1d[k][idx_lo[k]:idx_hi[k]] for k in range(ndim)]
            if any(sc.size == 0 for sc in sub_coords):
                continue   # degenerate (shouldn't normally happen)

            sub_mesh = np.meshgrid(*sub_coords, indexing="ij")
            dist_sq  = sum((m - c[k])**2 for k, m in enumerate(sub_mesh))
            mask     = dist_sq <= r * r

            region_slices = tuple(slice(idx_lo[k], idx_hi[k]) for k in range(ndim))
            sub_grid = grid[region_slices]
            sub_grid[mask] = SOLID
            grid[region_slices] = sub_grid

    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4  ─  GEOMETRY 2 : FIBROUS  (random cylinder / line-segment packing)
# ══════════════════════════════════════════════════════════════════════════════

def _dist_point_to_segment(
    pts: np.ndarray,   # (N, ndim)
    p0:  np.ndarray,   # (ndim,) — segment start
    p1:  np.ndarray,   # (ndim,) — segment end
) -> np.ndarray:
    """
    Vectorised minimum distance from each row of `pts` to the line segment p0→p1.

    Mathematical derivation
    -----------------------
    Parameterise the segment:  S(t) = p0 + t·d,   d = p1 - p0,  t ∈ [0, 1].

    The unconstrained nearest-point parameter is:
        t_raw = dot(v - p0, d) / |d|²     for each voxel v.

    Clamping:  t* = clamp(t_raw, 0, 1)  ensures the nearest point
    lies on the segment (not on its infinite extension).

    The squared distance to the segment is:
        dist² = |v - (p0 + t*·d)|²

    This is O(N) with pure numpy operations — no Python loops over voxels.
    """
    d      = p1 - p0                                    # direction vector
    len_sq = float(np.dot(d, d))

    if len_sq < 1e-20:                                  # degenerate (zero-length)
        return np.linalg.norm(pts - p0, axis=1)

    # Parameter for each voxel: clamp to [0, 1] to stay on the segment
    t = np.clip(
        (pts - p0) @ d / len_sq,   # dot product with direction, shape (N,)
        0.0, 1.0
    )

    nearest = p0 + t[:, np.newaxis] * d               # (N, ndim) — closest pts
    return np.linalg.norm(pts - nearest, axis=1)       # (N,)


def generate_fibrous(
    shape:        tuple  = (96, 96, 96),
    domain_size:  tuple  = None,
    fiber_radius: float  = 0.025,
    fiber_length: float  = 0.65,
    n_fibers:     int    = 100,
    seed:         int    = 7,
    verbose:      bool   = True,
) -> np.ndarray:
    """
    FIBROUS POROUS MEDIUM — Randomly oriented cylinders (3-D) / line segments (2-D).

    ┌────────────────────────────────────────────────────────────────────────┐
    │                   Fibre Generation Algorithm                          │
    │                                                                        │
    │  For each fibre i = 1 … n_fibres:                                     │
    │    1. Sample a random centre  c ~ Uniform(domain)                     │
    │    2. Sample a random unit-direction vector  u:                        │
    │         3-D: u = N(0, I₃) / |N(0, I₃)|   ← uniform on S²            │
    │              (sampling from a 3-D Gaussian and normalising avoids     │
    │               the polar-clustering bias of naïve azimuth sampling)    │
    │         2-D: u = (cos θ, sin θ),  θ ~ Uniform[0, π)                  │
    │    3. Endpoints:  p0 = c − (L/2)·u,   p1 = c + (L/2)·u              │
    │                                                                        │
    │  Rasterisation:                                                        │
    │    For every voxel v, accumulate the minimum distance to all fibre    │
    │    axes using  _dist_point_to_segment  (O(N_vox) per fibre).         │
    │    Mark SOLID  if  min_i dist(v, fibre_i) ≤ fiber_radius             │
    │    Fibres are allowed to extend beyond the domain boundary.           │
    └────────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    fiber_radius : cross-sectional radius of each cylinder [m]
    fiber_length : full axial length of each cylinder [m]
    n_fibers     : number of fibres to place (overlaps allowed)
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)

    rng     = np.random.default_rng(seed)
    mesh    = _meshgrid(shape, domain_size)
    pts     = _flat_pts(mesh)                    # (N_vox, ndim)
    min_d   = np.full(pts.shape[0], np.inf)     # running minimum distance

    for _ in range(n_fibers):
        # ── Random fibre centre (can be anywhere in domain) ───────────────────
        c = np.array([rng.uniform(0.0, domain_size[i]) for i in range(ndim)])

        # ── Random orientation (uniform on sphere / half-circle) ──────────────
        if ndim == 2:
            theta = rng.uniform(0.0, np.pi)
            u     = np.array([np.cos(theta), np.sin(theta)])
        else:
            # Uniformly distributed on S² via Gaussian normalisation trick
            raw = rng.standard_normal(3)
            u   = raw / (np.linalg.norm(raw) + 1e-20)

        # ── Endpoints: half-length on each side of the centre ─────────────────
        half  = (fiber_length / 2.0) * u
        p0, p1 = c - half, c + half

        # ── Update per-voxel minimum distance ─────────────────────────────────
        d_seg = _dist_point_to_segment(pts, p0, p1)
        np.minimum(min_d, d_seg, out=min_d)

    grid = np.full(shape, FLUID, dtype=np.int8)
    grid.ravel()[min_d <= fiber_radius] = SOLID

    if verbose:
        phi = compute_porosity(grid)
        print(f"  Fibrous: {n_fibers} fibres placed,  φ = {phi:.4f}")

    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5  ─  GEOMETRY 3 : CELLULAR  (Voronoi-tessellation foam)
# ══════════════════════════════════════════════════════════════════════════════

def generate_cellular(
    shape:          tuple  = (96, 96, 96),
    domain_size:    tuple  = None,
    n_seeds:        int    = 35,
    wall_thickness: float  = 0.018,
    target_porosity: float = None,   # if set, overrides wall_thickness
    seed:           int    = 13,
    verbose:        bool   = True,
) -> np.ndarray:
    """
    CELLULAR POROUS MEDIUM — Voronoi-tessellation open-cell foam.

    ┌────────────────────────────────────────────────────────────────────────┐
    │                   Voronoi Foam Algorithm                              │
    │                                                                        │
    │  1. Place N random seed points in the domain.                         │
    │  2. Assign each voxel to its nearest seed:  c(v) = argmin_s |v - s|  │
    │     This partitions the domain into Voronoi cells (foam bubbles).     │
    │  3. A voxel is a WALL if any of its 2·ndim face-neighbours belongs    │
    │     to a different Voronoi cell — i.e. it sits on a Voronoi edge.     │
    │     Implemented via np.roll (O(ndim) passes over the array).          │
    │  4. Dilate the single-voxel-thin walls by `wall_thickness` using      │
    │     binary morphological dilation, creating physically realistic       │
    │     solid struts of the desired width.                                │
    │                                                                        │
    │  The Voronoi cell boundary in 3-D is a Voronoi polyhedron face.       │
    │  No explicit Voronoi computation is needed — proximity to cell         │
    │  boundaries is detected implicitly via nearest-neighbour labels.       │
    └────────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    n_seeds        : number of foam cells (bubbles / pores)
    wall_thickness : physical thickness of the solid struts / walls [m]
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)

    rng = np.random.default_rng(seed)

    # ── Step 1: Random seed points ────────────────────────────────────────────
    seeds = np.column_stack([
        rng.uniform(0.0, domain_size[i], n_seeds) for i in range(ndim)
    ])

    # ── Step 2: Voronoi cell assignment via KD-tree ────────────────────────────
    # Each voxel belongs to the Voronoi cell of its nearest seed.
    mesh       = _meshgrid(shape, domain_size)
    pts        = _flat_pts(mesh)                         # (N_vox, ndim)
    tree       = cKDTree(seeds)
    _, cell_id = tree.query(pts, k=1, workers=-1)       # (N_vox,)
    cell_grid  = cell_id.reshape(shape)                  # label per voxel

    # ── Step 3: Detect Voronoi cell boundaries ────────────────────────────────
    # A voxel is on a wall ⟺ at least one face-adjacent neighbour has a
    # different cell label.  np.roll gives periodic neighbour access.
    on_wall = np.zeros(shape, dtype=bool)
    for axis in range(ndim):
        on_wall |= np.roll(cell_grid,  1, axis=axis) != cell_grid
        on_wall |= np.roll(cell_grid, -1, axis=axis) != cell_grid

    # ── Step 4: Wall thickness — physical value OR exact porosity target ──────
    mean_vox  = np.mean([domain_size[i] / shape[i] for i in range(ndim)])
    if target_porosity is not None:
        # EXACT porosity control: thicken the one-voxel Voronoi faces by
        # thresholding the Euclidean distance-to-wall field at the quantile
        # that leaves exactly `target_porosity` of the voxels fluid.
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~on_wall)
        wall = _solidify_to_porosity(dist, target_porosity, seed=seed)
        n_dilate = -1   # marker: porosity-driven
    else:
        n_dilate  = max(1, int(round(wall_thickness / mean_vox)))
        struct    = np.ones(tuple(3 for _ in range(ndim)), dtype=bool)
        wall      = on_wall.copy()
        for _ in range(n_dilate):
            wall = binary_dilation(wall, structure=struct)

    grid = np.where(wall, SOLID, FLUID).astype(np.int8)

    if verbose:
        phi = compute_porosity(grid)
        mode = (f"target φ={target_porosity}" if target_porosity is not None
                else f"wall = {n_dilate} vox")
        print(f"  Cellular: {n_seeds} cells,  {mode},  φ = {phi:.4f}")

    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6  ─  GEOMETRY 4 : CONSOLIDATED  (solid matrix + planar fractures)
# ══════════════════════════════════════════════════════════════════════════════

def generate_consolidated(
    shape:          tuple  = (96, 96, 96),
    domain_size:    tuple  = None,
    n_fractures:    int    = 12,
    fracture_width: float  = 0.022,
    orient_bounds:  tuple  = (15.0, 75.0),
    seed:           int    = 21,
    verbose:        bool   = True,
) -> np.ndarray:
    """
    CONSOLIDATED (FRACTURED) POROUS MEDIUM — solid matrix with random fractures.

    ┌────────────────────────────────────────────────────────────────────────┐
    │              Fractured-Matrix Generation Algorithm                    │
    │                                                                        │
    │  1. Initialise all voxels as SOLID (zero-porosity rock matrix).       │
    │  2. For each fracture f = 1 … n_fractures:                            │
    │       a. Random centre  c ~ Uniform(domain)                           │
    │       b. Random unit normal  n̂  within orientation bounds:            │
    │            dip angle θ (from horizontal):                             │
    │               3-D:  θ ~ Uniform[θ_min, θ_max]                        │
    │                     azimuth φ ~ Uniform[0, 2π]                       │
    │                     n̂ = (sin θ·cos φ, sin θ·sin φ, cos θ)           │
    │               2-D:  n̂ = (sin θ, cos θ)                               │
    │       c. Signed distance of each voxel v to the fracture plane:       │
    │              dist(v) = |(v - c) · n̂|                                 │
    │       d. Mark FLUID  if  dist(v) ≤ fracture_width / 2                │
    │              (fracture aperture = fracture_width)                     │
    │                                                                        │
    │  orient_bounds = (θ_min°, θ_max°)  controls fracture orientation:     │
    │    θ ≈ 0°  →  sub-horizontal fractures                               │
    │    θ ≈ 90° →  sub-vertical  fractures                                │
    └────────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    n_fractures    : number of discrete fractures to carve
    fracture_width : full aperture of each fracture [m]  (half = width / 2)
    orient_bounds  : (min_dip°, max_dip°) dip angle range from horizontal
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)

    rng   = np.random.default_rng(seed)
    mesh  = _meshgrid(shape, domain_size)
    pts   = _flat_pts(mesh)                             # (N_vox, ndim)
    half  = fracture_width / 2.0

    th_lo = np.radians(orient_bounds[0])
    th_hi = np.radians(orient_bounds[1])

    # Begin from a fully solid matrix
    grid = np.full(shape, SOLID, dtype=np.int8)

    for _ in range(n_fractures):
        # ── Random fracture centre ────────────────────────────────────────────
        c = np.array([rng.uniform(0.0, domain_size[i]) for i in range(ndim)])

        # ── Random fracture normal (unit vector within orientation bounds) ─────
        if ndim == 2:
            theta  = rng.uniform(th_lo, th_hi)
            normal = np.array([np.sin(theta), np.cos(theta)])
        else:
            theta  = rng.uniform(th_lo, th_hi)      # dip from horizontal
            phi    = rng.uniform(0.0, 2.0 * np.pi)  # strike azimuth
            normal = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])

        normal /= np.linalg.norm(normal) + 1e-20     # ensure unit vector

        # ── Point-to-plane distance ───────────────────────────────────────────
        # For a plane with unit normal n̂ passing through centre c,
        # the signed distance from a point v is:  (v - c) · n̂
        # The fracture occupies the slab  |dist| ≤ aperture / 2
        signed_dist = (pts - c) @ normal             # (N_vox,)
        grid.ravel()[np.abs(signed_dist) <= half] = FLUID

    if verbose:
        phi = compute_porosity(grid)
        print(f"  Consolidated: {n_fractures} fractures,  φ = {phi:.4f}")

    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7  ─  GEOMETRY 5 : ORDERED — Gyroid TPMS
# ══════════════════════════════════════════════════════════════════════════════

def _gyroid_value(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, L_cell: float
) -> np.ndarray:
    """
    Evaluate the Gyroid implicit function on a 3-D coordinate meshgrid.

    The Gyroid (Schoen, 1970) is defined by the implicit equation:

        G(x, y, z) =  sin(2π x / L) · cos(2π y / L)
                    +  sin(2π y / L) · cos(2π z / L)
                    +  sin(2π z / L) · cos(2π x / L)  =  t

    where L is the unit-cell period and t is the iso-surface threshold.

    Key properties:
      • At t = 0 the surface is a true minimal surface (mean curvature = 0
        everywhere) and divides space into two congruent, interpenetrating,
        and fully percolating channel networks — φ ≈ 0.50 for each phase.
      • Increasing t thins the solid network → higher porosity (φ > 0.50).
      • Decreasing t thickens walls → lower porosity (φ < 0.50).
      • Each channel network remains fully connected for all |t| < 1.413,
        guaranteeing no isolated pores at any reachable porosity.

    Reference: Schoen, A. H. (1970). NASA Technical Note D-5541.
    """
    k  = 2.0 * np.pi / L_cell
    kX, kY, kZ = k * X, k * Y, k * Z
    return (np.sin(kX) * np.cos(kY)
          + np.sin(kY) * np.cos(kZ)
          + np.sin(kZ) * np.cos(kX))


def _find_gyroid_threshold(
    target_porosity: float,
    L_cell:          float,
    mode:            str   = "3d",
    n_sample:        int   = 256,
) -> float:
    """
    Binary-search for the Gyroid iso-surface threshold t that achieves a
    given target porosity.

    Classification:  FLUID where G < t   →   higher t = more fluid = higher φ.

    A fine regular grid is used to estimate the CDF of G values, then
    binary search finds t such that  P(G < t) ≈ target_porosity.

    mode : '3d' evaluates the full Gyroid;  '2d' uses the z = 0 cross-section.
    """
    x  = np.linspace(0, 4.0 * L_cell, n_sample, endpoint=False)

    if mode == "3d":
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        G_all   = _gyroid_value(X, Y, Z, L_cell).ravel()
    else:
        # 2-D cross-section at z = 0:  sin(kx)cos(ky) + sin(ky)·1 + 0
        X2, Y2  = np.meshgrid(x, x, indexing="ij")
        k       = 2.0 * np.pi / L_cell
        G_all   = (np.sin(k * X2) * np.cos(k * Y2)
                 + np.sin(k * Y2)).ravel()

    lo, hi = float(G_all.min()), float(G_all.max())
    for _ in range(64):                              # 64 iterations ≈ 54-bit precision
        mid = 0.5 * (lo + hi)
        if np.mean(G_all < mid) < target_porosity:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_ordered_gyroid(
    shape:           tuple  = (96, 96, 96),
    domain_size:     tuple  = None,
    n_cells:         int    = 4,
    threshold:       float  = None,
    target_porosity: float  = 0.50,
    verbose:         bool   = True,
) -> np.ndarray:
    """
    ORDERED POROUS MEDIUM — Gyroid Triply Periodic Minimal Surface (TPMS).

    ┌────────────────────────────────────────────────────────────────────────┐
    │                      Gyroid Field                                     │
    │                                                                        │
    │   G(x,y,z) = sin(2πx/L)·cos(2πy/L)                                  │
    │            + sin(2πy/L)·cos(2πz/L)                                   │
    │            + sin(2πz/L)·cos(2πx/L)                                   │
    │                                                                        │
    │   Classification:                                                     │
    │     FLUID  ⟺  G(x, y, z) < threshold  t                             │
    │     SOLID  ⟺  G(x, y, z) ≥ threshold  t                             │
    │                                                                        │
    │   If threshold is None, a binary search finds the value of t that     │
    │   achieves target_porosity for the given grid and cell count.         │
    │                                                                        │
    │   2-D variant: cross-section at z = 0                                │
    │     G_2D(x,y) = sin(kx)·cos(ky) + sin(ky)                           │
    │     (3rd term vanishes because sin(0) = 0)                           │
    └────────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    n_cells         : number of complete Gyroid unit cells along each axis
    threshold       : iso-surface level t.  None → auto from target_porosity.
    target_porosity : target void fraction  (only used when threshold is None)
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)

    # Unit-cell period: n_cells complete cycles fit in domain_size[0]
    L_cell = domain_size[0] / n_cells

    # Coordinate arrays
    coords = _voxel_centres(shape, domain_size)

    if ndim == 2:
        X, Y = np.meshgrid(*coords, indexing="ij")
        k    = 2.0 * np.pi / L_cell
        G    = np.sin(k * X) * np.cos(k * Y) + np.sin(k * Y)

        if threshold is None:
            threshold = _find_gyroid_threshold(
                target_porosity, L_cell, mode="2d", n_sample=512)
    else:
        X, Y, Z = np.meshgrid(*coords, indexing="ij")
        G       = _gyroid_value(X, Y, Z, L_cell)

        if threshold is None:
            threshold = _find_gyroid_threshold(
                target_porosity, L_cell, mode="3d", n_sample=256)

    # Classify: fluid where G < t, solid where G ≥ t
    grid = np.where(G < threshold, FLUID, SOLID).astype(np.int8)

    if verbose:
        phi = compute_porosity(grid)
        print(f"  Gyroid: L={L_cell:.4f} m,  t={threshold:.4f},  "
              f"{n_cells} cells/axis,  φ = {phi:.4f}")

    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8  ─  VISUALISATION  (publication-quality catalog figure)
# ══════════════════════════════════════════════════════════════════════════════

def _add_panel(ax, arr2d, title, subtitle="", dark=True):
    """
    Render a single 2-D array panel with consistent styling.

    Uses nearest-neighbour interpolation to keep voxel edges crisp.
    """
    bg = _PANEL_BG if dark else "white"
    ax.set_facecolor(bg)
    ax.imshow(
        arr2d.T,                 # transpose so y-axis points up
        cmap    = _CT_CMAP,
        vmin    = 0, vmax = 1,
        origin  = "lower",
        aspect  = "equal",
        interpolation = "nearest",
    )
    ax.set_title(title,    fontsize=9.5, fontweight="bold",
                 color=_TEXT_CLR, pad=3)
    if subtitle:
        ax.text(0.5, -0.06, subtitle, transform=ax.transAxes,
                fontsize=7.5, color=_SUB_CLR, ha="center")
    ax.axis("off")


def visualize_catalog(
    grids_2d:  dict,
    grids_3d:  dict,
    save_path: str  = None,
    dpi:       int  = 180,
) -> plt.Figure:
    """
    Generate a publication-quality catalog figure for all five geometry types.

    Layout (3 rows × 5 columns):
    ─────────────────────────────
      Row 0  :  2-D geometries (native 2-D grids)
      Row 1  :  3-D geometries — XY mid-plane cross-section
      Row 2  :  3-D geometries — XZ mid-plane cross-section

    Parameters
    ----------
    grids_2d  : OrderedDict  name → 2-D np.ndarray
    grids_3d  : OrderedDict  name → 3-D np.ndarray
    save_path : file path for PNG/PDF output  (None = skip saving)
    """
    names = list(grids_2d.keys())
    n     = len(names)

    # ── Figure & GridSpec ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(3.2 * n, 10.5), facecolor=_FIG_BG)
    gs  = gridspec.GridSpec(
        4, n,
        figure=fig,
        height_ratios=[0.10, 1, 1, 1],   # row 0 = header spacer
        hspace=0.55, wspace=0.06,
        left=0.02, right=0.98, top=0.93, bottom=0.04,
    )

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "Porous Media Geometry Catalog  ·  PALABOS LBM Export",
        fontsize=14, fontweight="bold", color=_TEXT_CLR,
        x=0.5, y=0.975,
    )

    # Row labels (written as figure text, not axis titles)
    row_labels = [
        ("2-D geometries",                     0.985),
        ("3-D  ·  XY cross-section  (z = mid)", 0.656),
        ("3-D  ·  XZ cross-section  (y = mid)", 0.327),
    ]
    for txt, y in row_labels:
        fig.text(0.012, y, txt, color=_ACCENT, fontsize=8.5,
                 fontweight="bold", va="top")

    # Divider lines between rows
    for y_frac in [0.965, 0.645, 0.318]:
        line = plt.Line2D([0.01, 0.99], [y_frac, y_frac],
                          transform=fig.transFigure,
                          color=_ACCENT, linewidth=0.5, alpha=0.45)
        fig.add_artist(line)

    # ── Column: each geometry type ────────────────────────────────────────────
    for col, name in enumerate(names):
        g2 = grids_2d[name]
        g3 = grids_3d[name]
        φ2 = compute_porosity(g2)
        φ3 = compute_porosity(g3)

        # Column header (geometry name) in the top spacer row
        ax_hdr = fig.add_subplot(gs[0, col])
        ax_hdr.set_facecolor(_FIG_BG)
        ax_hdr.text(0.5, 0.4, name, transform=ax_hdr.transAxes,
                    ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=_ACCENT)
        ax_hdr.axis("off")

        # ── Row 1: native 2-D geometry ────────────────────────────────────────
        ax = fig.add_subplot(gs[1, col])
        _add_panel(ax, g2, f"φ = {φ2:.3f}")

        # ── Row 2: 3-D XY cross-section ───────────────────────────────────────
        ax = fig.add_subplot(gs[2, col])
        mid_z = g3.shape[2] // 2
        _add_panel(ax, g3[:, :, mid_z], f"φ = {φ3:.3f}")

        # ── Row 3: 3-D XZ cross-section ───────────────────────────────────────
        ax = fig.add_subplot(gs[3, col])
        mid_y = g3.shape[1] // 2
        _add_panel(ax, g3[:, mid_y, :], "")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [Figure]  saved → {save_path!r}")

    return fig


def visualize_3d_voxels(
    grid:     np.ndarray,
    title:    str   = "",
    alpha:    float = 0.8,
    max_vox:  int   = 28,
) -> plt.Figure:
    """
    Render a 3-D voxel plot using matplotlib's Axes3D.voxels().

    Only feasible for small grids (≤ max_vox³ ≈ 22 000 voxels) because
    each visible voxel requires 12 triangles.  For larger grids the
    function automatically falls back to three cross-section panels.

    Parameters
    ----------
    grid    : (nx, ny, nz) int8
    title   : figure / window title
    alpha   : transparency of solid voxels
    max_vox : maximum edge length for 3-D voxel rendering
    """
    too_large = any(s > max_vox for s in grid.shape)

    if too_large:
        # ── Fallback: three cross-section panels ──────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                                 facecolor=_FIG_BG)
        fig.suptitle(f"{title}  ·  3 orthogonal cross-sections",
                     color=_TEXT_CLR, fontsize=12)
        planes = [
            (grid[:, :, grid.shape[2] // 2], "XY  (z = mid)"),
            (grid[:, grid.shape[1] // 2, :], "XZ  (y = mid)"),
            (grid[grid.shape[0] // 2, :, :], "YZ  (x = mid)"),
        ]
        for ax, (sl, lbl) in zip(axes, planes):
            ax.set_facecolor(_PANEL_BG)
            ax.imshow(sl.T, cmap=_CT_CMAP, vmin=0, vmax=1,
                      origin="lower", aspect="equal",
                      interpolation="nearest")
            ax.set_title(lbl, color=_TEXT_CLR, fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        return fig

    # ── 3-D voxel plot ────────────────────────────────────────────────────────
    try:
        fig = plt.figure(figsize=(7, 6.5), facecolor=_FIG_BG)
        ax  = fig.add_subplot(111, projection="3d")
    except Exception as e:
        # Defensive fallback: if the 3-D projection is ever unavailable in a
        # given environment (e.g. a packaging edge case), never crash the
        # whole export — degrade gracefully to the 2-D cross-section view.
        plt.close("all")
        warnings.warn(f"3-D voxel rendering unavailable ({e}); "
                      f"falling back to cross-sections.")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=_FIG_BG)
        planes = [
            (grid[:, :, grid.shape[2] // 2], "XY  (z = mid)"),
            (grid[:, grid.shape[1] // 2, :], "XZ  (y = mid)"),
            (grid[grid.shape[0] // 2, :, :], "YZ  (x = mid)"),
        ]
        for ax2, (sl, lbl) in zip(axes, planes):
            ax2.set_facecolor(_PANEL_BG)
            ax2.imshow(sl.T, cmap=_CT_CMAP, vmin=0, vmax=1,
                      origin="lower", aspect="equal", interpolation="nearest")
            ax2.set_title(lbl, color=_TEXT_CLR, fontsize=10)
            ax2.axis("off")
        plt.tight_layout()
        return fig
    ax.set_facecolor(_FIG_BG)
    fig.patch.set_facecolor(_FIG_BG)

    # Colour map: solid → steel-blue, fluid → transparent
    solid_mask = grid == SOLID
    face_col   = np.where(solid_mask, "#4d94c8", "#00000000")   # RGBA string trick
    edge_col   = np.where(solid_mask, "#1e5f8c",  "none")

    ax.voxels(solid_mask,
              facecolors=np.where(solid_mask[..., np.newaxis],
                                  np.array([0.30, 0.58, 0.78, alpha]),
                                  np.array([0., 0., 0., 0.])),
              edgecolors=np.where(solid_mask[..., np.newaxis],
                                  np.array([0.12, 0.37, 0.55, 0.6]),
                                  np.array([0., 0., 0., 0.])))

    phi = compute_porosity(grid)
    ax.set_title(f"{title}   φ = {phi:.3f}", color=_TEXT_CLR, fontsize=11,
                 fontweight="bold", pad=8)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill  = False
        pane.set_edgecolor("#2c3346")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(_SUB_CLR)
        axis._axinfo["tick"]["color"] = _SUB_CLR

    ax.tick_params(colors=_SUB_CLR, labelsize=7)
    ax.set_xlabel("x", color=_SUB_CLR, fontsize=9)
    ax.set_ylabel("y", color=_SUB_CLR, fontsize=9)
    ax.set_zlabel("z", color=_SUB_CLR, fontsize=9)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9  ─  MAIN  (demonstration of all five geometry types)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print()
    print("━" * 70)
    print("  POROUS MEDIA GEOMETRY GENERATOR  ·  PALABOS LBM Export")
    print("━" * 70)

    OUT = "porous_media_output"
    os.makedirs(OUT, exist_ok=True)

    # ── Grid sizes for the demonstration ─────────────────────────────────────
    # 2-D: 128 × 128  –  large enough to see structure
    # 3-D:  64 × 64 × 64  –  fast generation, good cross-section quality
    SH2 = (128, 128)
    SH3 = (64,  64,  64)
    DS  = (1.0, 1.0)          # 2-D domain [m]
    DS3 = (1.0, 1.0, 1.0)    # 3-D domain [m]

    g2: dict = {}   # name → 2-D grid
    g3: dict = {}   # name → 3-D grid

    t0 = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. GRANULAR  —  sphere / disk packing via RSA
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/5]  Granular  (Random Sequential Addition)")

    g2["Granular"] = generate_granular(
        shape       = SH2, domain_size = DS,
        radius      = 0.06,             # disk radius
        n_particles = 120,
        overlap_tol = 0.01,
        seed        = 42,
    )

    g3["Granular"] = generate_granular(
        shape       = SH3, domain_size = DS3,
        radius      = 0.09,             # sphere radius
        n_particles = 55,
        overlap_tol = 0.01,
        seed        = 42,
    )

    export_palabos(g2["Granular"], f"{OUT}/granular_2d.dat")
    export_palabos(g3["Granular"], f"{OUT}/granular_3d.dat")
    export_stl    (g3["Granular"], f"{OUT}/granular_3d.stl")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. FIBROUS  —  random cylinders / line-segment network
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/5]  Fibrous  (Random cylinder placement)")

    g2["Fibrous"] = generate_fibrous(
        shape        = SH2, domain_size = DS,
        fiber_radius = 0.018,
        fiber_length = 0.65,
        n_fibers     = 120,
        seed         = 7,
    )

    g3["Fibrous"] = generate_fibrous(
        shape        = SH3, domain_size = DS3,
        fiber_radius = 0.030,
        fiber_length = 0.70,
        n_fibers     = 90,
        seed         = 7,
    )

    export_palabos(g2["Fibrous"], f"{OUT}/fibrous_2d.dat")
    export_palabos(g3["Fibrous"], f"{OUT}/fibrous_3d.dat")
    export_stl    (g3["Fibrous"], f"{OUT}/fibrous_3d.stl")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. CELLULAR  —  Voronoi-tessellation foam
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/5]  Cellular  (Voronoi foam)")

    g2["Cellular"] = generate_cellular(
        shape          = SH2, domain_size = DS,
        n_seeds        = 50,
        wall_thickness = 0.015,
        seed           = 13,
    )

    g3["Cellular"] = generate_cellular(
        shape          = SH3, domain_size = DS3,
        n_seeds        = 30,
        wall_thickness = 0.025,
        seed           = 13,
    )

    export_palabos(g2["Cellular"], f"{OUT}/cellular_2d.dat")
    export_palabos(g3["Cellular"], f"{OUT}/cellular_3d.dat")
    export_stl    (g3["Cellular"], f"{OUT}/cellular_3d.stl")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. CONSOLIDATED  —  solid rock matrix + random planar fractures
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4/5]  Consolidated  (Solid matrix + fractures)")

    g2["Consolidated"] = generate_consolidated(
        shape          = SH2, domain_size = DS,
        n_fractures    = 14,
        fracture_width = 0.012,
        orient_bounds  = (10.0, 80.0),
        seed           = 21,
    )

    g3["Consolidated"] = generate_consolidated(
        shape          = SH3, domain_size = DS3,
        n_fractures    = 10,
        fracture_width = 0.020,
        orient_bounds  = (15.0, 75.0),
        seed           = 21,
    )

    export_palabos(g2["Consolidated"], f"{OUT}/consolidated_2d.dat")
    export_palabos(g3["Consolidated"], f"{OUT}/consolidated_3d.dat")
    export_stl    (g3["Consolidated"], f"{OUT}/consolidated_3d.stl")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. ORDERED  —  Gyroid TPMS (Triply Periodic Minimal Surface)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[5/5]  Ordered  (Gyroid TPMS)")

    g2["Gyroid"] = generate_ordered_gyroid(
        shape           = SH2, domain_size = DS,
        n_cells         = 4,
        target_porosity = 0.50,
    )

    g3["Gyroid"] = generate_ordered_gyroid(
        shape           = SH3, domain_size = DS3,
        n_cells         = 4,
        target_porosity = 0.50,
    )

    export_palabos(g2["Gyroid"], f"{OUT}/gyroid_2d.dat")
    export_palabos(g3["Gyroid"], f"{OUT}/gyroid_3d.dat")
    export_stl    (g3["Gyroid"], f"{OUT}/gyroid_3d.stl")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    print(f"{'━'*56}")
    print(f"  {'Geometry':<16} {'2-D φ':>10}  {'3-D φ':>10}")
    print(f"  {'─'*14}  {'─'*9}  {'─'*9}")
    for name in g2:
        p2 = compute_porosity(g2[name])
        p3 = compute_porosity(g3[name])
        print(f"  {name:<16}  {p2:>9.4f}  {p3:>9.4f}")
    print(f"{'━'*56}")
    print(f"  Total elapsed:  {elapsed:.1f} s")
    print(f"  Output folder:  '{OUT}/'")
    print(f"{'━'*56}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALISATION  1  — full 5-geometry catalog
    # ─────────────────────────────────────────────────────────────────────────
    print("  Rendering catalog figure …")
    fig_cat = visualize_catalog(
        grids_2d  = g2,
        grids_3d  = g3,
        save_path = f"{OUT}/porous_media_catalog.png",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALISATION  2  —  individual 3-D voxel renders (small crop for speed)
    # ─────────────────────────────────────────────────────────────────────────
    print("  Rendering individual 3-D voxel figures …")
    crop = 24   # render a 24³ crop so voxels() is fast
    for name, grid_3d in g3.items():
        sub = grid_3d[:crop, :crop, :crop]
        fig_v = visualize_3d_voxels(sub, title=name, max_vox=crop + 1)
        fig_v.savefig(f"{OUT}/{name.lower()}_3d_voxel.png",
                      dpi=130, bbox_inches="tight",
                      facecolor=fig_v.get_facecolor())
        print(f"    saved  {OUT}/{name.lower()}_3d_voxel.png")
        plt.close(fig_v)

    print(f"\n  Done.  Open '{OUT}/porous_media_catalog.png' to view all results.")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9  ─  GEOMETRY 6 : OPEN-CELL FOAM  (Voronoi STRUT network)
#                — porosity is the primary, EXACTLY-honoured input —
# ══════════════════════════════════════════════════════════════════════════════
def generate_open_foam(
    shape:        tuple = (96, 96, 96),
    domain_size:  tuple = None,
    porosity:     float = 0.90,
    n_cells:      int   = 6,        # foam cells per axis (fineness / ~PPI)
    seed:         int   = 42,
    verbose:      bool  = True,
) -> np.ndarray:
    """
    OPEN-CELL FOAM — the strut (EDGE) network of a Voronoi tessellation.

    Real reticulated metal/ceramic foams are open-cell: the solid forms thin
    ligaments along the EDGES of space-filling polyhedra (where 3+ Voronoi
    cells meet), leaving large connected windows. This differs from
    `generate_cellular`, whose solid follows the FACES (closed-cell walls).

    Algorithm
    ---------
    1. Scatter n_cells**ndim jittered seed points (one per foam cell).
    2. For every voxel query the 1st and 3rd nearest seeds (cKDTree).
       A voxel lies near a cell EDGE when d3 - d1 is small (3+ cells meet).
    3. Take the thinnest edge skeleton (smallest d3-d1 quantile), then thicken
       it by thresholding the Euclidean distance-to-skeleton field at the
       quantile that leaves EXACTLY `porosity` of the voxels fluid.

    Parameters
    ----------
    porosity : open volume fraction — honoured exactly (quantile threshold).
    n_cells  : cells per axis; higher = finer struts (more ligaments).
    """
    from scipy.ndimage import distance_transform_edt
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)
    rng = np.random.default_rng(seed)

    # jittered lattice of seeds -> more uniform cells than pure random
    axes = [np.linspace(0, domain_size[i], n_cells, endpoint=False)
            + domain_size[i] / (2 * n_cells) for i in range(ndim)]
    seeds = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, ndim)
    seeds += rng.uniform(-0.35, 0.35, seeds.shape) * (domain_size[0] / n_cells)

    mesh = _meshgrid(shape, domain_size)
    pts  = _flat_pts(mesh)
    d, _ = cKDTree(seeds).query(pts, k=3, workers=-1)
    edge_metric = (d[:, 2] - d[:, 0]).reshape(shape)     # small near edges

    skeleton = edge_metric <= np.quantile(edge_metric, 0.02)
    dist = distance_transform_edt(~skeleton)
    solid = _solidify_to_porosity(dist, porosity, seed=seed)

    grid = np.where(solid, SOLID, FLUID).astype(np.int8)
    if verbose:
        print(f"  Open foam: {len(seeds)} cells,  target φ={porosity:.3f}"
              f"  ->  φ = {compute_porosity(grid):.4f}")
    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 ─  GEOMETRY 7 : STOCHASTIC BLOBS  (Gaussian random field)
# ══════════════════════════════════════════════════════════════════════════════
def generate_blob(
    shape:              tuple = (96, 96, 96),
    domain_size:        tuple = None,
    porosity:           float = 0.60,
    correlation_length: float = 0.08,   # blob size, in domain units
    seed:               int   = 7,
    verbose:            bool  = True,
) -> np.ndarray:
    """
    STOCHASTIC POROUS MEDIUM — thresholded Gaussian random field ("blobs").

    The standard stochastic-reconstruction benchmark (soils, sandstones,
    membranes): white noise is smoothed with a Gaussian kernel of width
    `correlation_length`, and the field is thresholded at the quantile that
    yields EXACTLY the requested porosity. The correlation length sets the
    characteristic blob / pore size.
    """
    from scipy.ndimage import gaussian_filter
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)
    rng = np.random.default_rng(seed)

    noise = rng.standard_normal(shape)
    sigma = [correlation_length / (domain_size[i] / shape[i]) for i in range(ndim)]
    field = gaussian_filter(noise, sigma=sigma, mode="wrap")

    solid = field > np.quantile(field, porosity)         # exact porosity
    grid  = np.where(solid, SOLID, FLUID).astype(np.int8)
    if verbose:
        print(f"  Blobs: corr={correlation_length},  target φ={porosity:.3f}"
              f"  ->  φ = {compute_porosity(grid):.4f}")
    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 ─  GEOMETRY 8 : OVERLAPPING SPHERES  (Boolean model)
# ══════════════════════════════════════════════════════════════════════════════
def generate_overlapping_spheres(
    shape:        tuple = (96, 96, 96),
    domain_size:  tuple = None,
    porosity:     float = 0.55,
    radius:       float = 0.07,     # sphere radius, in domain units
    seed:         int   = 11,
    verbose:      bool  = True,
) -> np.ndarray:
    """
    BOOLEAN (overlapping-spheres) MODEL — freely inter-penetrating spheres.

    The classic "Swiss-cheese" stochastic medium: sphere centres follow a
    Poisson process and spheres may overlap, producing consolidated-looking
    solids. Spheres are added one by one until the measured porosity drops
    to the target (last sphere included only if it improves the match), so
    the requested porosity is honoured to within a fraction of one sphere.
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)
    rng   = np.random.default_rng(seed)
    mesh  = _meshgrid(shape, domain_size)
    solid = np.zeros(shape, dtype=bool)
    total = solid.size
    target_solid = 1.0 - porosity

    # Poisson estimate of how many spheres are needed:  φ = exp(-n·v/V)
    v_sph = (np.pi * radius**2 if ndim == 2
             else 4.0 / 3.0 * np.pi * radius**3)
    V     = float(np.prod(domain_size))
    n_est = max(4, int(-np.log(max(porosity, 1e-6)) * V / v_sph * 1.15))

    for i in range(n_est * 3):
        c = [rng.uniform(0, domain_size[k]) for k in range(ndim)]
        d2 = sum((mesh[k] - c[k]) ** 2 for k in range(ndim))
        new = solid | (d2 <= radius * radius)
        frac_new = new.sum() / total
        if abs(frac_new - target_solid) >= abs(solid.sum()/total - target_solid) \
           and solid.sum()/total >= target_solid * 0.98:
            break
        solid = new
        if frac_new >= target_solid:
            break

    grid = np.where(solid, SOLID, FLUID).astype(np.int8)
    if verbose:
        print(f"  Overlapping spheres: r={radius},  target φ={porosity:.3f}"
              f"  ->  φ = {compute_porosity(grid):.4f}")
    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 ─  SIMULATION-READY EXPORT BUNDLE
#                info.txt  +  .dat  +  .stl  +  .png   in one call
# ══════════════════════════════════════════════════════════════════════════════
def write_info_file(
    grid:        np.ndarray,
    filename:    str,
    domain_size: tuple = None,
    params:      dict  = None,
    dat_name:    str   = "",
) -> str:
    """Human-readable geometry report with ready-to-paste PALABOS XML blocks."""
    shape = grid.shape
    ndim  = grid.ndim
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)
    phi   = compute_porosity(grid)
    vox   = [domain_size[i] / shape[i] for i in range(ndim)]
    L = []
    A = L.append
    A("=" * 66)
    A("  PorousGen — Geometry Report")
    A(f"  generated by porous_media_generator v{__version__}")
    A("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    A("=" * 66)
    A("")
    A("  -- Grid --")
    names = ["nx", "ny", "nz"][:ndim]
    for n, s in zip(names, shape):
        A(f"  {n:<4}: {s}")
    A(f"  total voxels : {grid.size:,}")
    A(f"  porosity     : {phi:.4f}   [fluid / total]")
    A(f"  voxel size   : " + " x ".join(f"{v:.4e}" for v in vox))
    A("")
    A("  <numDomain>")
    for n, s in zip(names, shape):
        A(f"      <{n}> {s} </{n}>")
    A("  </numDomain>")
    A("")
    A("  <domain>")
    for n, d in zip(["lx", "ly", "lz"][:ndim], domain_size):
        A(f"      <{n}> {d} </{n}>")
    A("  </domain>")
    if params:
        A("")
        A("  -- Generator parameters --")
        for k, v in params.items():
            A(f"  {k:<18}: {v}")
    if dat_name:
        A("")
        A(f"  -- How to run --")
        A(f"  1. Place '{dat_name}' next to your PALABOS solver.")
        A(f"  2. Copy the <numDomain>/<domain> blocks above into the XML.")
    A("=" * 66)
    text = "\n".join(L)
    with open(filename, "w") as fh:
        fh.write(text + "\n")
    return text


def export_all(
    grid:        np.ndarray,
    basename:    str,
    domain_size: tuple = None,
    params:      dict  = None,
    stl:         bool  = True,
    png:         bool  = True,
    verbose:     bool  = True,
) -> dict:
    """
    ONE-CALL simulation-ready bundle. Writes:
        <basename>.dat        PALABOS voxel file (0 fluid / 1 solid)
        <basename>.stl        binary surface mesh          (3-D grids only)
        <basename>.png        rendered preview
        <basename>_info.txt   dimensions, porosity, XML snippets, parameters
    Returns a dict of the written paths.
    """
    out = {}
    dat = f"{basename}.dat"
    export_palabos(grid, dat, verbose=verbose)
    out["dat"] = dat

    if stl and grid.ndim == 3:
        try:
            export_stl(grid, f"{basename}.stl", verbose=verbose)
            out["stl"] = f"{basename}.stl"
        except Exception as e:
            if verbose:
                print(f"  [stl] skipped: {e}")
    elif stl and grid.ndim == 2 and verbose:
        print("  [stl] skipped: STL export requires a 3-D grid.")

    if png:
        import matplotlib
        matplotlib.use("Agg")
        if grid.ndim == 3:
            fig = visualize_3d_voxels(grid, title=os.path.basename(basename))
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(grid.T, origin="lower", cmap="gray_r",
                      interpolation="nearest")
            ax.set_title(f"{os.path.basename(basename)}  "
                         f"(φ = {compute_porosity(grid):.3f})")
            ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(f"{basename}.png", dpi=150, facecolor="white",
                    bbox_inches="tight")
        plt.close(fig)
        out["png"] = f"{basename}.png"
        if verbose:
            print(f"  [png]  {basename}.png")

    write_info_file(grid, f"{basename}_info.txt", domain_size,
                    params, dat_name=os.path.basename(dat))
    out["info"] = f"{basename}_info.txt"
    if verbose:
        print(f"  [info] {basename}_info.txt")
    return out
