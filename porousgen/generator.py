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

__version__  = "1.1.0"
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

# ─── third-party (required) ───────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from . import _core as C
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

# ──────────────────────────────────────────────────────────────────────────────
#  Capability table (single source of truth for docs, CLI and tests)
# ──────────────────────────────────────────────────────────────────────────────
#  porosity : "exact"    - count selection, exactly round(phi*N) fluid voxels
#             "optional" - exact when a porosity is passed, else geometric
#             "indirect" - set by counts/radii only; realised value reported
#  periodic : "switch"    - `periodic=True/False` (default False)
#             "intrinsic" - periodic whenever the box holds whole unit cells
#             "no"        - non-periodic only
GENERATOR_INFO = {
    "granular":            dict(function="generate_granular",            porosity="indirect", periodic="switch"),
    "fibrous":             dict(function="generate_fibrous",             porosity="optional", periodic="switch"),
    "cellular":            dict(function="generate_cellular",            porosity="optional", periodic="switch"),
    "consolidated":        dict(function="generate_consolidated",        porosity="optional", periodic="no"),
    "gyroid":              dict(function="generate_ordered_gyroid",      porosity="exact",    periodic="intrinsic"),
    "open_foam":           dict(function="generate_open_foam",           porosity="exact",    periodic="switch"),
    "blob":                dict(function="generate_blob",                porosity="exact",    periodic="switch"),
    "overlapping_spheres": dict(function="generate_overlapping_spheres", porosity="exact",    periodic="switch"),
}


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
    EXACT porosity by construction: mark exactly k = round((1-phi)*N) voxels
    with the SMALLEST values of `dist` as solid (see _core.select_k_smallest).
    Ties at the cut-off value are resolved by a seeded random draw from the
    tie set, so the count is exact for any field, including fields with
    large plateaus of equal values where a scalar threshold cannot hit it.
    """
    return C.select_k_smallest(dist, C.solid_count(porosity, dist.size), seed=seed)


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

    if g.min() < 0 or g.max() > 9:
        raise ValueError("export_palabos writes single-digit voxel labels (0..9)")

    # Vectorised writer: each row "d\td\t...\td\n" is exactly 2*ncols bytes,
    # so a whole slice is assembled as one uint8 array and written at once.
    # Output is byte-identical to the v1.0 row-by-row writer.
    def _rows_bytes(block):                     # block: (nrows, ncols) ints
        nrows, ncols = block.shape
        buf = np.empty((nrows, 2 * ncols), dtype=np.uint8)
        buf[:, 0::2] = block.astype(np.uint8) + ord("0")
        buf[:, 1::2] = ord("\t")
        buf[:, -1] = ord("\n")
        return buf.tobytes()

    with open(filename, "wb") as fh:
        if g.ndim == 2:
            fh.write(_rows_bytes(g))
        else:
            nx = g.shape[0]
            for ix in range(nx):
                fh.write(_rows_bytes(g[ix]))
                if ix < nx - 1:
                    fh.write(b"\n")            # blank line between x-slices

    if verbose:
        phi  = compute_porosity(g)
        size = os.path.getsize(filename) / 1024
        print(f"  [PALABOS]  {filename!r:40s}  "
              f"shape={g.shape}  φ={phi:.4f}  ({size:.0f} KB)")


def read_palabos(filename: str, shape: tuple = None) -> np.ndarray:
    """
    Read a PorousGen / PALABOS text voxel file (.dat) back into an int8 grid.

    The shape is inferred from the layout written by export_palabos:
    one block of rows (2-D: nx rows of ny values), or nx blocks of ny rows
    of nz values separated by blank lines (3-D). A 3-D grid with nx = 1 has
    no separator and is indistinguishable from 2-D; pass `shape` to resolve
    it (the <numDomain> block of the info file records the shape).
    """
    with open(filename, "r", encoding="utf-8") as fh:
        text = fh.read().strip("\n")
    blocks = [b for b in text.split("\n\n") if b.strip()]
    first = blocks[0].split("\n")
    rows, cols = len(first), len(first[0].split())
    data = np.array(text.split(), dtype=np.int8)
    if shape is not None:
        return data.reshape(shape)
    if len(blocks) == 1:
        return data.reshape(rows, cols)
    return data.reshape(len(blocks), rows, cols)


def export_stl(
    grid:        np.ndarray,
    filename:    str,
    verbose:     bool  = True,
    domain_size: tuple = None,
    closed:      bool  = True,
) -> None:
    """
    Export the SOLID phase of a 3-D voxel grid as a binary STL surface mesh.

    Algorithm
    ---------
    1. Marching cubes (scikit-image) on the solid indicator at iso-level 0.5,
       i.e. through the fluid/solid voxel interfaces.
    2. Coordinates are written in PHYSICAL units, consistent with the .dat
       file and domain_size: voxel i is centred at (i + 1/2) * dx.
    3. closed=True (default) pads the grid with one layer of fluid before
       meshing, so the surface is also capped where solid meets a domain
       face and the mesh is WATERTIGHT (every edge shared by exactly two
       triangles). closed=False reproduces the v1.0 open surface.
    4. The binary STL is written in one vectorised call.

    Scope
    -----
    The mesh reproduces the voxel geometry, including its voxel-scale
    staircase; it is intended for visualisation, 3-D printing and as input
    to a surface-meshing tool. It is NOT a smoothed, body-fitted CFD mesh -
    the .dat voxel file is the simulation input for lattice-Boltzmann codes.
    """
    if not _HAS_MARCHING_CUBES:
        print("  [STL] Skipped — install scikit-image to enable mesh export.")
        return
    if grid.ndim != 3:
        print("  [STL] Skipped — STL export requires a 3-D grid.")
        return
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    if domain_size is None:
        h = np.ones(3)
    else:
        h = np.array([domain_size[i] / grid.shape[i] for i in range(3)])
    solid = (grid == SOLID).astype(np.float32)
    pad = 1 if closed else 0
    if closed:
        solid = np.pad(solid, 1, mode="constant", constant_values=0.0)
    if solid.min() == solid.max():
        print("  [STL] Skipped — grid has no fluid/solid interface.")
        return
    verts, faces, _, _ = _mc(solid, level=0.5, spacing=tuple(h),
                             allow_degenerate=False)
    verts = verts + (0.5 - pad) * h          # index -> physical voxel-centre frame

    if _HAS_TRIMESH:
        mesh = _trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        mesh.export(filename)
        n_tri = len(mesh.faces)
    else:
        tri = verts[faces].astype(np.float32)                 # (n, 3, 3)
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln  = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = np.where(ln > 1e-20, nrm / np.maximum(ln, 1e-20), 0.0)
        rec = np.zeros(len(tri), dtype=np.dtype([("n", "<f4", (3,)),
                                                 ("v", "<f4", (3, 3)),
                                                 ("a", "<u2")]))
        rec["n"], rec["v"] = nrm, tri
        n_tri = len(rec)
        with open(filename, "wb") as fh:
            fh.write(b"PorousGen voxel-to-STL export (solid phase)".ljust(80, b" "))
            fh.write(struct.pack("<I", n_tri))
            rec.tofile(fh)
    if verbose:
        size = os.path.getsize(filename) / 1024
        print(f"  [STL]  {filename!r}  ({n_tri} triangles, "
              f"{'watertight' if closed else 'open'}, {size:.0f} KB)")


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
    periodic:         bool   = False,
    return_info:      bool   = False,
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
        Boundary treatment
    ------------------
    periodic=False (default): wall-confined packing - every particle lies
        entirely inside the box. Expect a porosity excess in a layer about
        one particle radius thick next to each face (the classical wall
        effect of confined packings).
    periodic=True: periodic RVE - centres anywhere in [0, L), minimum-image
        overlap test, spheres that cross a face re-enter through the
        opposite face. Requires 2*r < L on every axis.
    Porosity is controlled indirectly (count and radii); the realised value
    is reported, and a RuntimeWarning is raised if RSA jams before all
    requested particles are placed.
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
    L_arr   = np.asarray(domain_size, dtype=float)
    centres = []   # list of np.ndarray, physical coordinates
    radii   = []   # list of float, accepted radius per particle

    attempt = -1
    for attempt in range(max_attempts):
        r_i = _draw_radius()

        if periodic:
            # Periodic RVE: centre anywhere in [0, L); a sphere crossing a
            # face re-enters through the opposite face.
            if np.any(2.0 * r_i >= L_arr):
                continue
            c = rng.uniform(np.zeros(ndim), L_arr)
        else:
            # Wall-confined packing: candidate drawn at least r_i from each
            # wall so the particle fits entirely inside the box.
            lo = np.full(ndim, r_i)
            hi = np.array([domain_size[k] - r_i for k in range(ndim)])
            if np.any(hi <= lo):
                continue   # this particle doesn't fit — try another draw
            c = rng.uniform(lo, hi)

        # Hard-sphere overlap check, generalised to unequal radii
        # (minimum-image separation when periodic)
        if centres:
            arr      = np.asarray(centres)             # (N, ndim)
            rarr     = np.asarray(radii)                # (N,)
            diffs    = arr - c[np.newaxis, :]
            if periodic:
                diffs -= L_arr * np.round(diffs / L_arr)
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
            if periodic:
                # Index window around the sphere, wrapped onto the grid; the
                # coordinate of each voxel is taken as its periodic image
                # nearest to the centre.
                idx, sub_axes = [], []
                for k in range(ndim):
                    a = int(np.floor((c[k] - r) / voxel_size[k])) - 1
                    b = int(np.ceil ((c[k] + r) / voxel_size[k])) + 1
                    raw = np.arange(a, b)
                    idx.append(np.mod(raw, shape[k]))
                    sub_axes.append((raw + 0.5) * voxel_size[k])
                sub_mesh = np.meshgrid(*sub_axes, indexing="ij")
                mask = sum((m - c[k])**2 for k, m in enumerate(sub_mesh)) <= r * r
                grid[np.ix_(*idx)] = np.where(mask, SOLID, grid[np.ix_(*idx)])
                continue

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

    info = dict(generator="granular", periodic=periodic,
                particles_placed=len(centres), particles_requested=n_particles,
                porosity_control="indirect (particle count and radii)")
    if len(centres) < n_particles:
        warnings.warn(f"generate_granular: only {len(centres)} of {n_particles} "
                      f"particles placed in {max_attempts} RSA attempts "
                      f"(jamming limit reached); porosity is higher than intended.",
                      RuntimeWarning, stacklevel=2)
    return (grid, info) if return_info else grid


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
    porosity:     float  = None,
    periodic:     bool   = False,
    return_info:  bool   = False,
) -> np.ndarray:
    """
    FIBROUS POROUS MEDIUM — randomly oriented fibres: capsules (spherocylinders)
    in 3-D, stadium-shaped segments in 2-D.

    Algorithm
    ---------
    1. Draw n_fibers centres uniformly in the box and orientations uniformly
       on the unit sphere (Gaussian-normalisation) or half-circle (2-D).
    2. Build the distance from every voxel centre to the nearest fibre axis
       segment. Each fibre is evaluated only inside its own bounding box
       (O(box) per fibre, not O(grid)).
    3. Solid where distance <= fiber_radius   (porosity=None, default), or
       COUNT SELECTION: the k = round((1-phi)*N) voxels closest to any fibre
       axis become solid (porosity given). All fibres then share one
       effective radius r* chosen so that the porosity is exact; the fibre
       SHAPE (capsule of uniform radius) is unchanged.

    Parameters
    ----------
    fiber_radius : cross-sectional radius [m] (initial guess when porosity set)
    fiber_length : full axial length of each fibre [m]
    n_fibers     : number of fibres (overlaps allowed)
    porosity     : if given, honoured EXACTLY by count selection
    periodic     : False (default) - fibres crossing a face are clipped.
                   True  - periodic RVE: every periodic image of each fibre
                   is included, so a fibre leaving through one face re-enters
                   through the opposite face (exact for any fibre length).
    return_info  : also return a dict incl. the effective radius r*.
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)

    rng  = np.random.default_rng(seed)
    segs = []
    for _ in range(n_fibers):
        c = np.array([rng.uniform(0.0, domain_size[i]) for i in range(ndim)])
        if ndim == 2:
            theta = rng.uniform(0.0, np.pi)
            u     = np.array([np.cos(theta), np.sin(theta)])
        else:
            raw = rng.standard_normal(3)
            u   = raw / (np.linalg.norm(raw) + 1e-20)
        half = (fiber_length / 2.0) * u
        segs.append((c - half, c + half))

    h = float(np.mean(C.voxel_size(shape, domain_size)))
    info = dict(generator="fibrous", periodic=periodic, n_fibers=n_fibers)

    if porosity is None:
        dist = C.segment_distance_field(shape, domain_size, segs,
                                        reach=fiber_radius, periodic=periodic)
        solid = dist <= fiber_radius
        info.update(porosity_control="indirect (fibre count and radius)",
                    fiber_radius=fiber_radius)
    else:
        k = C.solid_count(porosity, int(np.prod(shape)))
        reach = max(2.0 * fiber_radius, 2.0 * h)
        diag = float(np.linalg.norm(domain_size))
        while True:
            dist = C.segment_distance_field(shape, domain_size, segs,
                                            reach=reach, periodic=periodic)
            if k == 0 or C.kth_smallest(dist, k) <= reach or reach > diag:
                break
            reach *= 2.0            # threshold lies beyond the evaluated band
        solid = C.select_k_smallest(dist, k, seed=seed)
        r_eff = C.kth_smallest(dist, k) if k > 0 else 0.0
        info.update(porosity_control="exact", fiber_radius_effective=r_eff)

    grid = np.where(solid, SOLID, FLUID).astype(np.int8)
    if verbose:
        phi = compute_porosity(grid)
        extra = (f", r* = {info['fiber_radius_effective']:.4g}"
                 if porosity is not None else "")
        print(f"  Fibrous: {n_fibers} fibres,  φ = {phi:.4f}{extra}"
              f"  ({'periodic' if periodic else 'non-periodic'})")
    return (grid, info) if return_info else grid


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
    periodic:       bool   = False,
    return_info:    bool   = False,
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
    target_porosity: if given, walls are grown to EXACTLY this porosity
    periodic       : False (default) - free boundaries: seeds are ordinary
                     points in the box and a domain face is never treated
                     as a cell wall by itself.
                     True - periodic RVE: toroidal nearest-seed distances
                     (cKDTree boxsize), wrap-around wall detection, periodic
                     dilation and periodic distance field; the geometry
                     tiles seamlessly.
                     (v1.0 mixed the two: non-periodic seeds with wrap-around
                     wall detection, which created spurious walls on faces.)
    return_info    : also return a dict of realised generation parameters
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)

    rng = np.random.default_rng(seed)

    # ── Step 1: Random seed points (in [0, L) on every axis) ─────────────────
    seeds = np.column_stack([
        rng.uniform(0.0, domain_size[i], n_seeds) for i in range(ndim)
    ])

    # ── Step 2: Voronoi cell label per voxel (nearest seed) ──────────────────
    # Chunked KD-tree query; toroidal distances when periodic.
    cell_grid = C.query_nearest(seeds, shape, domain_size, k=1,
                                periodic=periodic, want="index")

    # ── Step 3: Cell boundaries, with the SAME convention as Step 2 ─────────
    on_wall = C.label_boundaries(cell_grid, periodic=periodic)
    del cell_grid

    # ── Step 4: Wall thickness — physical value OR exact porosity target ──────
    mean_vox = np.mean([domain_size[i] / shape[i] for i in range(ndim)])
    if target_porosity is not None:
        k = C.solid_count(target_porosity, on_wall.size)
        dist = C.distance_to_feature(on_wall, periodic=periodic, k_needed=k)
        wall = C.select_k_smallest(dist, k, seed=seed)
        n_dilate = -1   # marker: porosity-driven
    else:
        n_dilate = max(1, int(round(wall_thickness / mean_vox)))
        wall = C.dilate_voxels(on_wall, n_dilate, periodic=periodic)

    grid = np.where(wall, SOLID, FLUID).astype(np.int8)

    info = dict(generator="cellular", periodic=periodic, n_seeds=n_seeds,
                porosity_control="exact" if target_porosity is not None
                else "geometric (wall_thickness)")
    if n_dilate > 0:
        info["wall_dilations_vox"] = n_dilate
    if verbose:
        phi = compute_porosity(grid)
        mode = (f"target φ={target_porosity}" if target_porosity is not None
                else f"wall = {n_dilate} vox")
        print(f"  Cellular: {n_seeds} cells,  {mode},  "
              f"{'periodic' if periodic else 'non-periodic'},  φ = {phi:.4f}")
    return (grid, info) if return_info else grid


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
    porosity:       float  = None,
    return_info:    bool   = False,
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
    porosity       : if given, honoured EXACTLY: the round(phi*N) voxels
                     closest to any fracture plane become fluid (all
                     fractures share one effective aperture).
    Boundary treatment: non-periodic only. Fractures are infinite planes
    of arbitrary orientation; a plane is periodic-compatible only for
    orientations commensurate with the box, so no periodic mode is offered.
    return_info    : also return a dict incl. the effective aperture.
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)

    rng   = np.random.default_rng(seed)
    th_lo = np.radians(orient_bounds[0])
    th_hi = np.radians(orient_bounds[1])

    planes = []
    for _ in range(n_fractures):
        c = np.array([rng.uniform(0.0, domain_size[i]) for i in range(ndim)])
        if ndim == 2:
            theta  = rng.uniform(th_lo, th_hi)
            normal = np.array([np.sin(theta), np.cos(theta)])
        else:
            theta  = rng.uniform(th_lo, th_hi)      # dip from horizontal
            phi_s  = rng.uniform(0.0, 2.0 * np.pi)  # strike azimuth
            normal = np.array([np.sin(theta) * np.cos(phi_s),
                               np.sin(theta) * np.sin(phi_s),
                               np.cos(theta)])
        normal /= np.linalg.norm(normal) + 1e-20
        planes.append((c, normal))

    # Distance to the nearest fracture plane, by broadcasting (no N x ndim array)
    dist = C.plane_distance_field(shape, domain_size, planes)
    info = dict(generator="consolidated", periodic=False, n_fractures=n_fractures)
    if porosity is None:
        fluid = dist <= fracture_width / 2.0
        info.update(porosity_control="indirect (fracture count and aperture)")
    else:
        # fluid = the round(phi N) voxels closest to a plane
        #       = complement of the k solid voxels farthest from every plane
        solid = C.select_k_smallest(-dist, C.solid_count(porosity, dist.size), seed=seed)
        fluid = ~solid
        a_eff = 2.0 * float(dist[fluid].max()) if fluid.any() else 0.0
        info.update(porosity_control="exact", fracture_width_effective=a_eff)
    grid = np.where(fluid, FLUID, SOLID).astype(np.int8)

    if verbose:
        print(f"  Consolidated: {n_fractures} fractures,  φ = {compute_porosity(grid):.4f}")
    return (grid, info) if return_info else grid


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


def generate_ordered_gyroid(
    shape:           tuple  = (96, 96, 96),
    domain_size:     tuple  = None,
    n_cells:         int    = 4,
    threshold:       float  = None,
    target_porosity: float  = 0.50,
    seed:            int    = 0,
    verbose:         bool   = True,
    return_info:     bool   = False,
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
    │   If threshold is None, the EXACT number of fluid voxels needed for   │
    │   target_porosity is selected directly on this grid's own G field     │
    │   via _solidify_to_porosity (rank-selection with a random tie-break,  │
    │   not a threshold estimated on a separate calibration grid) -- see    │
    │   the note below for why that distinction matters.                   │
    │                                                                        │
    │   2-D variant: cross-section at z = 0                                │
    │     G_2D(x,y) = sin(kx)·cos(ky) + sin(ky)                           │
    │     (3rd term vanishes because sin(0) = 0)                           │
    └────────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    n_cells         : number of complete Gyroid unit cells along each axis
    threshold       : iso-surface level t. None -> exact-porosity mode (below).
    target_porosity : target void fraction (only used when threshold is None)
    seed            : tie-break seed for the exact-porosity selection. The
                      Gyroid FIELD itself is fully deterministic; this seed
                      only resolves ties between voxels that land on
                      (near-)identical G values, which occur often enough on
                      a discretised trigonometric field that they cannot be
                      ignored -- see the note below. Same inputs, same seed
                      -> same output, every time.

    Boundary treatment
    ------------------
    The Gyroid is intrinsically periodic with period L_cell =
    domain_size[0] / n_cells on every axis. The generated grid is exactly
    periodic along axis i when domain_size[i] / L_cell is an integer (always
    true for a cubic box); otherwise the structure is cut mid-cell on that
    axis. No `periodic` switch is needed; the realised per-axis periodicity
    is reported in the info dict and a warning is issued if any axis is cut.
    return_info     : also return a dict of realised generation parameters.

    Note on exact porosity
    -----------------------
    Earlier versions of this function estimated the threshold t on a
    SEPARATE, fixed-size calibration grid (independent of the caller's
    requested shape/domain/n_cells), then applied that single scalar t to
    the real field with a plain G < t comparison. Because the calibration
    grid and the real grid are different discretisations of the same
    continuous field, the achieved porosity on the real grid could differ
    from the target by far more than one voxel -- worse at low resolution,
    where G takes on many repeated or near-repeated values ("tie
    plateaus") that a scalar threshold cannot split predictably.
    generate_cellular/_solidify_to_porosity never had this problem because
    they always select the target vote count directly from the ACTUAL
    field being classified. Auto mode below does the same for Gyroid: G is
    evaluated once, on the real grid, and exactly
    round(target_porosity * N) voxels -- the N with the smallest G, i.e.
    the correct polarity for "FLUID where G < t" -- are marked fluid by
    rank, with ties broken by an infinitesimal random perturbation rather
    than left to floating-point comparison order.
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)

    # Unit-cell period: n_cells complete cycles fit in domain_size[0]
    L_cell = domain_size[0] / n_cells
    k_wave = 2.0 * np.pi / L_cell

    # Broadcast 1-D coordinate axes instead of full meshgrids (memory: one
    # grid-sized float array instead of ndim + 1 of them).
    ax = C.broadcast_axes(C.voxel_centres(shape, domain_size))
    if ndim == 2:
        X, Y = ax
        G = np.sin(k_wave * X) * np.cos(k_wave * Y)
        G = G + np.sin(k_wave * Y)
    else:
        X, Y, Z = ax
        G = (np.sin(k_wave * X) * np.cos(k_wave * Y)     # first sum allocates
             + np.sin(k_wave * Y) * np.cos(k_wave * Z))   # the full grid
        G += np.sin(k_wave * Z) * np.cos(k_wave * X)

    if threshold is not None:
        # Explicit iso-surface level requested: honoured as given. No
        # exact-porosity guarantee is implied in this mode.
        grid = np.where(G < threshold, FLUID, SOLID).astype(np.int8)
        display_threshold = threshold
    else:
        # Exact-porosity mode: rank-select on the REAL field. FLUID is the
        # smallest-G voxels, i.e. SOLID = the k largest G = k smallest (-G).
        k = C.solid_count(target_porosity, G.size)
        solid = C.select_k_smallest(-G, k, seed=seed)
        grid  = np.where(solid, SOLID, FLUID).astype(np.int8)
        display_threshold = (C.kth_smallest(-G, k) * -1.0) if k > 0 else float(G.max())

    cells_per_axis = [domain_size[i] / L_cell for i in range(ndim)]
    periodic_axes  = [bool(abs(c - round(c)) < 1e-9 and round(c) >= 1)
                      for c in cells_per_axis]
    if not all(periodic_axes):
        warnings.warn("generate_ordered_gyroid: the box does not hold a whole "
                      "number of unit cells on every axis "
                      f"(cells per axis = {[round(c, 4) for c in cells_per_axis]}); "
                      "the structure is not periodic on the cut axes.",
                      RuntimeWarning, stacklevel=2)
    info = dict(generator="gyroid", periodic=all(periodic_axes),
                periodic_axes=periodic_axes, cell_length=L_cell,
                iso_level=float(display_threshold),
                porosity_control="exact" if threshold is None else "iso-level")
    if verbose:
        phi = compute_porosity(grid)
        print(f"  Gyroid: L={L_cell:.4f} m,  t~{display_threshold:.4f},  "
              f"{n_cells} cells/axis,  φ = {phi:.4f},  "
              f"periodic={'yes' if all(periodic_axes) else periodic_axes}")
    return (grid, info) if return_info else grid


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
    periodic:     bool  = False,
    return_info:  bool  = False,
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
    2. For every voxel query the 1st and 3rd nearest seeds (chunked cKDTree).
       A voxel lies near a cell EDGE when d3 - d1 is small (3+ cells meet).
    3. Take the thinnest edge skeleton (smallest 2 % of d3 - d1).
    4. Grow the struts by COUNT SELECTION on the Euclidean distance-to-
       skeleton field: exactly k = round((1-phi)*N) voxels become solid.

    Parameters
    ----------
    porosity : open volume fraction — honoured exactly (count selection).
    n_cells  : cells per axis; higher = finer struts (more ligaments).
    periodic : False (default) free boundaries; True = periodic RVE
               (toroidal seed distances and periodic distance field).
               The jittered seed lattice is itself periodic-compatible.
    return_info : also return a dict of realised generation parameters.
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)
    rng = np.random.default_rng(seed)

    # jittered lattice of seeds -> more uniform cells than pure random
    axes = [np.linspace(0, domain_size[i], n_cells, endpoint=False)
            + domain_size[i] / (2 * n_cells) for i in range(ndim)]
    seeds = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, ndim)
    seeds += rng.uniform(-0.35, 0.35, seeds.shape) * (domain_size[0] / n_cells)

    edge_metric = C.query_nearest(                        # small near edges
        seeds, shape, domain_size, k=3, periodic=periodic,
        reduce=lambda d: d[:, 2] - d[:, 0])
    skeleton = edge_metric <= np.quantile(edge_metric, 0.02)
    del edge_metric

    k = C.solid_count(porosity, skeleton.size)
    dist = C.distance_to_feature(skeleton, periodic=periodic, k_needed=k)
    solid = C.select_k_smallest(dist, k, seed=seed)
    grid = np.where(solid, SOLID, FLUID).astype(np.int8)

    info = dict(generator="open_foam", periodic=periodic,
                n_cells_total=len(seeds), porosity_control="exact")
    if verbose:
        print(f"  Open foam: {len(seeds)} cells,  target φ={porosity:.3f}"
              f"  ->  φ = {compute_porosity(grid):.4f}"
              f"  ({'periodic' if periodic else 'non-periodic'})")
    return (grid, info) if return_info else grid


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
    periodic:           bool  = False,
    return_info:        bool  = False,
) -> np.ndarray:
    """
    STOCHASTIC POROUS MEDIUM — thresholded Gaussian random field ("blobs").

    The standard stochastic-reconstruction benchmark (soils, sandstones,
    membranes): white noise is smoothed with a Gaussian kernel of width
    `correlation_length` (sets the characteristic blob / pore size), and the
    (1-phi) fraction of voxels with the LARGEST field values is made solid
    by COUNT SELECTION - exactly k = round((1-phi)*N) solid voxels.

    Boundary treatment
    ------------------
    periodic=True  : noise smoothed with wrap-around (mode="wrap"); the
                     field is exactly periodic and the geometry tiles
                     seamlessly. This is the v1.0 behaviour.
    periodic=False : (default) non-periodic but STATISTICALLY STATIONARY up
                     to the faces: the noise is drawn on a box padded by 3
                     standard deviations of the kernel and the smoothed
                     field is cropped back. (Reflect/mirror padding is
                     avoided on purpose: near a mirror plane each noise
                     sample is counted twice, which inflates the field
                     variance there and biases the local porosity.)
                     Memory is higher than periodic mode when the
                     correlation length is a sizeable fraction of the box.
    """
    from scipy.ndimage import gaussian_filter
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)
    rng = np.random.default_rng(seed)
    sigma = [correlation_length / (domain_size[i] / shape[i]) for i in range(ndim)]

    if correlation_length > 0.5 * min(domain_size):
        warnings.warn(f"generate_blob: correlation_length={correlation_length} is "
                      f"more than half the smallest box edge ({min(domain_size)}); "
                      "correlation_length is in the same (absolute) units as "
                      "domain_size, so blobs will be larger than the box.",
                      RuntimeWarning, stacklevel=2)
    if periodic:
        noise = rng.standard_normal(shape)
        field = gaussian_filter(noise, sigma=sigma, mode="wrap")
    else:
        truncate = 3.0
        # Padding of 3 kernel sigmas makes the cropped field stationary. It is
        # capped at half the box per side (<= 8x memory in 3-D): a kernel that
        # wide means blobs larger than the box, which is warned about above.
        pad = [min(int(np.ceil(truncate * sg)), n // 2 + 1)
               for sg, n in zip(sigma, shape)]
        noise = rng.standard_normal([n + 2 * p for n, p in zip(shape, pad)],
                                    dtype=np.float32)
        field = gaussian_filter(noise, sigma=sigma, mode="wrap",
                                truncate=truncate, output=np.float32)
        field = field[tuple(slice(p, p + n) for p, n in zip(pad, shape))]
    del noise

    # largest field values -> solid  ==  smallest (-field) -> solid
    solid = C.select_k_smallest(-field, C.solid_count(porosity, field.size),
                                seed=seed)
    grid = np.where(solid, SOLID, FLUID).astype(np.int8)

    info = dict(generator="blob", periodic=periodic, porosity_control="exact",
                kernel_sigma_vox=[round(float(x), 3) for x in sigma])
    if verbose:
        print(f"  Blobs: corr={correlation_length},  target φ={porosity:.3f}"
              f"  ->  φ = {compute_porosity(grid):.4f}"
              f"  ({'periodic' if periodic else 'non-periodic'})")
    return (grid, info) if return_info else grid


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 ─  GEOMETRY 8 : OVERLAPPING SPHERES  (Boolean model)
# ══════════════════════════════════════════════════════════════════════════════
def generate_overlapping_spheres(
    shape:        tuple = (96, 96, 96),
    domain_size:  tuple = None,
    porosity:     float = 0.55,
    radius:       float = 0.07,     # nominal sphere radius, in domain units
    seed:         int   = 11,
    verbose:      bool  = True,
    periodic:     bool  = False,
    return_info:  bool  = False,
) -> np.ndarray:
    """
    BOOLEAN (overlapping-spheres) MODEL — freely inter-penetrating spheres.

    The classic "Swiss-cheese" stochastic medium: sphere centres follow a
    Poisson point process and spheres may overlap, producing consolidated-
    looking solids.

    Algorithm (v1.1)
    ----------------
    1. Number of spheres from the Boolean-model relation
           phi = exp(-lambda * v_sph)   ->   n = -ln(phi) * V_w / v_sph
       with the nominal `radius`, so the expected porosity is the target.
    2. Centres uniform in the sampling window V_w: the box itself when
       periodic, otherwise the box grown by 1.5 radii on every side, so that
       spheres centred just outside also cut into the box (the correct
       window sampling of a Boolean model - no porosity excess near faces).
    3. Distance from every voxel to the nearest centre (chunked KD-tree,
       toroidal when periodic); COUNT SELECTION makes exactly
       k = round((1-phi)*N) voxels solid. The result is a union of equal
       spheres of one effective radius r* (reported), close to `radius`.

    Porosity is therefore EXACT. (v1.0 added spheres one at a time until the
    porosity crossed the target; the overshoot could exceed one sphere and
    each addition cost a full-grid pass.)
    """
    ndim = len(shape)
    if domain_size is None:
        domain_size = tuple(1.0 for _ in shape)
    rng = np.random.default_rng(seed)
    L   = np.asarray(domain_size, dtype=float)

    v_sph  = (np.pi * radius**2 if ndim == 2 else 4.0 / 3.0 * np.pi * radius**3)
    margin = 0.0 if periodic else 1.5 * radius
    lo, hi = -margin * np.ones(ndim), L + margin
    V_w    = float(np.prod(hi - lo))
    n      = max(1, int(round(-np.log(max(porosity, 1e-12)) * V_w / v_sph)))
    centres = rng.uniform(lo, hi, size=(n, ndim))

    k = C.solid_count(porosity, int(np.prod(shape)))
    dist = C.query_nearest(centres, shape, domain_size, k=1, periodic=periodic)
    solid = C.select_k_smallest(dist, k, seed=seed)
    r_eff = C.kth_smallest(dist, k) if k > 0 else 0.0
    grid = np.where(solid, SOLID, FLUID).astype(np.int8)

    info = dict(generator="overlapping_spheres", periodic=periodic,
                n_spheres=n, radius_nominal=radius, radius_effective=r_eff,
                porosity_control="exact")
    if verbose:
        print(f"  Overlapping spheres: {n} spheres, r*={r_eff:.4g} "
              f"(nominal {radius}),  target φ={porosity:.3f}"
              f"  ->  φ = {compute_porosity(grid):.4f}"
              f"  ({'periodic' if periodic else 'non-periodic'})")
    return (grid, info) if return_info else grid


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
    info:        dict  = None,
    metrics:     dict  = None,
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
    A(f"  generated by PorousGen v{__version__}")
    A("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    A("=" * 66)
    A("")
    A("  -- Grid --")
    names = ["nx", "ny", "nz"][:ndim]
    for n, s in zip(names, shape):
        A(f"  {n:<4}: {s}")
    A(f"  total voxels : {grid.size:,}")
    A(f"  fluid voxels : {int(np.sum(grid == FLUID)):,}   (exact integer count)")
    A(f"  porosity     : {phi:.6f}   [fluid / total]")
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
    if info:
        A("")
        A("  -- Realised generation --")
        for k, v in info.items():
            if isinstance(v, float):
                v = f"{v:.6g}"
            A(f"  {k:<18}: {v}")
    A("")
    A("  -- File format --")
    A("  .dat : plain-text (UTF-8) voxel file, 0 = fluid, 1 = solid,")
    A("         tab-separated; x-slices of ny rows x nz columns separated by")
    A("         a blank line (3-D). Read with whitespace-delimited >> in C++.")
    if metrics:
        from .metrics import format_metrics
        A("")
        A(format_metrics(metrics))
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
    info:        dict  = None,
    metrics:     bool  = False,
    flow_axis:   int   = 0,
) -> dict:
    """
    ONE-CALL simulation-ready bundle. Writes:
        <basename>.dat        PALABOS voxel file (0 fluid / 1 solid)
        <basename>.stl        binary surface mesh          (3-D grids only)
        <basename>.png        rendered preview
        <basename>_info.txt   dimensions, exact voxel counts, porosity,
                              XML snippets, parameters, realised info
        <basename>_metrics.json  structural/topological metrics
                              (only when metrics=True; see porousgen.metrics)
    `info` is the dict returned by a generator called with return_info=True.
    Returns a dict of the written paths.
    """
    out = {}
    dat = f"{basename}.dat"
    export_palabos(grid, dat, verbose=verbose)
    out["dat"] = dat

    if stl and grid.ndim == 3:
        try:
            export_stl(grid, f"{basename}.stl", verbose=verbose,
                       domain_size=domain_size)
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

    m = None
    if metrics:
        import json
        from .metrics import compute_metrics
        m = compute_metrics(grid, domain_size=domain_size, flow_axis=flow_axis)
        with open(f"{basename}_metrics.json", "w") as fh:
            json.dump(m, fh, indent=2)
        out["metrics"] = f"{basename}_metrics.json"
        if verbose:
            print(f"  [metrics] {basename}_metrics.json")
    write_info_file(grid, f"{basename}_info.txt", domain_size,
                    params, dat_name=os.path.basename(dat), info=info, metrics=m)
    out["info"] = f"{basename}_info.txt"
    if verbose:
        print(f"  [info] {basename}_info.txt")
    return out
