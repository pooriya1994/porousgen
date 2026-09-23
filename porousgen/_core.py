"""
PorousGen numerical core (v1.1).

Shared, generator-independent building blocks:

* ``select_k_smallest``   – the exact-porosity count-selection engine;
* ``distance_to_feature`` – Euclidean distance to a voxel skeleton, with an
                            exact periodic mode (wrap-padded EDT);
* ``query_nearest``       – chunked KD-tree queries over all voxel centres,
                            optionally with periodic (toroidal) distances;
* ``segment_distance_field`` / ``plane_distance_field`` – distance fields for
                            the fibrous and fractured architectures.

Design rules
------------
1. Exactness is by construction: every porosity-driven generator produces a
   raw scalar field and hands it to ``select_k_smallest``, which marks
   exactly ``k = round((1 - phi) * N)`` voxels as solid.
2. Periodic modes are exact, not approximate: periodic distances are
   computed with the minimum-image convention, never by post-hoc wrapping.
3. No routine materialises an (N, ndim) float64 coordinate array for the
   whole grid; memory scales with the grid itself, not with ndim times it.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────────────────────────────────────
#  Exact count selection
# ─────────────────────────────────────────────────────────────────────────────
def solid_count(porosity: float, n: int) -> int:
    """Number of solid voxels for a target porosity: k = round((1-phi) N)."""
    if not 0.0 <= porosity <= 1.0:
        raise ValueError(f"porosity must lie in [0, 1], got {porosity!r}")
    return int(min(max(round((1.0 - porosity) * n), 0), n))


def kth_smallest(values: np.ndarray, k: int) -> float:
    """The k-th smallest value (1-based) of an array, without a full sort."""
    flat = np.asarray(values).ravel()
    return float(np.partition(flat, k - 1)[k - 1])


def _smooth_tie_key(flat_idx: np.ndarray, shape, seed: int, n_modes: int = 12,
                    k_max: int = 3) -> np.ndarray:
    """
    Smooth, seeded, low-frequency random field evaluated only at the given
    voxels: a sum of cosines with small INTEGER wave numbers per axis, hence
    exactly periodic on the grid (safe for periodic and non-periodic modes).
    """
    rng = np.random.default_rng([seed, 7919])
    coords = np.unravel_index(flat_idx, shape)
    key = np.zeros(flat_idx.size)
    for _ in range(n_modes):
        kvec = rng.integers(-k_max, k_max + 1, size=len(shape))
        if not kvec.any():
            kvec[rng.integers(len(shape))] = 1
        arg = sum(2.0 * np.pi * kvec[a] * coords[a] / shape[a] for a in range(len(shape)))
        key += np.cos(arg + rng.uniform(0.0, 2.0 * np.pi))
    return key


def select_k_smallest(values: np.ndarray, k: int, seed: int = 0,
                      tie_break: str = "smooth") -> np.ndarray:
    """
    Boolean mask marking EXACTLY ``k`` voxels: those with the smallest values.

    Let t be the k-th smallest value. Every voxel with value < t is selected;
    the remaining ``need = k - #{v < t}`` voxels are chosen from the tie set
    {v == t}. The result therefore contains exactly ``k`` voxels for ANY
    input, including fields with large plateaus of equal values (integer
    distance shells, or a discretised analytic field sampled at symmetric
    points) - the situation in which a scalar threshold such as
    ``v < quantile`` cannot hit the target count.

    tie_break
      "smooth" (default): tied voxels are ranked by a smooth, seeded,
               low-frequency random field, so the partially filled shell
               forms coherent patches - a gentle +/-1-voxel variation of
               wall/strut thickness - instead of isolated single voxels.
               This avoids one-voxel surface roughness, which would act as
               artificial wall roughness in a bounce-back LBM simulation.
      "random": uniform random choice among the ties (v1.0-like; produces
               salt-and-pepper texture on the outermost shell).

    Deterministic for a given ``seed``. Memory: one copy of the input for
    the partition, plus boolean masks.
    """
    shape = np.shape(values)
    flat = np.asarray(values).ravel()
    n = flat.size
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, {n}]")
    out = np.zeros(n, dtype=bool)
    if k == 0:
        return out.reshape(shape)
    if k == n:
        out[:] = True
        return out.reshape(shape)
    t = np.partition(flat, k - 1)[k - 1]
    np.less(flat, t, out=out)
    need = k - int(np.count_nonzero(out))
    if need > 0:
        ties = np.flatnonzero(flat == t)
        if need == ties.size:
            out[ties] = True
        elif tie_break == "smooth" and len(shape) >= 1:
            key = _smooth_tie_key(ties, shape, seed)
            out[ties[np.argpartition(key, need - 1)[:need]]] = True
        elif tie_break in ("random", "smooth"):
            rng = np.random.default_rng(seed)
            out[rng.choice(ties, size=need, replace=False)] = True
        else:
            raise ValueError(f"unknown tie_break {tie_break!r}")
    return out.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
#  Coordinates
# ─────────────────────────────────────────────────────────────────────────────
def voxel_size(shape, domain_size) -> np.ndarray:
    return np.array([domain_size[i] / shape[i] for i in range(len(shape))])


def voxel_centres(shape, domain_size) -> list:
    """1-D arrays of voxel-centre coordinates; first centre at dx/2."""
    return [(np.arange(shape[i]) + 0.5) * (domain_size[i] / shape[i])
            for i in range(len(shape))]


def broadcast_axes(coords_1d: list) -> list:
    """Reshape 1-D coordinate arrays so they broadcast against the grid."""
    ndim = len(coords_1d)
    out = []
    for i, c in enumerate(coords_1d):
        s = [1] * ndim
        s[i] = c.size
        out.append(c.reshape(s))
    return out


def _slab_points(coords_1d: list, a: int, b: int) -> np.ndarray:
    """(M, ndim) centres of voxels with first index in [a, b)."""
    mesh = np.meshgrid(coords_1d[0][a:b], *coords_1d[1:], indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


# ─────────────────────────────────────────────────────────────────────────────
#  Chunked nearest-neighbour queries over all voxel centres
# ─────────────────────────────────────────────────────────────────────────────
def query_nearest(points: np.ndarray, shape, domain_size, k: int = 1,
                  periodic: bool = False, want: str = "distance",
                  reduce=None, chunk_voxels: int = 2_000_000):
    """
    Query the ``k`` nearest ``points`` for every voxel centre, in slabs.

    periodic=True uses toroidal distances (``cKDTree(boxsize=domain_size)``);
    the points are wrapped into [0, L) first.

    want = "distance" -> float64 array, shape (*shape, k) (or shape if k == 1)
    want = "index"    -> int32/int64 array of point indices, same layout
    reduce            -> optional callable applied to each chunk's (M, k)
                         distance block, returning one value per voxel; the
                         result then has the grid's shape (memory: 8 B/voxel
                         instead of 8k B/voxel).
    """
    pts = np.asarray(points, dtype=np.float64)
    if periodic:
        L = np.asarray(domain_size, dtype=np.float64)
        pts = np.mod(pts, L)
        pts[pts >= L] = 0.0           # guard against x == L after mod
        tree = cKDTree(pts, boxsize=L)
    else:
        tree = cKDTree(pts)
    coords = voxel_centres(shape, domain_size)
    per_slab = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    step = max(1, chunk_voxels // max(per_slab, 1))
    idx_dtype = np.int32 if len(pts) < 2**31 - 1 else np.int64
    tail = () if (k == 1 or reduce is not None) else (k,)
    out = np.empty(tuple(shape) + tail,
                   dtype=np.float64 if want == "distance" else idx_dtype)
    for a in range(0, shape[0], step):
        b = min(shape[0], a + step)
        d, i = tree.query(_slab_points(coords, a, b), k=k, workers=-1)
        block = d if want == "distance" else i
        if reduce is not None:
            block = reduce(block)
        out[a:b] = block.reshape((b - a,) + tuple(shape[1:]) + tail)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Euclidean distance to a voxel skeleton - exact periodic mode
# ─────────────────────────────────────────────────────────────────────────────
def _wrap_padded_edt(feature: np.ndarray, pad: int) -> np.ndarray:
    """EDT of ~feature on a grid wrap-padded by ``pad`` voxels, then cropped."""
    widths = [min(pad, int(math.ceil(n / 2))) for n in feature.shape]
    padded = np.pad(feature, [(w, w) for w in widths], mode="wrap")
    dist = distance_transform_edt(~padded)
    core = tuple(slice(w, w + n) for w, n in zip(widths, feature.shape))
    return dist[core]


def distance_to_feature(feature: np.ndarray, periodic: bool = False,
                        k_needed: int | None = None) -> np.ndarray:
    """
    Euclidean distance (in voxel units) from every voxel to the nearest
    ``True`` voxel of ``feature``.

    Non-periodic: plain ``scipy.ndimage.distance_transform_edt``.

    Periodic: minimum-image distance. The grid is wrap-padded by ``w``
    voxels and the EDT is taken on the padded grid. A voxel whose true
    periodic distance d satisfies d <= w is computed exactly (its nearest
    feature image lies inside the padded box); otherwise the computed value
    can only be too large. If ``k_needed`` is given, ``w`` is doubled until
    the k-th smallest computed distance t satisfies t <= w, which proves
    that the k voxels a count-selection will take are all exact. With
    w >= ceil(n/2) on every axis the whole field is exact.
    """
    if not np.any(feature):
        raise ValueError("feature is empty - no skeleton voxels to measure from")
    if not periodic:
        return distance_transform_edt(~feature)
    full = max(int(math.ceil(n / 2)) for n in feature.shape)
    if k_needed is None:
        return _wrap_padded_edt(feature, full)
    w = 8
    while True:
        dist = _wrap_padded_edt(feature, w)
        if w >= full or kth_smallest(dist, max(k_needed, 1)) <= w:
            return dist
        w *= 2


def dilate_voxels(mask: np.ndarray, n_iter: int, periodic: bool = False) -> np.ndarray:
    """``n_iter`` binary dilations with the full 3^ndim structuring element."""
    from scipy.ndimage import binary_dilation
    if n_iter <= 0:
        return mask.copy()
    struct = np.ones((3,) * mask.ndim, dtype=bool)
    if not periodic:
        return binary_dilation(mask, structure=struct, iterations=n_iter)
    w = [min(n_iter, n) for n in mask.shape]
    padded = np.pad(mask, [(p, p) for p in w], mode="wrap")
    padded = binary_dilation(padded, structure=struct, iterations=n_iter)
    return padded[tuple(slice(p, p + n) for p, n in zip(w, mask.shape))]


def label_boundaries(labels: np.ndarray, periodic: bool = False) -> np.ndarray:
    """
    Voxels that have at least one face neighbour with a different label.

    periodic=False compares neighbours inside the domain only, so a domain
    face is never reported as a boundary by itself. periodic=True also
    compares across opposite faces (wrap-around), which is correct only
    when the labels themselves were built periodically.
    """
    on = np.zeros(labels.shape, dtype=bool)
    for ax in range(labels.ndim):
        if periodic:
            on |= np.roll(labels, 1, axis=ax) != labels
            on |= np.roll(labels, -1, axis=ax) != labels
        else:
            diff = np.diff(labels, axis=ax) != 0
            lo = [slice(None)] * labels.ndim
            hi = [slice(None)] * labels.ndim
            lo[ax] = slice(0, -1)
            hi[ax] = slice(1, None)
            on[tuple(lo)] |= diff
            on[tuple(hi)] |= diff
    return on


# ─────────────────────────────────────────────────────────────────────────────
#  Distance fields for segments (fibres) and planes (fractures)
# ─────────────────────────────────────────────────────────────────────────────
def _point_segment_distance(sub_axes, p0, p1):
    """Distance from a broadcast sub-grid of points to segment p0->p1."""
    d = p1 - p0
    len_sq = float(d @ d)
    rel = [ax - p0[i] for i, ax in enumerate(sub_axes)]
    if len_sq < 1e-30:
        return np.sqrt(sum(r * r for r in rel))
    t = sum(r * d[i] for i, r in enumerate(rel)) / len_sq
    np.clip(t, 0.0, 1.0, out=t)
    return np.sqrt(sum((r - t * d[i]) ** 2 for i, r in enumerate(rel)))


def segment_distance_field(shape, domain_size, segments, reach: float,
                           periodic: bool = False) -> np.ndarray:
    """
    Distance from every voxel centre to the nearest segment, exact for all
    voxels within ``reach`` of some segment (np.inf beyond).

    Each segment is evaluated only inside its bounding box grown by
    ``reach`` - O(box) rather than O(grid) work per fibre. In periodic mode
    every periodic image of a segment whose grown box meets the domain is
    evaluated, which is exact for fibres of any length (no minimum-image
    assumption on fibre length).
    """
    ndim = len(shape)
    L = np.asarray(domain_size, dtype=np.float64)
    h = voxel_size(shape, domain_size)
    coords = voxel_centres(shape, domain_size)
    field = np.full(shape, np.inf)
    shifts = [np.zeros(ndim)]
    if periodic:
        import itertools
        shifts = [np.array(s, dtype=float) for s in itertools.product((-1, 0, 1), repeat=ndim)]
    for p0, p1 in segments:
        for s in shifts:
            q0, q1 = p0 + s * L, p1 + s * L
            lo = np.minimum(q0, q1) - reach
            hi = np.maximum(q0, q1) + reach
            if np.any(hi < 0) or np.any(lo > L):
                continue
            i0 = [max(0, int(np.floor(lo[i] / h[i] - 0.5))) for i in range(ndim)]
            i1 = [min(shape[i], int(np.ceil(hi[i] / h[i] + 0.5)) + 1) for i in range(ndim)]
            if any(b <= a for a, b in zip(i0, i1)):
                continue
            sub = broadcast_axes([coords[i][i0[i]:i1[i]] for i in range(ndim)])
            d = _point_segment_distance(sub, q0, q1)
            region = tuple(slice(i0[i], i1[i]) for i in range(ndim))
            np.minimum(field[region], d, out=field[region])
    return field


def plane_distance_field(shape, domain_size, planes) -> np.ndarray:
    """Distance from every voxel centre to the nearest of a set of planes
    given as (point, unit_normal) pairs. Built by broadcasting - no
    (N, ndim) coordinate array."""
    axes = broadcast_axes(voxel_centres(shape, domain_size))
    field = np.full(shape, np.inf)
    for c, nrm in planes:
        s = sum((axes[i] - c[i]) * nrm[i] for i in range(len(shape)))
        np.minimum(field, np.abs(s), out=field)
    return field
