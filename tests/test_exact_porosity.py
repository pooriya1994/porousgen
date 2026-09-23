"""
Exact-porosity campaign (reviewer-requested protocol): odd and even grids,
2-D and 3-D, several cell/seed counts, extreme targets, and INTEGER voxel
counts rather than rounded porosity. The contract is exact:
N_solid == round((1 - phi) * N), hence |N_fluid - round(phi * N)| <= 1.
"""
import itertools
import numpy as np
import pytest
import porousgen as pg
from conftest import n_fluid

GRIDS = [(31, 31), (32, 32), (64, 65), (17, 17, 17), (24, 24, 24), (25, 23, 21)]
TARGETS = [0.02, 0.10, 0.35, 0.50, 0.75, 0.90, 0.9873]

EXACT = {
    "cellular":  lambda s, p, per, e: pg.generate_cellular(s, n_seeds=e, target_porosity=p, seed=5, periodic=per, verbose=False),
    "open_foam": lambda s, p, per, e: pg.generate_open_foam(s, porosity=p, n_cells=e, seed=5, periodic=per, verbose=False),
    "blob":      lambda s, p, per, e: pg.generate_blob(s, porosity=p, correlation_length=0.04 * e, seed=5, periodic=per, verbose=False),
    "spheres":   lambda s, p, per, e: pg.generate_overlapping_spheres(s, porosity=p, radius=0.03 * e, seed=5, periodic=per, verbose=False),
    "fibrous":   lambda s, p, per, e: pg.generate_fibrous(s, porosity=p, n_fibers=6 * e, fiber_radius=0.02, seed=5, periodic=per, verbose=False),
    "gyroid":    lambda s, p, per, e: pg.generate_ordered_gyroid(s, n_cells=e, target_porosity=p, verbose=False),
    "fractured": lambda s, p, per, e: pg.generate_consolidated(s, n_fractures=3 * e, porosity=p, seed=5, verbose=False),
}
EXTRA = [2, 3]
PERIODIC_ABLE = {"cellular", "open_foam", "blob", "spheres", "fibrous"}

cases = []
for name in EXACT:
    for per in ([False, True] if name in PERIODIC_ABLE else [False]):
        for grid, e in itertools.product(GRIDS, EXTRA):
            cases.append((name, per, grid, e))


@pytest.mark.parametrize("name,periodic,grid,extra", cases,
                         ids=[f"{n}-{'per' if p else 'np'}-{'x'.join(map(str, g))}-{e}" for n, p, g, e in cases])
def test_exact_integer_voxel_count(name, periodic, grid, extra):
    for phi in TARGETS:
        g = EXACT[name](grid, phi, periodic, extra)
        N = g.size
        assert g.shape == grid
        assert N - n_fluid(g) == round((1 - phi) * N), (name, grid, phi)
        assert abs(n_fluid(g) - round(phi * N)) <= 1


def test_gyroid_reviewer_worst_case_region():
    """The reviewer's failing regime: 3-D, odd/even resolutions 31..96."""
    for r in (31, 32, 47, 48, 63, 64, 95, 96):
        for cells in (3, 4, 7):
            for phi in (0.10, 0.5, 0.9873):
                g = pg.generate_ordered_gyroid((r, r, r), n_cells=cells, target_porosity=phi, verbose=False)
                assert g.size - n_fluid(g) == round((1 - phi) * g.size)


def test_count_selection_is_deterministic():
    a = pg.generate_open_foam((24, 24, 24), porosity=0.8, n_cells=3, seed=9, verbose=False)
    b = pg.generate_open_foam((24, 24, 24), porosity=0.8, n_cells=3, seed=9, verbose=False)
    assert np.array_equal(a, b)


def test_smooth_tie_break_avoids_single_voxel_roughness():
    """The partially filled cut-off shell must form coherent patches, not
    salt-and-pepper voxels (which would act as artificial wall roughness)."""
    from scipy import ndimage as ndi
    from porousgen import _core as C
    rng = np.random.default_rng(13)
    lab = C.query_nearest(rng.random((20, 3)), (48,) * 3, (1, 1, 1), want="index")
    skel = C.label_boundaries(lab)
    k = C.solid_count(0.85, skel.size)
    dist = C.distance_to_feature(skel, k_needed=k)
    kern = ndi.generate_binary_structure(3, 1).astype(int)
    kern[1, 1, 1] = 0

    def bumps(s):
        return int((s & (ndi.convolve(s.astype(int), kern, mode="constant") <= 1)).sum())
    smooth = C.select_k_smallest(dist, k, seed=1, tie_break="smooth")
    rand = C.select_k_smallest(dist, k, seed=1, tie_break="random")
    assert smooth.sum() == rand.sum() == k
    assert bumps(smooth) * 3 < bumps(rand)      # 4.8x at 48^3, ~70x at 96^3
