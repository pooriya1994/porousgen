"""
Boundary-convention tests.

Core level: periodic distances are compared with brute-force minimum-image
references (exact equality). Generator level: a periodic geometry must tile
without a seam. Three checks, matched to the structure:
  * stationary media (random seeds): the agreement between the two voxel
    layers that meet across the seam ranks like any interior adjacent pair
    (mean percentile ~ 50 %); a non-periodic control ranks far lower;
  * jittered-lattice foam: cell walls concentrate on lattice planes, one of
    which IS the seam, so the seam is compared with the interior lattice
    planes instead;
  * Gyroid: exactly invariant under a shift of one unit cell.
"""
import itertools
import numpy as np
import pytest
import porousgen as pg
from porousgen import _core as C


def _brute_periodic_edt(feat):
    idx = np.argwhere(feat)
    sh = np.array(feat.shape)
    out = np.empty(feat.shape)
    for p in itertools.product(*[range(s) for s in feat.shape]):
        d = np.abs(idx - np.array(p))
        d = np.minimum(d, sh - d)
        out[p] = np.sqrt((d ** 2).sum(1)).min()
    return out


@pytest.mark.parametrize("shape", [(9, 11), (7, 8, 9), (12, 5, 6)])
def test_periodic_edt_exact(shape):
    rng = np.random.default_rng(sum(shape))
    feat = rng.random(shape) < 0.04
    feat.flat[0] = True
    assert np.array_equal(C.distance_to_feature(feat, periodic=True), _brute_periodic_edt(feat))


def test_periodic_knn_exact():
    rng = np.random.default_rng(1)
    shape, L = (13, 11, 9), np.array([1.0, 0.8, 0.7])
    pts = rng.random((7, 3)) * L
    got = C.query_nearest(pts, shape, L, k=3, periodic=True).reshape(-1, 3)
    P = np.column_stack([m.ravel() for m in np.meshgrid(*C.voxel_centres(shape, L), indexing="ij")])
    d = P[:, None, :] - pts[None]
    d -= L * np.round(d / L)
    ref = np.sort(np.sqrt((d ** 2).sum(-1)), axis=1)[:, :3]
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_nonperiodic_walls_do_not_wrap():
    """Reviewer 2: v1.0 compared labels across opposite faces (np.roll)
    although the seeds were not periodic."""
    lab = np.zeros((6, 6), int)
    lab[:, 3:] = 1
    assert C.label_boundaries(lab, periodic=False)[0].tolist() == [0, 0, 1, 1, 0, 0]
    assert C.label_boundaries(lab, periodic=True)[0].tolist() == [1, 0, 1, 1, 0, 1]


def _seam_percentile(g):
    out = []
    for ax in range(g.ndim):
        n = g.shape[ax]
        pair = np.array([(np.take(g, i, axis=ax) == np.take(g, (i + 1) % n, axis=ax)).mean()
                         for i in range(n)])
        inner, seam = pair[:-1], pair[-1]
        out.append((inner < seam).mean() + 0.5 * (inner == seam).mean())
    return out


STATIONARY = {
    "cellular": lambda per, s: pg.generate_cellular((24, 24, 24), n_seeds=12, target_porosity=0.8, seed=s, periodic=per, verbose=False),
    "blob":     lambda per, s: pg.generate_blob((24, 24, 24), porosity=0.5, correlation_length=0.1, seed=s, periodic=per, verbose=False),
    "spheres":  lambda per, s: pg.generate_overlapping_spheres((24, 24, 24), porosity=0.6, radius=0.12, seed=s, periodic=per, verbose=False),
    "fibrous":  lambda per, s: pg.generate_fibrous((24, 24, 24), fiber_radius=0.05, n_fibers=25, seed=s, periodic=per, verbose=False),
    "granular": lambda per, s: pg.generate_granular((24, 24, 24), radius=0.11, n_particles=30, seed=s, periodic=per, verbose=False),
}


@pytest.mark.parametrize("name", list(STATIONARY))
def test_periodic_mode_is_seamless(name):
    r = [x for s in range(10) for x in _seam_percentile(STATIONARY[name](True, s))]
    assert 0.30 < np.mean(r) < 0.70, f"{name}: seam ranks at {np.mean(r):.2f}"


@pytest.mark.parametrize("name", ["cellular", "blob", "spheres"])
def test_nonperiodic_control_has_a_seam(name):
    """Power check: the same statistic does detect a seam when there is one."""
    r = [x for s in range(10) for x in _seam_percentile(STATIONARY[name](False, s))]
    assert np.mean(r) < 0.30


def test_open_foam_periodic_seam_matches_lattice_planes():
    N, cells = 30, 3            # lattice planes at pairs 9|10, 19|20 and the seam 29|0
    acc = {True: np.zeros(N), False: np.zeros(N)}
    for per in (True, False):
        for s in range(8):
            g = pg.generate_open_foam((N, N, N), porosity=0.85, n_cells=cells, seed=s, periodic=per, verbose=False)
            for ax in range(3):
                acc[per] += [(np.take(g, i, axis=ax) == np.take(g, (i + 1) % N, axis=ax)).mean() for i in range(N)]
    for per in (True, False):
        p = acc[per] / 24
        gap = abs(p[29] - 0.5 * (p[9] + p[19]))
        if per:
            assert gap < 0.01, f"periodic seam differs from lattice planes by {gap:.4f}"
        else:
            assert gap > 0.015, "non-periodic control should show a seam anomaly"


def test_gyroid_invariant_under_one_cell_shift():
    g = pg.generate_ordered_gyroid((48, 48, 48), n_cells=3, threshold=0.2, verbose=False)
    for ax in range(3):
        assert np.array_equal(g, np.roll(g, 16, axis=ax))


def test_gyroid_reports_cut_axes():
    with pytest.warns(RuntimeWarning):
        _, info = pg.generate_ordered_gyroid((40, 36, 40), domain_size=(1.0, 0.9, 1.0), n_cells=3,
                                             target_porosity=0.5, verbose=False, return_info=True)
    assert info["periodic_axes"] == [True, False, True]


def test_registry_matches_signatures():
    import inspect
    for name, v in pg.GENERATOR_INFO.items():
        params = inspect.signature(getattr(pg, v["function"])).parameters
        assert ("periodic" in params) == (v["periodic"] == "switch"), name
