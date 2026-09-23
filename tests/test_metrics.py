"""Metrics validated against geometries with analytic answers."""
import numpy as np
import pytest
from porousgen.metrics import compute_metrics, euler_characteristic


def _ball(n, R):
    x = np.indices((n,) * 3) - (n - 1) / 2
    return np.sqrt((x ** 2).sum(0)) <= R


def test_ball_area_and_estimator_bias():
    n, R = 80, 15.0
    m = compute_metrics(_ball(n, R).astype(np.int8), domain_size=(n,) * 3)
    true = 4 * np.pi * R ** 2
    assert abs(m["specific_surface_mesh"] * n ** 3 / true - 1) < 0.12
    assert abs(m["specific_surface_voxel"] * n ** 3 / true - 1.5) < 0.05   # Cauchy-Crofton bias
    assert abs(m["solid_thickness"]["d50_vox"] - 2 * R) <= 2.0            # ~1 voxel per side
    assert m["euler_fluid"] == 2


def test_disc_perimeter_2d():
    n, R = 100, 30
    x = np.indices((n, n)) - (n - 1) / 2
    disc = (np.sqrt((x ** 2).sum(0)) <= R).astype(np.int8)
    m = compute_metrics(disc, domain_size=(n, n))
    assert abs(m["specific_surface_mesh"] * n * n / (2 * np.pi * R) - 1) < 0.07
    assert abs(m["specific_surface_voxel"] * n * n / (2 * np.pi * R) - 4 / np.pi) < 0.03


def test_slab_pore_width_and_percolation():
    w = 12
    g = np.ones((60, 40, 40), np.int8)
    g[:, 14:14 + w, :] = 0
    m = compute_metrics(g)
    assert m["percolates"] and m["percolating_porosity"] == pytest.approx(m["porosity"])
    assert m["pore_size"]["d50_vox"] == pytest.approx(w, abs=0.5)


def test_enclosed_cavity_does_not_percolate():
    g = np.ones((40, 40, 40), np.int8)
    g[10:30, 10:30, 10:30] = 0
    m = compute_metrics(g, sizes=False)
    assert not m["percolates"] and m["percolating_porosity"] == 0.0
    assert m["isolated_fluid_fraction"] == 1.0


def test_flow_axis_matters():
    g = np.ones((30, 30, 30), np.int8)
    g[:, 10:20, 10:20] = 0            # channel along x only
    assert compute_metrics(g, flow_axis=0, sizes=False)["percolates"]
    assert not compute_metrics(g, flow_axis=1, sizes=False)["percolates"]


def test_torus_topology():
    n = 70
    x = np.indices((n,) * 3) - (n - 1) / 2
    torus = (np.sqrt(x[0] ** 2 + x[1] ** 2) - 18) ** 2 + x[2] ** 2 <= 36
    assert euler_characteristic(torus, 1) == 0
    assert euler_characteristic(~torus, 1) == 1


def test_floating_solid_detected():
    g = np.zeros((30, 30, 30), np.int8)
    g[:, :, :3] = 1                   # wall
    g[14:17, 14:17, 14:17] = 1        # detached fragment
    m = compute_metrics(g, sizes=False)
    assert m["n_solid_clusters"] == 2 and m["largest_solid_cluster_fraction"] < 1.0
