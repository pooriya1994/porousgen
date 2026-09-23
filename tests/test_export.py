import json
import struct
import numpy as np
import pytest
import porousgen as pg

STL_REC = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])


@pytest.mark.parametrize("shape", [(7, 5), (6, 5, 4), (1, 3, 2)])
def test_dat_round_trip(tmp_path, shape):
    g = (np.random.default_rng(0).random(shape) < 0.4).astype(np.int8)
    f = tmp_path / "g.dat"
    pg.export_palabos(g, str(f), verbose=False)
    hint = shape if shape[0] == 1 else None      # nx = 1 is ambiguous with 2-D
    assert np.array_equal(pg.read_palabos(str(f), shape=hint), g)
    text = f.read_text()
    assert set(text.split()) <= {"0", "1"}                    # plain text, not binary
    if len(shape) == 3:
        assert text.count("\n\n") == shape[0] - 1             # blank line between x-slices


def _load_stl(path):
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[80:84])[0]
    assert len(raw) == 84 + 50 * n
    return np.frombuffer(raw[84:], dtype=STL_REC)


def _edge_use_counts(rec, quant):
    q = np.round(rec["v"] / quant).astype(np.int64)
    edges = {}
    for t in q:
        p = [tuple(x) for x in t]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = tuple(sorted((p[a], p[b])))
            edges[e] = edges.get(e, 0) + 1
    return np.array(list(edges.values()))


def test_stl_is_watertight_and_in_physical_units(tmp_path):
    L = (0.012, 0.010, 0.008)
    g = pg.generate_open_foam((36, 30, 24), domain_size=L, porosity=0.85, n_cells=3, seed=2, verbose=False)
    f = tmp_path / "f.stl"
    pg.export_stl(g, str(f), verbose=False, domain_size=L)
    rec = _load_stl(f)
    v = rec["v"].reshape(-1, 3)
    assert np.all(v.min(0) >= -1e-9) and np.all(v.max(0) <= np.array(L) + 1e-9)
    assert np.allclose(v.max(0), L, rtol=0.02)       # solid touches the far faces
    assert np.all(_edge_use_counts(rec, 1e-7) == 2)  # every edge in exactly 2 triangles


def test_stl_open_mode_reproduces_v10_surface(tmp_path):
    g = pg.generate_open_foam((24, 24, 24), porosity=0.85, n_cells=2, seed=2, verbose=False)
    pg.export_stl(g, str(tmp_path / "o.stl"), verbose=False, closed=False)
    counts = _edge_use_counts(_load_stl(tmp_path / "o.stl"), 1e-4)
    assert np.any(counts == 1)                       # open at the domain faces


def test_export_all_bundle(tmp_path):
    L = (0.01,) * 3
    g, info = pg.generate_blob((24, 24, 24), domain_size=L, porosity=0.6, seed=1,
                               correlation_length=0.001, verbose=False, return_info=True)
    out = pg.export_all(g, str(tmp_path / "b"), L, dict(type="blob"), info=info,
                        metrics=True, verbose=False)
    assert set(out) == {"dat", "stl", "png", "info", "metrics"}
    txt = open(out["info"]).read()
    assert f"fluid voxels : {int((g == 0).sum()):,}" in txt
    assert "plain-text" in txt and "periodic" in txt
    m = json.load(open(out["metrics"]))
    assert m["n_fluid"] == int((g == 0).sum())


def test_blob_large_correlation_length_is_bounded(tmp_path):
    """Physical units with the default correlation_length (0.08 on a 1 cm box)
    must warn, not exhaust memory."""
    with pytest.warns(RuntimeWarning, match="correlation_length"):
        g = pg.generate_blob((24, 24, 24), domain_size=(0.01,) * 3, porosity=0.5, verbose=False)
    assert g.size - int((g == 0).sum()) == round(0.5 * g.size)
