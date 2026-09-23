"""
Regression against the committed reference outputs (tests/reference/).

Porosity-exact cases must match the reference fluid-voxel count exactly on
every platform. Voxel-by-voxel agreement is required up to a small
tolerance, because transcendental functions (sin/cos) and summation order
may differ in the last bit across operating systems / BLAS builds, which
can move a voxel within a tie set at the cut-off value.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from make_reference import CASES, build, sha256  # noqa: E402

REF = os.path.join(HERE, "reference")
MANIFEST = json.load(open(os.path.join(REF, "manifest.json")))["cases"]
GRIDS = np.load(os.path.join(REF, "reference_grids.npz"))


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_reference(case):
    ref_meta = MANIFEST[case]
    g = build(case)
    assert list(g.shape) == ref_meta["shape"]
    ref = np.unpackbits(GRIDS[case])[: g.size].reshape(g.shape).astype(np.int8)
    exact = ("porosity" in ref_meta["kwargs"] or "target_porosity" in ref_meta["kwargs"])
    if exact:
        assert int((g == 0).sum()) == ref_meta["n_fluid"]
    if sha256(g) != ref_meta["sha256"]:
        mismatch = float((g != ref).mean())
        assert mismatch <= 2e-3, f"{case}: {mismatch:.2e} of voxels differ from reference"
