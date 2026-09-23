import os
import shutil
import subprocess
import sys
import pytest


def _cmd():
    exe = shutil.which("porousgen")          # lower-case console script
    return [exe] if exe else [sys.executable, "-m", "porousgen.cli"]


def test_version():
    import porousgen
    out = subprocess.run(_cmd() + ["--version"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == porousgen.__version__


@pytest.mark.parametrize("args", [
    ["foam", "--resolution", "20", "--cells", "2", "--porosity", "0.85"],
    ["foam", "--resolution", "20", "--cells", "2", "--periodic", "--metrics"],
    ["fibrous", "--resolution", "20", "--fibers", "10", "--porosity", "0.8", "--dim", "2"],
    ["gyroid", "--resolution", "24", "--cells", "2"],
    ["fractured", "--resolution", "20", "--porosity", "0.3"],
])
def test_generate_and_analyze(tmp_path, args):
    base = str(tmp_path / "g")
    subprocess.run(_cmd() + args + ["--out", base, "--no-stl"], check=True, capture_output=True)
    assert os.path.exists(base + ".dat") and os.path.exists(base + "_info.txt")
    out = subprocess.run(_cmd() + ["analyze", base + ".dat"], check=True,
                         capture_output=True, text=True).stdout
    assert "percolating porosity" in out


def test_list():
    out = subprocess.run(_cmd() + ["list"], check=True, capture_output=True, text=True).stdout
    assert "gyroid" in out and "intrinsic" in out
