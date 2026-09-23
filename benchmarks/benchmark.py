"""
PorousGen performance benchmark: wall time and peak memory per generator
and grid resolution.

    python benchmarks/benchmark.py                          # 64^3 .. 512^3, all generators
    python benchmarks/benchmark.py --sizes 64 128 256 --generators foam gyroid
    python benchmarks/benchmark.py --dim 2 --sizes 256 1024 4096
    python benchmarks/benchmark.py --no-stl --timeout 3600

Every (generator, size) case runs in FRESH Python subprocesses, so the peak
memory of one case never inherits allocations from another, and in TWO
passes: a timing pass with no instrumentation (wall times and peak RSS),
then a separate memory pass under tracemalloc (peak traced allocations).
tracemalloc intercepts every allocation and can inflate the run time of
Python-object-heavy code by an order of magnitude, so it is never active
while anything is being timed. Per case:

  t_generate   wall time of the generator call
  t_dat        wall time of the plain-text .dat export
  t_stl        wall time of the watertight STL export (3-D only)
  peak_traced  peak memory allocated through Python/NumPy (tracemalloc),
               i.e. the working set of the algorithm itself
  peak_rss     peak resident set size of the whole process (includes the
               interpreter and imported libraries; Linux/macOS via
               resource.getrusage, Windows via psutil if installed)

Results: <out>.csv, <out>.md (table) and <out>_system.json (hardware and
software versions). A case that runs out of memory or time is recorded as
such instead of aborting the run.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

GENERATORS = {
    # name: (function, kwargs) - the CLI defaults on a unit box
    "granular":  ("generate_granular",            dict(radius=0.05, n_particles=120, seed=42)),
    "fibrous":   ("generate_fibrous",             dict(fiber_radius=0.02, n_fibers=120, seed=42)),
    "cellular":  ("generate_cellular",            dict(n_seeds=35, target_porosity=0.80, seed=42)),
    "fractured": ("generate_consolidated",        dict(n_fractures=8, fracture_width=0.02, seed=42)),
    "gyroid":    ("generate_ordered_gyroid",      dict(n_cells=3, target_porosity=0.70)),
    "foam":      ("generate_open_foam",           dict(porosity=0.90, n_cells=6, seed=42)),
    "blob":      ("generate_blob",                dict(porosity=0.60, correlation_length=0.08, seed=42)),
    "spheres":   ("generate_overlapping_spheres", dict(porosity=0.55, radius=0.07, seed=42)),
}


def system_info():
    import numpy, scipy
    info = dict(
        platform=platform.platform(), machine=platform.machine(),
        processor=platform.processor() or "unknown",
        python=platform.python_version(), numpy=numpy.__version__,
        scipy=scipy.__version__, logical_cpus=os.cpu_count(),
    )
    try:
        import skimage
        info["scikit_image"] = skimage.__version__
    except ImportError:
        pass
    try:
        import psutil
        info["ram_total_GB"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    try:
        import porousgen
        info["porousgen"] = porousgen.__version__
    except ImportError:
        pass
    return info


def _peak_rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2**20 if sys.platform == "darwin" else r / 1024   # bytes on macOS, KiB on Linux
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 2**20
        except Exception:
            return float("nan")


def worker(name, n, dim, stl, outdir, trace):
    """Runs ONE case (one pass); prints a JSON line."""
    import tracemalloc
    import porousgen as pg
    fn, kw = GENERATORS[name]
    shape = (n,) * dim
    rec = dict(generator=name, dim=dim, n=n, voxels=n ** dim)
    rec["rss_after_imports_MB"] = _peak_rss_mb()
    base = os.path.join(outdir, f"_bench_{name}_{n}")
    if trace:
        tracemalloc.start()
    t0 = time.perf_counter()
    g = getattr(pg, fn)(shape, verbose=False, **kw)
    rec["t_generate_s"] = time.perf_counter() - t0
    if trace:
        rec["peak_traced_generate_MB"] = tracemalloc.get_traced_memory()[1] / 2**20
    rec["porosity"] = float((g == 0).mean())
    t0 = time.perf_counter()
    pg.export_palabos(g, base + ".dat", verbose=False)
    rec["t_dat_s"] = time.perf_counter() - t0
    rec["dat_MB"] = os.path.getsize(base + ".dat") / 2**20
    os.remove(base + ".dat")
    if stl and dim == 3:
        t0 = time.perf_counter()
        pg.export_stl(g, base + ".stl", verbose=False, domain_size=(1.0,) * 3)
        rec["t_stl_s"] = time.perf_counter() - t0
        if os.path.exists(base + ".stl"):
            rec["stl_MB"] = os.path.getsize(base + ".stl") / 2**20
            os.remove(base + ".stl")
    if trace:
        rec["peak_traced_MB"] = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
    rec["peak_rss_MB"] = _peak_rss_mb()
    print("RESULT " + json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--dim", type=int, choices=(2, 3), default=3)
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS),
                    choices=list(GENERATORS))
    ap.add_argument("--no-stl", action="store_true")
    ap.add_argument("--timeout", type=float, default=7200, help="seconds per case")
    ap.add_argument("--out", default="benchmark_results")
    ap.add_argument("--worker", nargs=2, metavar=("GENERATOR", "N"), help=argparse.SUPPRESS)
    ap.add_argument("--trace", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    outdir = os.path.dirname(os.path.abspath(a.out)) or "."
    if a.worker:
        worker(a.worker[0], int(a.worker[1]), a.dim, not a.no_stl, outdir, a.trace)
        return

    sysinfo = system_info()
    with open(a.out + "_system.json", "w") as fh:
        json.dump(sysinfo, fh, indent=2)
    print(json.dumps(sysinfo, indent=2))
    rows = []
    for n in a.sizes:
        for name in a.generators:
            base_cmd = [sys.executable, os.path.abspath(__file__), "--worker", name, str(n),
                        "--dim", str(a.dim), "--out", a.out] + (["--no-stl"] if a.no_stl else [])
            t0 = time.perf_counter()
            rec = _run_pass(base_cmd, name, n, a)                    # timing pass
            if "status" not in rec:
                mem = _run_pass(base_cmd + ["--trace"], name, n, a)  # memory pass
                for k in ("peak_traced_generate_MB", "peak_traced_MB"):
                    rec[k] = mem.get(k, float("nan"))
            rec["t_case_total_s"] = time.perf_counter() - t0
            rows.append(rec)
            if "status" in rec:
                print(f"  {name:10s} {n:5d}^{a.dim}: {rec['status']}")
            else:
                print(f"  {name:10s} {n:5d}^{a.dim}: generate {rec['t_generate_s']:8.2f} s, "
                      f".dat {rec['t_dat_s']:7.2f} s, "
                      + (f"STL {rec.get('t_stl_s', float('nan')):7.2f} s, " if a.dim == 3 else "")
                      + f"peak {rec['peak_traced_MB']:8.1f} MB traced / {rec['peak_rss_MB']:8.1f} MB RSS")
            write_outputs(rows, a.out, sysinfo)


def _run_pass(cmd, name, n, a):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
    except subprocess.TimeoutExpired:
        return dict(generator=name, dim=a.dim, n=n, voxels=n ** a.dim,
                    status=f"timeout > {a.timeout:.0f} s")
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if line:
        return json.loads(line[-1][7:])
    err = (p.stderr.strip().splitlines() or ["unknown error"])[-1]
    oom = "MemoryError" in p.stderr or p.returncode in (-9, 137)
    return dict(generator=name, dim=a.dim, n=n, voxels=n ** a.dim,
                status="out of memory" if oom else err)


def write_outputs(rows, out, sysinfo):
    keys = ["generator", "dim", "n", "voxels", "t_generate_s", "t_dat_s", "t_stl_s",
            "peak_traced_generate_MB", "peak_traced_MB", "peak_rss_MB",
            "rss_after_imports_MB", "porosity",
            "dat_MB", "stl_MB", "status"]
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    L = [f"PorousGen {sysinfo.get('porousgen', '?')} benchmark - "
         f"{sysinfo.get('cpu_model', sysinfo.get('processor'))}, "
         f"{sysinfo.get('logical_cpus')} logical CPUs, "
         f"{sysinfo.get('ram_total_GB', '?')} GB RAM, {sysinfo['platform']}, "
         f"Python {sysinfo['python']}, NumPy {sysinfo['numpy']}, SciPy {sysinfo['scipy']}",
         "",
         "| generator | grid | generate [s] | .dat [s] | STL [s] | peak traced [MB] | peak RSS [MB] |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        grid = f"{r['n']}^{r['dim']}"
        if "status" in r:
            L.append(f"| {r['generator']} | {grid} | {r['status']} | | | | |")
            continue
        stl = f"{r['t_stl_s']:.2f}" if "t_stl_s" in r else "-"
        L.append(f"| {r['generator']} | {grid} | {r['t_generate_s']:.2f} | {r['t_dat_s']:.2f} | "
                 f"{stl} | {r['peak_traced_MB']:.0f} | {r['peak_rss_MB']:.0f} |")
    with open(out + ".md", "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
