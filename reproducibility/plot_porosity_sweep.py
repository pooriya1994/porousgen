"""
Render the porosity-sweep figure from the .dat files written by
porosity_sweep.sh: the mid-plane section of each 128^3 volume, titled with
the requested porosity and the value measured on the WHOLE volume.

    python plot_porosity_sweep.py foam_85.dat foam_90.dat foam_95.dat
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import porousgen as pg

files = sys.argv[1:] or ["foam_85.dat", "foam_90.dat", "foam_95.dat"]
fig, axes = plt.subplots(1, len(files), figsize=(4.6 * len(files), 4.9), constrained_layout=True)
for ax, f in zip(axes, files):
    g = pg.read_palabos(f)
    target = int(f.split("_")[-1].split(".")[0]) / 100
    ax.imshow(g[:, :, g.shape[2] // 2].T, origin="lower", cmap="gray_r", interpolation="nearest")
    ax.set_title(f"target \u03c6 = {target:.2f}\nmeasured \u03c6 = {(g == 0).mean():.4f}  "
                 f"({(g == 0).sum():,} of {g.size:,} voxels)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Open-cell foam, 128\u00b3 voxels, 6 cells per axis (mid-plane z-sections)", fontsize=11)
fig.savefig("porosity_sweep.png", dpi=200, bbox_inches="tight", facecolor="white")
print("written porosity_sweep.png")
