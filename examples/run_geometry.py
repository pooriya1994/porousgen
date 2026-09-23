"""PorousGen examples — every call writes .dat + .stl + .png + _info.txt."""
from porousgen import (generate_open_foam, generate_blob, generate_fibrous,
                       generate_ordered_gyroid, export_all)

# 1) Open-cell Voronoi foam — porosity is honoured EXACTLY
foam = generate_open_foam(shape=(128, 128, 128), domain_size=(2e-3,)*3,
                          porosity=0.90, n_cells=6, seed=42)
export_all(foam, "foam_90", domain_size=(2e-3,)*3,
           params={"type": "open_foam", "porosity": 0.90, "n_cells": 6})

# 2) Stochastic blobs (Gaussian random field), 2-D
soil = generate_blob(shape=(256, 256), domain_size=(2.0, 2.0),
                     porosity=0.60, correlation_length=0.08, seed=7)
export_all(soil, "soil_2d", domain_size=(2.0, 2.0),
           params={"type": "blob", "porosity": 0.60})

# 3) Fibrous medium (as in the original quick-start)
fib = generate_fibrous(shape=(256, 256), domain_size=(2.0, 2.0),
                       fiber_radius=0.025, fiber_length=0.20,
                       n_fibers=230, seed=713)
export_all(fib, "fibrous_2d", domain_size=(2.0, 2.0),
           params={"type": "fibrous", "n_fibers": 230})

# 4) Gyroid TPMS at 70 % porosity
tpms = generate_ordered_gyroid(shape=(96, 96, 96), n_cells=3, target_porosity=0.70)
export_all(tpms, "gyroid_70", params={"type": "gyroid", "porosity": 0.70})
