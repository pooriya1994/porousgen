#!/usr/bin/env bash
# Open-cell foam porosity sweep of the article (Section 3), exactly as printed
# in the text. Then render the figure from these very files.
set -e
porousgen foam --resolution 128 --dim 3 --porosity 0.85 --cells 6 --out foam_85
porousgen foam --resolution 128 --dim 3 --porosity 0.90 --cells 6 --out foam_90
porousgen foam --resolution 128 --dim 3 --porosity 0.95 --cells 6 --out foam_95
python "$(dirname "$0")/plot_porosity_sweep.py" foam_85.dat foam_90.dat foam_95.dat
