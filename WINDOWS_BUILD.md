# Installing / building on Windows x64

## Option A — pip install (recommended, 2 minutes)
Works on any Windows x64 with Python 3.9+ (python.org or Anaconda):

    cd porousgen-package
    pip install .

Then from any terminal:

    porousgen foam --resolution 128 --porosity 0.90 --cells 6 --out my_foam

This writes my_foam.dat / .stl / .png / _info.txt in the current folder.

## Option B — standalone .exe (no Python needed on target PC)
Build ONCE on a Windows x64 machine (PyInstaller cannot cross-compile,
so this step must run on Windows):

    pip install . pyinstaller
    pyinstaller porousgen.spec

(The spec builds from `run_porousgen.py`, a small wrapper needed because
`porousgen/cli.py` uses a relative import that only resolves when PyInstaller
loads it as part of the package, not as a bare script.)

The single-file executable appears at dist\porousgen.exe and can be copied
to any Windows x64 computer:

    porousgen.exe foam --resolution 128 --porosity 0.90 --out my_foam

Note: the exe is ~150-250 MB because it bundles numpy/scipy/matplotlib.
