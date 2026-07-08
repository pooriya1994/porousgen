"""
PyInstaller entry point.

cli.py uses a RELATIVE import ("from . import generator"), which only works
when Python loads it as part of the `porousgen` PACKAGE. PyInstaller's
Analysis(['porousgen/cli.py']) instead runs that file as a top-level SCRIPT,
so Python has no parent package and raises:
    ImportError: attempted relative import with no known parent package

Fix: build from THIS file instead. It sits outside the package, imports
porousgen properly (so the relative import inside cli.py resolves normally),
and simply calls main().
"""
import sys
from porousgen.cli import main

if __name__ == "__main__":
    sys.exit(main())
