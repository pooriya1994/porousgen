# PyInstaller spec — build on Windows x64:  pyinstaller porousgen.spec
#
# IMPORTANT: the entry point is run_porousgen.py (NOT porousgen/cli.py).
# cli.py contains a relative import ("from . import generator") that only
# resolves when porousgen is loaded as a package; pointing PyInstaller
# directly at cli.py runs it as a standalone script instead, which breaks
# that import (ImportError: attempted relative import with no known
# parent package). run_porousgen.py imports the package correctly and
# just calls main().
from PyInstaller.utils.hooks import collect_submodules
a = Analysis(['run_porousgen.py'],
             pathex=['.'],
             hiddenimports=(['porousgen', 'porousgen.generator', 'porousgen.cli',
                             'porousgen._core', 'porousgen.metrics',
                             'mpl_toolkits.mplot3d', 'matplotlib.backends.backend_agg']
                            + collect_submodules('scipy')
                            + collect_submodules('skimage')
                            + collect_submodules('mpl_toolkits')),
             datas=[], hookspath=[], excludes=['tkinter'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas,
          name='porousgen', console=True, onefile=True)
