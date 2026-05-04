"""
DynaSD researcher tooling.

Modules in :mod:`DynaSD.tools` provide helpers used *around* the detection
models — for loading data, preprocessing iEEG signals, plotting, and basic
statistics — but not by the model classes themselves at runtime. Imports
here may carry heavier optional dependencies (matplotlib, ieeg, scipy
filters); none of them are required for ``import DynaSD`` to succeed.

Submodules:

- :mod:`DynaSD.tools.io`            — iEEG data loading and config helpers
- :mod:`DynaSD.tools.preprocessing` — channel and signal preprocessing
- :mod:`DynaSD.tools.viz`           — multi-channel iEEG plotting
- :mod:`DynaSD.tools.stats`         — small statistical helpers
"""
