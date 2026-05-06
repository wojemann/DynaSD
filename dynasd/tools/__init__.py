"""
DynaSD researcher tooling.

Modules in :mod:`dynasd.tools` provide helpers used *around* the detection
models — for loading data, preprocessing iEEG signals, plotting, and basic
statistics — but not by the model classes themselves at runtime. Imports
here may carry heavier optional dependencies (matplotlib, ieeg, scipy
filters); none of them are required for ``import dynasd`` to succeed.

Submodules:

- :mod:`dynasd.tools.io`            — iEEG data loading and config helpers
- :mod:`dynasd.tools.preprocessing` — channel and signal preprocessing
- :mod:`dynasd.tools.viz`           — multi-channel iEEG plotting
- :mod:`dynasd.tools.stats`         — small statistical helpers
"""
