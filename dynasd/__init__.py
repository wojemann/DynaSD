"""DynaSD — dynamic seizure detection models and utilities for iEEG analysis.

Core install (`pip install dynasd`) ships the closed-form detectors
(``ABSSLP``, ``HFER``, ``IMPRINT``), the shared base class, and the
quickstart fixture loader without pulling in heavy ML frameworks.

Detectors that require optional frameworks are lazy-loaded:

- ``NDD``, ``GIN``, ``LiNDDA``, ``NDDBase``, ``ONCET`` require ``torch``;
  install via ``pip install "dynasd[torch]"``.
- ``WVNT`` requires ``tensorflow``; install via
  ``pip install "dynasd[tensorflow]"``.

Accessing one of these names without the corresponding extra installed
raises an ``ImportError`` naming the missing extra.
"""

from importlib import import_module as _import_module

# Eager imports: the core, framework-free surface.
from .base import DynaSDBase
from .ABSSLP import ABSSLP
from .HFER import HFER
from .IMPRINT import IMPRINT
from .data import load_example_seizure

__version__ = "0.1.0"

# Lazy attribute → (module_name, extra). When the user accesses a lazy
# name, we import the corresponding module on demand. If the import
# fails because the optional framework isn't installed, we re-raise with
# an actionable message naming the extra.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "NDD":     (".NDD",     "torch"),
    "GIN":     (".GIN",     "torch"),
    "LiNDDA":  (".LiNDDA",  "torch"),
    "NDDBase": (".NDDBase", "torch"),
    "ONCET":   (".ONCET",   "torch"),
    "WVNT":    (".WAVENET", "tensorflow"),
}

__all__ = [
    "DynaSDBase",
    "ABSSLP",
    "HFER",
    "IMPRINT",
    "load_example_seizure",
    *_LAZY_ATTRS.keys(),
]


def __getattr__(name: str):
    """PEP 562 lazy attribute access.

    Loads optional-framework detectors on first access and raises a
    helpful ``ImportError`` if the relevant extra isn't installed.
    """
    if name in _LAZY_ATTRS:
        module_name, extra = _LAZY_ATTRS[name]
        try:
            module = _import_module(module_name, __package__)
        except ImportError as e:
            raise ImportError(
                f"`from dynasd import {name}` requires the optional "
                f"`{extra}` extra. Install with: "
                f"pip install 'dynasd[{extra}]'"
            ) from e
        attr = getattr(module, name)
        globals()[name] = attr  # cache so subsequent accesses skip __getattr__
        return attr
    raise AttributeError(f"module 'dynasd' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
