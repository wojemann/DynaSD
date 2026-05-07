"""Download-on-first-use helper for bundled checkpoint files.

Pretrained checkpoints are not shipped in the wheel. They are hosted as
assets on a GitHub Release and fetched into a per-user cache directory
on first use. Subsequent calls load from the cache.

To override the cache directory set ``DYNASD_CACHE_DIR``. To skip the
download path entirely, pass an explicit ``checkpoint_path`` /
``model_path`` to the detector constructor.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

__all__ = ["fetch_checkpoint", "cache_dir", "CHECKPOINTS"]


@dataclass(frozen=True)
class _Checkpoint:
    """Manifest entry for one downloadable checkpoint file."""
    relpath: str       # path within the cache dir, e.g. "ONCET/best_model.pth"
    url: str           # canonical download URL
    sha256: str        # hex-encoded SHA-256 of the expected payload
    size_bytes: int    # for the progress bar / sanity check


# Pinned to the v0.1.0-checkpoints release. Bump the tag + sha256 here
# when re-training a checkpoint.
_RELEASE_TAG = "v0.1.0-checkpoints"
_RELEASE_BASE = (
    f"https://github.com/wojemann/DynaSD/releases/download/{_RELEASE_TAG}"
)

CHECKPOINTS: dict[str, _Checkpoint] = {
    "ONCET/best_model.pth": _Checkpoint(
        relpath="ONCET/best_model.pth",
        url=f"{_RELEASE_BASE}/best_model.pth",
        sha256="34afd9c56ccd11d66f97bd224e3e0447ceae07ac9b3e503c30939d37f59a5c2f",
        size_bytes=522_240,
    ),
    "ONCET/final_training_config.json": _Checkpoint(
        relpath="ONCET/final_training_config.json",
        url=f"{_RELEASE_BASE}/final_training_config.json",
        sha256="0481bbcc0e69c855112e5aa7d56737725f05662dd35162952d63983ca5e07dda",
        size_bytes=1_843,
    ),
    "WVNT/v111.hdf5": _Checkpoint(
        relpath="WVNT/v111.hdf5",
        url=f"{_RELEASE_BASE}/v111.hdf5",
        sha256="6331140afdc75100c101e9596bf06f1365d5bfca30e53fb361c6af3701a795ea",
        size_bytes=7_447_104,
    ),
}


def cache_dir() -> Path:
    """Return the per-user cache directory for DynaSD checkpoints.

    Honors ``DYNASD_CACHE_DIR``, then platform conventions (XDG on
    Linux, ``~/Library/Caches`` on macOS, ``%LOCALAPPDATA%`` on Windows),
    falling back to ``~/.cache/dynasd``.
    """
    env = os.environ.get("DYNASD_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "dynasd"
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "dynasd" / "Cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "dynasd"


def _sha256(path: Path, chunk_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, expected_size: int, *, verbose: bool) -> None:
    """Stream ``url`` to ``dst`` atomically, with optional progress output."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".partial-", dir=str(dst.parent))
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    try:
        if verbose:
            print(f"DynaSD: downloading {url} → {dst}", file=sys.stderr)
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            written = 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if verbose and expected_size:
                    pct = min(100, int(100 * written / expected_size))
                    print(
                        f"\r  {written / 1e6:5.1f} / {expected_size / 1e6:5.1f} MB ({pct}%)",
                        end="", file=sys.stderr,
                    )
            if verbose:
                print("", file=sys.stderr)
        shutil.move(str(tmp), str(dst))
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def fetch_checkpoint(name: str, *, verbose: bool = True) -> Path:
    """Return the cached path for ``name``, downloading on first use.

    Parameters
    ----------
    name : str
        Manifest key, e.g. ``"ONCET/best_model.pth"``.
    verbose : bool, default True
        Print a progress line to ``stderr`` while downloading.

    Returns
    -------
    pathlib.Path
        Verified path to the cached checkpoint.

    Raises
    ------
    KeyError
        If ``name`` is not a known checkpoint.
    RuntimeError
        If the download fails or the SHA-256 does not match.
    """
    try:
        spec = CHECKPOINTS[name]
    except KeyError:
        known = ", ".join(sorted(CHECKPOINTS))
        raise KeyError(
            f"Unknown checkpoint {name!r}. Known checkpoints: {known}"
        ) from None

    target = cache_dir() / spec.relpath

    if target.exists():
        actual = _sha256(target)
        if actual == spec.sha256:
            return target
        if verbose:
            print(
                f"DynaSD: cached {target} has wrong SHA-256 "
                f"(got {actual}, expected {spec.sha256}); re-downloading.",
                file=sys.stderr,
            )
        target.unlink()

    try:
        _download(spec.url, target, spec.size_bytes, verbose=verbose)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to download {spec.url}: {e}. "
            f"You can manually place a file with SHA-256 {spec.sha256} at "
            f"{target} and DynaSD will pick it up."
        ) from e

    actual = _sha256(target)
    if actual != spec.sha256:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {spec.url} but SHA-256 did not match "
            f"(got {actual}, expected {spec.sha256}). The release asset "
            f"may have been replaced; please file an issue."
        )
    return target
