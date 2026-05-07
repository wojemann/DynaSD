"""Bundled example data for quickstart and demos."""

from importlib.resources import files
from typing import NamedTuple

import numpy as np
import pandas as pd


class ExampleSeizure(NamedTuple):
    """Bundled synthetic seizure fixture.

    Attributes
    ----------
    signal : pandas.DataFrame
        Multichannel iEEG-like signal, shape ``(n_samples, n_channels)``.
    fs : int
        Sampling frequency in Hz.
    seizure_start_sec : float
        Onset time of the planted seizure in seconds.
    seizure_end_sec : float
        End time of the planted seizure in seconds.
    focal_channels : list[str]
        Channel names that carry the planted seizure (others stay baseline).
    """
    signal: pd.DataFrame
    fs: int
    seizure_start_sec: float
    seizure_end_sec: float
    focal_channels: list


def load_example_seizure() -> ExampleSeizure:
    """Load the bundled synthetic seizure fixture used by the quickstart.

    Returns
    -------
    ExampleSeizure
        Named tuple with the multichannel signal, sampling rate, and
        ground-truth onset/offset/focal-channel labels.
    """
    path = files(__package__) / "example_seizure.npz"
    with path.open("rb") as f:
        data = np.load(f, allow_pickle=False)
        signal = data["signal"]
        fs = int(data["fs"])
        ch_names = [str(c) for c in data["channel_names"]]
        sz_start = float(data["seizure_start_sec"])
        sz_end = float(data["seizure_end_sec"])
        focal_idx = [int(i) for i in data["focal_channels"]]

    df = pd.DataFrame(signal, columns=ch_names)
    focal_channels = [ch_names[i] for i in focal_idx]
    return ExampleSeizure(
        signal=df,
        fs=fs,
        seizure_start_sec=sz_start,
        seizure_end_sec=sz_end,
        focal_channels=focal_channels,
    )
