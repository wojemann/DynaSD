"""
Channel and signal preprocessing for iEEG data.

These helpers run *before* model fitting: bad-channel detection, channel
type inference, bipolar-montage construction, downsampling, notch and
bandpass filtering, and an AR(1) prewhitening step. They are not imported
by any model class at runtime; users wire them into their own preprocessing
pipelines.
"""

import re

import numpy as np
import pandas as pd
import scipy as sc
from scipy.signal import butter, filtfilt, sosfiltfilt


def detect_bad_channels(data, fs, lf_stim=False):
    """Identify bad channels in raw iEEG data.

    Returns ``(channel_mask, details)`` where ``channel_mask`` is a boolean
    array (True = good) and ``details`` is a dict with category-specific
    rejection lists.
    """
    values = data.copy()
    which_chs = np.arange(values.shape[1])
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 5e3
    percent_60_hz = 0.7
    mult_std = 10

    bad = []
    high_ch = []
    nan_ch = []
    zero_ch = []
    flat_ch = []
    high_var_ch = []
    noisy_ch = []
    all_std = np.empty((len(which_chs), 1))
    all_std[:] = np.nan
    details = {}

    for i in range(len(which_chs)):
        ich = which_chs[i]
        eeg = values[:, ich]
        bl = np.nanmedian(eeg)
        all_std[i] = np.nanstd(eeg)

        if sum(np.isnan(eeg)) > 0.5 * len(eeg):
            bad.append(ich)
            nan_ch.append(ich)
            continue

        if sum(eeg == 0) > (0.5 * len(eeg)):
            bad.append(ich)
            zero_ch.append(ich)
            continue

        if (sum(np.diff(eeg, 1) == 0) > (0.02 * len(eeg))) and (
            sum(abs(eeg - bl) > abs_thresh) > (0.02 * len(eeg))
        ):
            bad.append(ich)
            flat_ch.append(ich)

        if sum(abs(eeg - bl) > abs_thresh) > 10:
            if not lf_stim:
                bad.append(ich)
            high_ch.append(ich)
            continue

        pct = np.percentile(eeg, [100 - tile, tile])
        thresh = [bl - mult * (bl - pct[0]), bl + mult * (pct[1] - bl)]
        sum_outside = sum(((eeg > thresh[1]) + (eeg < thresh[0])) > 0)
        if sum_outside >= num_above:
            if not lf_stim:
                bad.append(ich)
            high_var_ch.append(ich)
            continue

        Y = np.fft.fft(eeg - np.nanmean(eeg))
        P = abs(Y) ** 2
        freqs = np.linspace(0, fs, len(P) + 1)
        freqs = freqs[:-1]
        P = P[: np.ceil(len(P) / 2).astype(int)]
        freqs = freqs[: np.ceil(len(freqs) / 2).astype(int)]
        P_60Hz = sum(P[(freqs > 58) * (freqs < 62)]) / sum(P)
        if P_60Hz > percent_60_hz:
            bad.append(ich)
            noisy_ch.append(ich)
            continue

    median_std = np.nanmedian(all_std)
    higher_std = which_chs[(all_std > (mult_std * median_std)).squeeze()]
    bad_std = higher_std

    channel_mask = np.ones((values.shape[1],), dtype=bool)
    channel_mask[bad] = False
    details["noisy"] = noisy_ch
    details["nans"] = nan_ch
    details["zeros"] = zero_ch
    details["flat"] = flat_ch
    details["var"] = high_var_ch
    details["higher_std"] = bad_std
    details["high_voltage"] = high_ch

    return channel_mask, details


def check_channel_types(ch_list, threshold=16):
    """Classify channel labels as ``ecg``, ``eeg``, ``ecog``, ``seeg``, or ``misc``.

    Returns a DataFrame with columns ``name``, ``lead``, ``contact``, ``type``.
    """
    ch_df = []
    for i in ch_list:
        regex_match = re.match(r"([A-Za-z0-9]+)(\d{2})$", i)
        if regex_match is None:
            ch_df.append({"name": i, "lead": i, "contact": 0, "type": "misc"})
            continue
        lead = regex_match.group(1)
        contact = int(regex_match.group(2))
        ch_df.append({"name": i, "lead": lead, "contact": contact})
    ch_df = pd.DataFrame(ch_df)
    for lead, group in ch_df.groupby("lead"):
        if lead in ["ECG", "EKG"]:
            ch_df.loc[group.index, "type"] = "ecg"
            continue
        if lead in [
            "C", "Cz", "CZ",
            "F", "Fp", "FP", "Fz", "FZ",
            "O", "P", "Pz", "PZ", "T",
        ]:
            ch_df.loc[group.index.to_list(), "type"] = "eeg"
            continue
        if len(group) > threshold:
            ch_df.loc[group.index.to_list(), "type"] = "ecog"
        else:
            ch_df.loc[group.index.to_list(), "type"] = "seeg"
    return ch_df


def bipolar_montage(data, ch_types):
    """Apply a bipolar montage to ``data`` (channels × time).

    Returns ``(new_data, new_ch_types)`` where ``new_ch_types`` is a DataFrame
    listing the bipolar pair names and source-channel indices.
    """
    new_ch_types = []
    for ind, row in ch_types.iterrows():
        if row["type"] not in ["ecog", "seeg"]:
            continue

        ch1 = row["name"]
        ch2 = ch_types.loc[
            (ch_types["lead"] == row["lead"])
            & (ch_types["contact"] == row["contact"] + 1),
            "name",
        ]
        if len(ch2) > 0:
            ch2 = ch2.iloc[0]
            entry = {
                "name": ch1 + "-" + ch2,
                "type": row["type"],
                "idx1": ind,
                "idx2": ch_types.loc[ch_types["name"] == ch2].index[0],
            }
            new_ch_types.append(entry)

    new_ch_types = pd.DataFrame(new_ch_types)
    new_data = np.empty((len(new_ch_types), data.shape[1]))
    for ind, row in new_ch_types.iterrows():
        new_data[ind, :] = data[row["idx1"], :] - data[row["idx2"], :]

    return new_data, new_ch_types


def downsample(data, fs, target):
    """Resample ``data`` along axis 0 from ``fs`` to ``target`` Hz."""
    signal_len = int(data.shape[0] / fs * target)
    data_bpd = sc.signal.resample(data, signal_len, axis=0)
    return data_bpd, target


def notch_filter(data, fs):
    """Apply 60Hz and 120Hz notch filters along the last axis."""
    b, a = butter(4, (58, 62), "bandstop", fs=fs)
    d, c = butter(4, (118, 122), "bandstop", fs=fs)
    data_filt = filtfilt(b, a, data, axis=-1)
    return filtfilt(d, c, data_filt, axis=-1)


def bandpass_filter(data, fs, order=3, lo=1, hi=150):
    """Bandpass-filter ``data`` along the last axis from ``lo`` to ``hi`` Hz."""
    sos = butter(order, [lo, hi], output="sos", fs=fs, btype="bandpass")
    return sosfiltfilt(sos, data, axis=-1)


def ar_one(data):
    """Fit an AR(1) model and return the residual ("prewhitened") signal.

    Parameters
    ----------
    data : ndarray, shape (T, N)
        Input signal with ``T`` samples over ``N`` variates.

    Returns
    -------
    ndarray, shape (T-1, N)
        Whitened signal with reduced autocorrelative structure.
    """
    n_samp, n_chan = data.shape
    data_white = np.zeros((n_samp - 1, n_chan))
    for i in range(n_chan):
        win_x = np.vstack((data[:-1, i], np.ones(n_samp - 1)))
        w = np.linalg.lstsq(win_x.T, data[1:, i], rcond=None)[0]
        data_white[:, i] = data[1:, i] - (data[:-1, i] * w[0] + w[1])
    return data_white


def preprocess_for_detection(
    data, fs, montage="bipolar", target=256, wavenet=False, pre_mask=None
):
    """Apply the full preprocessing pipeline used by detection models.

    Steps: montage → bad-channel rejection → notch → bandpass → resample →
    AR(1) prewhitening. Returns ``(df, fsd, mask_list)`` when ``pre_mask`` is
    None, otherwise ``(df, fsd)``.
    """
    chs = data.columns.to_list()
    ch_df = check_channel_types(chs)

    if montage == "bipolar":
        data_bp_np, bp_ch_df = bipolar_montage(data.to_numpy().T, ch_df)
        bp_ch = bp_ch_df.name.to_numpy()
    elif montage == "car":
        data_bp_np = data.to_numpy().T - np.mean(data.to_numpy(), 1)
        bp_ch = chs

    if pre_mask is None:
        mask, _ = detect_bad_channels(data_bp_np.T * 1e3, fs)
        data_bp_np = data_bp_np[mask, :]
        mask_list = [ch for ch in bp_ch[~mask]]
        bp_ch = bp_ch[mask]
    else:
        mask = np.atleast_1d([ch not in pre_mask for ch in bp_ch])
        data_bp_np = data_bp_np[mask, :]
        bp_ch = bp_ch[mask]

    if wavenet:
        target = 128
        data_bp_notch = notch_filter(data_bp_np, fs)
        data_bp_filt = bandpass_filter(data_bp_notch, fs, lo=3, hi=127)
    else:
        data_bp_notch = notch_filter(data_bp_np, fs)
        data_bp_filt = bandpass_filter(data_bp_notch, fs, lo=3, hi=150)

    signal_len = int(data_bp_filt.shape[1] / fs * target)
    data_bpd = sc.signal.resample(data_bp_filt, signal_len, axis=1).T
    fsd = int(target)
    data_white = ar_one(data_bpd)
    data_white_df = pd.DataFrame(data_white, columns=bp_ch)
    if pre_mask is None:
        return data_white_df, fsd, mask_list
    else:
        return data_white_df, fsd
