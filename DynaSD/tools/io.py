"""
Data loading and configuration helpers.

This module contains code that touches the iEEG.org data service and project
config files. It depends on the optional ``ieeg`` Python client; that
import is performed lazily inside :func:`get_ieeg_data` so users without
the client installed can still ``import DynaSD.tools.io``.
"""

import json
import logging
import pickle
import re
import time
from numbers import Number

import numpy as np
import pandas as pd


def load_config(config_path):
    """Load a DynaSD project config JSON file and return key paths."""
    with open(config_path, "r") as f:
        config = json.load(f)
    usr = config["paths"]["iEEG_USR"]
    passpath = config["paths"]["iEEG_PWD"]
    datapath = config["paths"]["RAW_DATA"]
    prodatapath = config["paths"]["PROCESSED_DATA"]
    figpath = config["paths"]["FIGURES"]
    metapath = config["paths"]["METADATA"]
    return usr, passpath, datapath, prodatapath, metapath, figpath


def clean_labels(channel_li, pt):
    """Standardize channel labels for a given patient identifier.

    Returns a list of cleaned channel names, applying patient-specific
    remappings encountered in HUP/RID datasets.
    """
    new_channels = []
    for i in channel_li:
        i = i.replace("-", "")
        i = i.replace("GRID", "G")  # mne has limits on channel name size
        pattern = re.compile(r"([A-Za-z0-9]+?)(\d+)$")
        regex_match = pattern.match(i)

        if regex_match is None:
            new_channels.append(i)
            continue

        lead = regex_match.group(1).replace("EEG", "").strip()
        contact = int(regex_match.group(2))
        if pt in ("HUP75_phaseII", "HUP075", "sub-RID0065"):
            if lead == "Grid":
                lead = "G"

        if pt in ("HUP78_phaseII", "HUP078", "sub-RID0068"):
            if lead == "Grid":
                lead = "LG"

        if pt in ("HUP86_phaseII", "HUP086", "sub-RID0018"):
            conv_dict = {
                "AST": "LAST",
                "DA": "LA",
                "DH": "LH",
                "Grid": "LG",
                "IPI": "LIPI",
                "MPI": "LMPI",
                "MST": "LMST",
                "OI": "LOI",
                "PF": "LPF",
                "PST": "LPST",
                "SPI": "RSPI",
            }
            if lead in conv_dict:
                lead = conv_dict[lead]

        if pt in ("HUP93_phaseII", "HUP093", "sub-RID0050"):
            if lead.startswith("G"):
                lead = "G"

        if pt in ("HUP89_phaseII", "HUP089", "sub-RID0024"):
            if lead in ("GRID", "G"):
                lead = "RG"
            if lead == "AST":
                lead = "AS"
            if lead == "MST":
                lead = "MS"

        if pt in ("HUP99_phaseII", "HUP099", "sub-RID0032"):
            if lead == "G":
                lead = "RG"

        if pt in ("HUP112_phaseII", "HUP112", "sub-RID0042"):
            if "-" in i:
                new_channels.append(f"{lead}{contact:02d}-{i.strip().split('-')[-1]}")
                continue
        if pt in ("HUP116_phaseII", "HUP116", "sub-RID0175"):
            new_channels.append(f"{lead}{contact:02d}".replace("-", ""))
            continue

        if pt in ("HUP119_phaseII", "HUP119"):
            if i == "LG7":
                continue

        if pt in ("HUP123_phaseII_D02", "HUP123", "sub-RID0193"):
            if lead == "RS":
                lead = "RSO"
            if lead == "GTP":
                lead = "RG"

        new_channels.append(f"{lead}{contact:02d}")

        if pt in ("HUP189", "HUP189_phaseII", "sub-RID0520"):
            conv_dict = {"LG": "LGr"}
            if lead in conv_dict:
                lead = conv_dict[lead]

    return new_channels


def remove_scalp_electrodes(raw_labels):
    """Filter scalp/non-iEEG electrode labels from a list of channel names."""
    scalp_list = [
        "CZ", "FZ", "PZ",
        "A01", "A02",
        "C03", "C04",
        "F03", "F04", "F07", "F08",
        "FP01", "FP02",
        "O01", "O02",
        "P03", "P04",
        "T03", "T04", "T05", "T06",
        "EKG01", "EKG02",
        "ECG01", "ECG02",
        "ROC", "LOC",
        "EMG01", "EMG02",
        "DC01", "DC07",
    ]
    chop_scalp = ["C1" + str(x) for x in range(19, 29)]
    scalp_list += chop_scalp
    return [l for l in raw_labels if l.upper() not in scalp_list]


def _pull_ieeg(ds, start_usec, duration_usec, channel_ids):
    """Pull data from iEEG.org with retries on connection errors."""
    i = 0
    while True:
        if i == 50:
            logger = logging.getLogger()
            logger.error(
                f"failed to pull data for {ds.name}, {start_usec / 1e6}, "
                f"{duration_usec / 1e6}, {len(channel_ids)} channels"
            )
            return None
        try:
            return ds.get_data(start_usec, duration_usec, channel_ids)
        except Exception:
            time.sleep(1)
            i += 1


def get_ieeg_data(
    username,
    password_bin_file,
    ieeg_filename,
    start_time_usec,
    stop_time_usec,
    select_electrodes=None,
    ignore_electrodes=None,
    outputfile=None,
    force_pull=False,
):
    """Download a clip from iEEG.org and return a (DataFrame, fs) tuple.

    If ``outputfile`` is provided, pickle the (df, fs) tuple to that path
    instead of returning it.
    """
    from ieeg.auth import Session

    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec

    with open(password_bin_file, "r") as f:
        pwd = f.read()

    iter_count = 0
    while True:
        try:
            if iter_count == 50:
                raise ValueError("Failed to open dataset")
            s = Session(username, pwd)
            ds = s.open_dataset(ieeg_filename)
            all_channel_labels = ds.get_channel_labels()
            break
        except Exception:
            time.sleep(1)
            iter_count += 1
    all_channel_labels = clean_labels(all_channel_labels, ieeg_filename)

    if select_electrodes is not None:
        if isinstance(select_electrodes[0], Number):
            channel_ids = select_electrodes
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(select_electrodes[0], str):
            select_electrodes = clean_labels(select_electrodes, ieeg_filename)
            if any(i not in all_channel_labels for i in select_electrodes):
                if force_pull:
                    select_electrodes = [
                        e for e in select_electrodes if e in all_channel_labels
                    ]
                else:
                    raise ValueError("Channel not in iEEG")
            channel_ids = [
                i for i, e in enumerate(all_channel_labels) if e in select_electrodes
            ]
            channel_names = select_electrodes
        else:
            print("Electrodes not given as a list of ints or strings")

    elif ignore_electrodes is not None:
        if isinstance(ignore_electrodes[0], int):
            channel_ids = [
                i
                for i in np.arange(len(all_channel_labels))
                if i not in ignore_electrodes
            ]
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(ignore_electrodes[0], str):
            ignore_electrodes = clean_labels(ignore_electrodes, ieeg_filename)
            channel_ids = [
                i
                for i, e in enumerate(all_channel_labels)
                if e not in ignore_electrodes
            ]
            channel_names = [
                e for e in all_channel_labels if e not in ignore_electrodes
            ]
        else:
            print("Electrodes not given as a list of ints or strings")

    else:
        channel_ids = np.arange(len(all_channel_labels))
        channel_names = all_channel_labels

    if (duration < 120 * 1e6) and (len(channel_ids) < 100):
        data = _pull_ieeg(ds, start_time_usec, duration, channel_ids)
    elif (duration > 120 * 1e6) and (len(channel_ids) < 100):
        clip_size = 60 * 1e6
        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                data = _pull_ieeg(ds, clip_start, clip_size, channel_ids)
            else:
                new_data = _pull_ieeg(ds, clip_start, clip_size, channel_ids)
                data = np.concatenate((data, new_data), axis=0)
            clip_start = clip_start + clip_size
        last_clip_size = stop_time_usec - clip_start
        new_data = _pull_ieeg(ds, clip_start, last_clip_size, channel_ids)
        data = np.concatenate((data, new_data), axis=0)
    else:
        channel_size = 20
        channel_start = 0
        data = None
        while channel_start + channel_size < len(channel_ids):
            if data is None:
                data = _pull_ieeg(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
            else:
                new_data = _pull_ieeg(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
                data = np.concatenate((data, new_data), axis=1)
            channel_start = channel_start + channel_size
        last_channel_size = len(channel_ids) - channel_start
        new_data = _pull_ieeg(
            ds,
            start_time_usec,
            duration,
            channel_ids[channel_start : channel_start + last_channel_size],
        )
        data = np.concatenate((data, new_data), axis=1)

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate

    if outputfile:
        with open(outputfile, "wb") as f:
            pickle.dump([df, fs], f)
    else:
        return df, fs
