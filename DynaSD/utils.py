import numpy as np
import pandas as pd
import scipy as sc
from scipy.signal import butter,filtfilt,sosfiltfilt
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import logging
import json
import re
from os.path import join as ospj
from ieeg.auth import Session

# UTILITY FUNCTIONS
## FEATURE EXTRACTION
def num_wins(xLen, fs, winLen, winDisp):
  return int(((xLen/fs - winLen + winDisp) - ((xLen/fs - winLen + winDisp)%winDisp))/winDisp)

def MovingWinClips(x,fs,winLen,winDisp):
  # calculate number of windows and initialize receiver
  nWins = num_wins(len(x),fs,winLen,winDisp)
  samples = np.empty((nWins,int(winLen*fs)))
  # create window indices - these windows are left aligned
  idxs = np.array([(winDisp*fs*i,(winLen+winDisp*i)*fs)\
                   for i in range(nWins)]).astype(int)
  # apply feature function to each channel
  for i in range(idxs.shape[0]):
    samples[i,:] = x[idxs[i,0]:idxs[i,1]]
  
  return samples

## DATA PULLING

def _pull_iEEG(ds, start_usec, duration_usec, channel_ids):
    """
    Pull data while handling iEEGConnectionError
    """
    i = 0
    while True:
        if i == 50:
            logger = logging.getLogger()
            logger.error(
                f"failed to pull data for {ds.name}, {start_usec / 1e6}, {duration_usec / 1e6}, {len(channel_ids)} channels"
            )
            return None
        try:
            data = ds.get_data(start_usec, duration_usec, channel_ids)
            return data
        except Exception as _:
            time.sleep(1)
            i += 1

def load_config(config_path):
    with open(config_path,'r') as f:
        CONFIG = json.load(f)
    usr = CONFIG["paths"]["iEEG_USR"]
    passpath = CONFIG["paths"]["iEEG_PWD"]
    datapath = CONFIG["paths"]["RAW_DATA"]
    prodatapath = CONFIG["paths"]["PROCESSED_DATA"]
    figpath = CONFIG["paths"]["FIGURES"]
    metapath = CONFIG["paths"]["METADATA"]
    return usr,passpath,datapath,prodatapath,metapath,figpath

def get_iEEG_data(
    username: str,
    password_bin_file: str,
    iEEG_filename: str,
    start_time_usec: float,
    stop_time_usec: float,
    select_electrodes=None,
    ignore_electrodes=None,
    outputfile=None,
    force_pull = False
):
    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec

    with open(password_bin_file, "r") as f:
        pwd = f.read()

    iter = 0
    while True:
        try:
            if iter == 50:
                raise ValueError("Failed to open dataset")
            s = Session(username, pwd)
            ds = s.open_dataset(iEEG_filename)
            all_channel_labels = ds.get_channel_labels()
            break
            
        except Exception as e:
            time.sleep(1)
            iter += 1
    all_channel_labels = clean_labels(all_channel_labels, iEEG_filename)
    
    if select_electrodes is not None:
        if isinstance(select_electrodes[0], Number):
            channel_ids = select_electrodes
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(select_electrodes[0], str):
            select_electrodes = clean_labels(select_electrodes, iEEG_filename)
            if any([i not in all_channel_labels for i in select_electrodes]):
                if force_pull:
                    select_electrodes = [e for e in select_electrodes
                                          if e in all_channel_labels]
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
            ignore_electrodes = clean_labels(ignore_electrodes, iEEG_filename)
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

    # if clip is small enough, pull all at once, otherwise pull in chunks
    if (duration < 120 * 1e6) and (len(channel_ids) < 100):
        data = _pull_iEEG(ds, start_time_usec, duration, channel_ids)
    elif (duration > 120 * 1e6) and (len(channel_ids) < 100):
        # clip is probably too big, pull chunks and concatenate
        clip_size = 60 * 1e6

        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
            else:
                new_data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
                data = np.concatenate((data, new_data), axis=0)
            clip_start = clip_start + clip_size

        last_clip_size = stop_time_usec - clip_start
        new_data = _pull_iEEG(ds, clip_start, last_clip_size, channel_ids)
        data = np.concatenate((data, new_data), axis=0)
    else:
        # there are too many channels, pull chunks and concatenate
        channel_size = 20
        channel_start = 0
        data = None
        while channel_start + channel_size < len(channel_ids):
            if data is None:
                data = _pull_iEEG(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
            else:
                new_data = _pull_iEEG(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
                data = np.concatenate((data, new_data), axis=1)
            channel_start = channel_start + channel_size

        last_channel_size = len(channel_ids) - channel_start
        new_data = _pull_iEEG(
            ds,
            start_time_usec,
            duration,
            channel_ids[channel_start : channel_start + last_channel_size],
        )
        data = np.concatenate((data, new_data), axis=1)

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate  # get sample rate

    if outputfile:
        with open(outputfile, "wb") as f:
            pickle.dump([df, fs], f)
    else:
        return df, fs
    
## CHANNEL PREPROCESSING

def detect_bad_channels(data,fs,lf_stim = False):
    '''
    data: raw EEG traces after filtering (i think)
    fs: sampling frequency
    channel_labels: string labels of channels to use
    '''
    values = data.copy()
    which_chs = np.arange(values.shape[1])
    ## Parameters to reject super high variance
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 5e3

    ## Parameter to reject high 60 Hz
    percent_60_hz = 0.7

    ## Parameter to reject electrodes with much higher std than most electrodes
    mult_std = 10

    bad = []
    high_ch = []
    nan_ch = []
    zero_ch = []
    flat_ch = []
    high_var_ch = []
    noisy_ch = []
    all_std = np.empty((len(which_chs),1))
    all_std[:] = np.nan
    details = {}

    for i in range(len(which_chs)):       
        ich = which_chs[i]
        eeg = values[:,ich]
        bl = np.nanmedian(eeg)
        all_std[i] = np.nanstd(eeg)
        
        ## Remove channels with nans in more than half
        if sum(np.isnan(eeg)) > 0.5*len(eeg):
            bad.append(ich)
            nan_ch.append(ich)
            continue
        
        ## Remove channels with zeros in more than half
        if sum(eeg == 0) > (0.5 * len(eeg)):
            bad.append(ich)
            zero_ch.append(ich)
            continue

        ## Remove channels with extended flat-lining
        if (sum(np.diff(eeg,1) == 0) > (0.02 * len(eeg))) and (sum(abs(eeg - bl) > abs_thresh) > (0.02 * len(eeg))):
            bad.append(ich)
            flat_ch.append(ich)
        
        ## Remove channels with too many above absolute thresh
        if sum(abs(eeg - bl) > abs_thresh) > 10:
            if not lf_stim:
                bad.append(ich)
            high_ch.append(ich)
            continue

        ## Remove channels if there are rare cases of super high variance above baseline (disconnection, moving, popping)
        pct = np.percentile(eeg,[100-tile,tile])
        thresh = [bl - mult*(bl-pct[0]), bl + mult*(pct[1]-bl)]
        sum_outside = sum(((eeg > thresh[1]) + (eeg < thresh[0])) > 0)
        if sum_outside >= num_above:
            if not lf_stim:
                bad.append(ich)
            high_var_ch.append(ich)
            continue
        
        ## Remove channels with a lot of 60 Hz noise, suggesting poor impedance
        # Calculate fft
        Y = np.fft.fft(eeg-np.nanmean(eeg))
        
        # Get power
        P = abs(Y)**2
        freqs = np.linspace(0,fs,len(P)+1)
        freqs = freqs[:-1]
        
        # Take first half
        P = P[:np.ceil(len(P)/2).astype(int)]
        freqs = freqs[:np.ceil(len(freqs)/2).astype(int)]
        
        P_60Hz = sum(P[(freqs > 58) * (freqs < 62)])/sum(P)
        if P_60Hz > percent_60_hz:
            bad.append(ich)
            noisy_ch.append(ich)
            continue

    ## Remove channels for whom the std is much larger than the baseline
    median_std = np.nanmedian(all_std)
    higher_std = which_chs[(all_std > (mult_std * median_std)).squeeze()]
    bad_std = higher_std
    # for ch in bad_std:
    #     if ch not in bad:
    #         if ~lf_stim:
    #             bad.append(ch)
    channel_mask = np.ones((values.shape[1],),dtype=bool)
    channel_mask[bad] = False
    details['noisy'] = noisy_ch
    details['nans'] = nan_ch
    details['zeros'] = zero_ch
    details['flat'] = flat_ch
    details['var'] = high_var_ch
    details['higher_std'] = bad_std
    details['high_voltage'] = high_ch
    
    return channel_mask,details

def check_channel_types(ch_list, threshold=16):
    """Function to check channel types

    Args:
        ch_list (_type_): _description_
        threshold (int, optional): _description_. Defaults to 15.

    Returns:
        _type_: _description_
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
            "C",
            "Cz",
            "CZ",
            "F",
            "Fp",
            "FP",
            "Fz",
            "FZ",
            "O",
            "P",
            "Pz",
            "PZ",
            "T",
        ]:
            ch_df.loc[group.index.to_list(), "type"] = "eeg"
            continue
        if len(group) > threshold:
            ch_df.loc[group.index.to_list(), "type"] = "ecog"
        else:
            ch_df.loc[group.index.to_list(), "type"] = "seeg"
    return ch_df

def bipolar_montage(data: np.ndarray, ch_types: pd.DataFrame) -> np.ndarray:
    """_summary_

    Args:
        data (np.ndarray): _description_
        ch_types (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """

    n_ch = len(ch_types)
    new_ch_types = []
    for ind, row in ch_types.iterrows():
        # do only if type is ecog or seeg
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
    # apply montage to data
    new_data = np.empty((len(new_ch_types), data.shape[1]))
    for ind, row in new_ch_types.iterrows():
        new_data[ind, :] = data[row["idx1"], :] - data[row["idx2"], :]

    return new_data, new_ch_types

def remove_scalp_electrodes(raw_labels):
    scalp_list = ['CZ','FZ','PZ',
                  'A01','A02',
                  'C03','C04',
                  'F03','F04','F07','F08',
                  'FP01','FP02',
                  'O01','O02',
                  'P03','P04',
                  'T03','T04','T05','T06',
                  'EKG01','EKG02',
                  'ECG01','ECG02',
                  'ROC','LOC',
                  'EMG01','EMG02',
                  'DC01','DC07'
                  ]
    chop_scalp = ['C1'+str(x) for x in range(19,29)]
    scalp_list += chop_scalp
    return [l for l in raw_labels if l.upper() not in scalp_list]

def clean_labels(channel_li: list, pt: str) -> list:
    """This function cleans a list of channels and returns the new channels

    Args:
        channel_li (list): _description_

    Returns:
        list: _description_
    """

    new_channels = []
    for i in channel_li:
        i = i.replace("-", "")
        i = i.replace("GRID", "G")  # mne has limits on channel name size
        # standardizes channel names
        pattern = re.compile(r"([A-Za-z0-9]+?)(\d+)$")
        regex_match = pattern.match(i)

        if regex_match is None:
            new_channels.append(i)
            continue

        # if re.search('Cz|Fz|C3|C4|EKG',i):
        #     continue
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
            if (i == 'LG7'):# or (i == 'LG8'):
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

## SIGNAL PREPROCESSING

def downsample(data,fs,target):
    signal_len = int(data.shape[0]/fs*target)
    data_bpd = sc.signal.resample(data,signal_len,axis=0)
    return data_bpd,target

def preprocess_for_detection(data,fs,montage='bipolar',target=256, wavenet=False, pre_mask = None):
    # This function implements preprocessing steps for seizure detection
    chs = data.columns.to_list()
    ch_df = check_channel_types(chs)
    # Montage
    if montage == 'bipolar':
        data_bp_np,bp_ch_df = bipolar_montage(data.to_numpy().T,ch_df)
        bp_ch = bp_ch_df.name.to_numpy()
    elif montage == 'car':
        data_bp_np = (data.to_numpy().T - np.mean(data.to_numpy(),1))
        bp_ch = chs
    
    # Channel rejection
    if pre_mask is None:
        mask,_ = detect_bad_channels(data_bp_np.T*1e3,fs)
        data_bp_np = data_bp_np[mask,:]
        mask_list = [ch for ch in bp_ch[~mask]]
        bp_ch = bp_ch[mask]
    else:
        mask = np.atleast_1d([ch not in pre_mask for ch in bp_ch])
        data_bp_np = data_bp_np[mask,:]
        bp_ch = bp_ch[mask]
        
    # Filtering and autocorrelation
    if wavenet:
        target=128
        data_bp_notch = notch_filter(data_bp_np,fs)
        data_bp_filt = bandpass_filter(data_bp_notch,fs,lo=3,hi=127)
        signal_len = int(data_bp_filt.shape[1]/fs*target)
        data_bpd = sc.signal.resample(data_bp_filt,signal_len,axis=1).T
        fsd = int(target)
    else:
        # Bandpass filtering
        data_bp_notch = notch_filter(data_bp_np,fs)
        data_bp_filt = bandpass_filter(data_bp_notch,fs,lo=3,hi=150)
        # Down sampling
        signal_len = int(data_bp_filt.shape[1]/fs*target)
        data_bpd = sc.signal.resample(data_bp_filt,signal_len,axis=1).T
        fsd = int(target)
    data_white = ar_one(data_bpd)
    data_white_df = pd.DataFrame(data_white,columns = bp_ch)
    if pre_mask is None:
        return data_white_df,fsd,mask_list
    else:
        return data_white_df,fsd

def notch_filter(data: np.ndarray, fs: float) -> np.array:
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_

    Returns:
        np.array: _description_
    """
    # remove 60Hz noise
    b, a = butter(4,(58,62),'bandstop',fs=fs)
    d, c = butter(4,(118,122),'bandstop',fs=fs)

    data_filt = filtfilt(b, a, data, axis=-1)
    data_filt_filt = filtfilt(d, c, data_filt, axis = -1)
    # TODO: add option for causal filter
    # TODO: add optional argument for order

    return data_filt_filt

def bandpass_filter(data: np.ndarray, fs: float, order=3, lo=1, hi=150) -> np.array:
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_
        order (int, optional): _description_. Defaults to 3.
        lo (int, optional): _description_. Defaults to 1.
        hi (int, optional): _description_. Defaults to 120.

    Returns:
        np.array: _description_
    """
    # TODO: add causal function argument
    # TODO: add optional argument for order
    sos = butter(order, [lo, hi], output="sos", fs=fs, btype="bandpass")
    data_filt = sosfiltfilt(sos, data, axis=-1)
    return data_filt

def ar_one(data):
    """
    The ar_one function fits an AR(1) model to the data and retains the residual as
    the pre-whitened data
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
    Returns
    -------
        data_white: ndarray, shape (T, N)
            Whitened signal with reduced autocorrelative structure
    """
    # Retrieve data attributes
    n_samp, n_chan = data.shape
    # Apply AR(1)
    data_white = np.zeros((n_samp-1, n_chan))
    for i in range(n_chan):
        win_x = np.vstack((data[:-1, i], np.ones(n_samp-1)))
        w = np.linalg.lstsq(win_x.T, data[1:, i], rcond=None)[0]
        data_white[:, i] = data[1:, i] - (data[:-1, i]*w[0] + w[1])
    return data_white

## EXAMPLE FUNCTIONS
def set_plot_params():
    plt.rcParams['image.cmap'] = 'magma'
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['lines.linewidth'] = 2

    plt.rcParams['xtick.major.size'] = 5  # Change to your desired major tick size
    plt.rcParams['ytick.major.size'] = 5  # Change to your desired major tick size
    plt.rcParams['xtick.minor.size'] = 3   # Change to your desired minor tick size
    plt.rcParams['ytick.minor.size'] = 3   # Change to your desired minor tick size

    plt.rcParams['xtick.major.width'] = 2  # Change to your desired major tick width
    plt.rcParams['ytick.major.width'] = 2  # Change to your desired major tick width
    plt.rcParams['xtick.minor.width'] = 1  # Change to your desired minor tick width
    plt.rcParams['ytick.minor.width'] = 1  # Change to your desired minor tick width

def cohens_d(group1, group2):
    # Calculating means of the two groups
    mean1, mean2 = np.mean(group1), np.mean(group2)
     
    # Calculating pooled standard deviation
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
     
    # Calculating Cohen's d
    d = (mean1 - mean2) / pooled_std
     
    return d

def preprocess_for_scalp(data,fs,montage='bipolar',target=256, pre_mask = None):
    return None


def _shade_y_ticks_background(ax, y_ticks, colors, alpha=1):
    """
    Add colored background shading to a plot based on y-axis tick positions.
    
    This function creates horizontal colored bands behind plot data, with each band
    centered on a y-tick position. Useful for visually grouping or highlighting
    different data series in multi-channel plots.
    
    Args:
        ax (matplotlib.axes.Axes): The axis object to modify
        y_ticks (list or array): Y-tick values to center the shading bands on
        colors (list, array, or str): Colors for each shading band. Can be:
            - List of colors (one per y-tick)
            - Single color string (applied to all y-ticks)
            - None values skip shading for that y-tick
        alpha (float, optional): Transparency level (0=transparent, 1=opaque). 
            Defaults to 1.
            
    Notes:
        - Shading extends halfway to adjacent y-ticks on each side
        - For edge ticks, extends same distance as to nearest neighbor
        - Uses axhspan() to create horizontal spans across full x-axis width
        - Automatically sorts y-ticks to ensure proper ordering
        
    Raises:
        ValueError: If y_ticks and colors arrays have different lengths (when colors is a list)
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> y_positions = [0, 1, 2, 3]
        >>> channel_colors = ['red', 'blue', None, 'green']  # No shading for channel 2
        >>> shade_y_ticks_background(ax, y_positions, channel_colors, alpha=0.3)
    """
    if isinstance(colors,str):
        colors = [colors]*len(y_ticks)
    if len(y_ticks) != len(colors):
        raise ValueError("The length of y_ticks and colors must be the same.")

    # Sort y_ticks and colors together to ensure proper ordering
    sorted_indices = np.argsort(y_ticks)
    y_ticks = np.array(y_ticks)[sorted_indices]
    colors = np.array(colors)[sorted_indices]

    # Add shading between each pair of y-ticks
    for i in range(len(y_ticks)):
        if colors[i] is not None:
            if i == 0:
                # First tick: shade from this tick down to halfway to the next tick
                lower_bound = y_ticks[i] - (y_ticks[i + 1] - y_ticks[i]) / 2
            else:
                # Shade from halfway between this tick and the previous one
                lower_bound = (y_ticks[i - 1] + y_ticks[i]) / 2

            if i == len(y_ticks) - 1:
                # Last tick: shade up to halfway to the previous tick
                upper_bound = y_ticks[i] + (y_ticks[i] - y_ticks[i - 1]) / 2
            else:
                # Shade up to halfway between this tick and the next one
                upper_bound = (y_ticks[i] + y_ticks[i + 1]) / 2

            # Add a colored rectangle spanning the full x-axis width, avoiding overlap
            ax.axhspan(lower_bound, upper_bound, color=colors[i], alpha=alpha, linewidth=0)

def plot_iEEG_data(
    data, 
    fs=None,
    t=None,
    t_offset=0,
    colors=None,
    plot_color = 'k',
    shade_color = None,
    shade_alpha = 0.3,
    empty=False,
    dr=None,
    fig_size=None,
    minmax=False
):
    """
    Create a multi-channel iEEG data plot with customizable styling and layout.
    
    This function generates a standard "waterfall" or "butterfly" plot for multi-channel
    iEEG data, where each channel is plotted at a different y-offset for easy visualization.
    Supports various customization options including color coding, background shading,
    and automatic scaling.
    
    Args:
        data (pandas.DataFrame or numpy.ndarray): iEEG data matrix
            - If DataFrame: Uses column names as channel labels
            - Shape should be (time_points, channels) or (channels, time_points)
        fs (float, optional): Sampling frequency in Hz. Required if t is None.
        t (numpy.ndarray, optional): Time vector. If None, generated from fs.
        t_offset (float, optional): Time offset to add to time vector. Defaults to 0.
        colors (list, optional): List of colors for y-tick labels (channel names).
        plot_color (str, optional): Color for the data traces. Defaults to 'k' (black).
        shade_color (list, optional): Colors for background shading. See shade_y_ticks_background().
        shade_alpha (float, optional): Alpha transparency for background shading. Defaults to 0.3.
        empty (bool, optional): If True, removes plot borders and ticks for clean appearance.
        dr (float, optional): Vertical spacing between channels. If None, auto-calculated.
        fig_size (tuple, optional): Figure size (width, height). If None, auto-calculated.
        minmax (bool, optional): If True, z-score normalizes the data. Defaults to False.
        
    Returns:
        tuple: (fig, ax) - matplotlib figure and axis objects
        
    Notes:
        - Automatically transposes data if dimensions don't match time vector
        - Auto-calculates figure size based on duration and channel count
        - Channel labels appear as y-tick labels (if DataFrame input)
        - Supports both normalized and raw data display
        - Can overlay colored background shading for channel grouping
        
    Example:
        >>> data = pd.DataFrame(ieeg_data, columns=channel_names)
        >>> fig, ax = plot_iEEG_data(data, fs=500, colors=channel_colors)
        >>> plt.show()
    """
    if minmax:
        data = data.apply(sc.stats.zscore)
    if t is None:
        t = np.arange(len(data))/fs

    t += t_offset
    if data.shape[0] != np.size(t):
        data = data.T

    n_rows = data.shape[1]
    duration = t[-1] - t[0]
    
    if fig_size is not None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig, ax = plt.subplots(figsize=(duration / 3, n_rows / 5))
        
    sns.despine()

    ticklocs = []
    ax.set_xlim(t[0], t[-1])

    dmin = data.min().min()
    dmax = data.max().min()

    if dr is None:
        dr = (dmax - dmin) * 0.8  # Crowd them a bit.
    # if minmax and (dr is None):
    #     dr = 1

    y0 = dmin - dr
    y1 = (n_rows-1) * dr + dmax + dr/2
    ax.set_ylim([y0,y1])
    segs = []
    
    for i in range(n_rows):
        if isinstance(data, pd.DataFrame):
            segs.append(np.column_stack((t, data.iloc[:, i])))
        elif isinstance(data, np.ndarray):
            segs.append(np.column_stack((t, data[:, i])))
        else:
            print("Data is not in valid format")

    for i in reversed(range(n_rows)):
        ticklocs.append(i * dr)

    offsets = np.zeros((n_rows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    # # Set the yticks to use axes coordinates on the y axis
    ax.set_yticks(ticklocs)
    if isinstance(data, pd.DataFrame):
        ax.set_yticklabels(data.columns)

    if colors:
        for col, lab in zip(colors, ax.get_yticklabels()):
            if col is None:
                col = 'black'
            lab.set_color(col)

    ax.set_xlabel("Time (s)")

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    ax.plot(t, data + ticklocs, color=plot_color, lw=0.4)

    if shade_color is not None:    
        _shade_y_ticks_background(ax, ticklocs, shade_color, alpha=shade_alpha)

    if empty:
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Optionally, remove grid lines if present
        ax.grid(False)

        # Keep tick labels but remove tick markers
        ax.tick_params(axis='both', which='both', length=0)

    return fig, ax