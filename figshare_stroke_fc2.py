# %% Figshare stroke data process with functional connectivity method
# % 0-2s:prepare, 2-6s:MI, 6-8s:break
# % labels: 1-left motor attempt, 2-right motor attemptimport matplotlib
import networkx as nx
from joblib import load
import numpy as np
import matplotlib
import pandas as pd
from mne_connectivity import spectral_connectivity_time, spectral_connectivity_epochs
from mne_connectivity.viz import plot_sensors_connectivity
from nilearn.plotting import plot_connectome, plot_matrix
from scipy import signal
import scipy.io
import mne
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.decoding import CSP, UnsupervisedSpatialFilter
from mne.time_frequency import tfr_morlet, tfr_array_morlet, tfr_multitaper
from mne.time_frequency import psd_array_welch
from scipy.fftpack import fft, fftshift
from sklearn.preprocessing import minmax_scale, scale
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from scipy.signal import hilbert, welch
from matplotlib.colors import Normalize, TwoSlopeNorm
import os
import scipy.io as sio
from sklearn.metrics import accuracy_score
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
import matplotlib.colors as colors
from spectral_connectivity import multitaper_connectivity, Multitaper, Connectivity

matplotlib.use("Qt5Agg")


def epochs_make(eeg, labels, trigger):
    """build mne epochs from raw eeg data

    :param eeg: raw eeg data, 40 trials, 29 channels, 4000 sample points
    :param labels: labels for each trial (1:left, 2:right)
    :param trigger: start point of MI for each trial
    :return: mne epochs
    """
    info = mne.create_info(
        ch_names=["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
                  "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"],
        ch_types="eeg",  # channel type
        sfreq=500,  # frequency
    )
    location = {'FP1': [-0.309, 0.95, -0.0349], 'FP2': [0.309, 0.95, -0.0349], 'Fz': [-0, 0.719, 0.695],
                'F3': [-0.545, 0.673, 0.5],
                'F4': [0.545, 0.673, 0.5], 'F7': [-0.809, 0.587, -0.0349], 'F8': [0.809, 0.587, -0.0349],
                'FCz': [-0, 0.391, 0.921],
                'FC3': [-0.676, 0.36, 0.643], 'FC4': [0.676, 0.36, 0.643], 'FT7': [-0.95, 0.309, 0.0349],
                'FT8': [0.95, 0.309, -0.0349],
                'Cz': [0, 0, 1], 'C3': [-0.719, 0, 0.695], 'C4': [0.719, 0, 0.695], 'T3': [-0.999, 6.12e-17, -0.0349],
                'T4': [0.999, 6.12e-17, -0.0349], 'CP3': [-0.676, -0.36, 0.643], 'CP4': [0.676, -0.36, 0.643],
                'TP7': [-0.95, -0.309, -0.0349],
                'TP8': [0.95, -0.309, -0.0349], 'Pz': [-8.81e-17, -0.719, 0.695], 'P3': [-0.545, -0.673, 0.5],
                'P4': [0.545, -0.673, 0.5],
                'T5': [-0.809, -0.587, -0.0349], 'T6': [0.809, -0.587, -0.0349], 'Oz': [-1.22e-16, -0.999, -0.0349],
                'O1': [-0.309, -0.95, -0.0349],
                'O2': [0.309, -0.95, -0.0349]
                }
    for key in location:
        location[key] = 0.1 * np.array(location[key])
    montage = mne.channels.make_dig_montage(location)
    # montage = mne.channels.read_custom_montage('dataset/task-motor-imagery_electrodes.tsv', head_size=1)
    info.set_montage(montage)
    trials = eeg.shape[0]
    marks = np.zeros(trials)
    # concatenate trials to single raw data
    for i in range(trials):
        if i == 0:
            raw_data = np.squeeze(eeg[0, :, :])
        else:
            raw_data = np.concatenate((raw_data, np.squeeze(eeg[i, :, :])), axis=-1)
        marks[i] = trigger[i] + 4000 * i
    raw = mne.io.RawArray(raw_data, info)
    # sphere = mne.make_sphere_model("auto", "auto", raw.info)
    # src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0, pos=15.0)
    # forward = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere)
    # raw = raw.copy().set_eeg_reference("REST", forward=forward)

    events = np.zeros((trials, 3), dtype='int')
    events[:, 0] = marks
    events[:, -1] = labels
    event_id = dict(left_hand=1, right_hand=2)
    tmin, tmax = -2., 4.999     # select time period [-2, 5]s
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=True)

    return epochs


# def preprocess(epochs, low, high):
#     # epochs.filter(low, high, method='iir')
#     epochs._data = epochs._data * 1e-6
#     epochs.plot(n_epochs=10)
#     # bad = [11, 60, 75]  # S3:37,73  S5:11,60,75  S6:11, 18, 24, 26, 41, 55, 57, 68, 72, 79, 80
#     # epochs.drop(bad)
#     # labels = np.delete(labels, bad)
#     # epochs.info["bads"] = ["PO7", "O2"]
#     # epochs = epochs.interpolate_bads()
#
#     # ICA
#     ica = ICA(n_components=12, method="fastica", max_iter="auto", random_state=97).fit(epochs)
#     explained_var_ratio = ica.get_explained_variance_ratio(epochs)
#     ica.plot_sources(epochs, show_scrollbars=False)
#     ica.plot_components(inst=epochs, psd_args={'fmin': 0, 'fmax': 50})
#     # ica.exclude = [4,5,12,13,14]  # S2:4,5,12,13,14  S3:0,3,5,7,8,12,13  S4:1,4,12  S6:0,1,7,9,10,11,14  S7:2,3,5,6,9,10,11,12,13  S8:0,1,2,9, 10,11,12,13,14
#     ica.apply(epochs)
#
#     epochs_data = epochs.get_data()
#     plt.figure()
#     plt.plot(range(epochs_data.shape[-1]), epochs_data[0, :, :].T)
#     tmp = fft(epochs_data, axis=-1)
#     tmp1 = np.array(np.abs(tmp) / 1024)
#     freq_x = 512 * np.array(range(8 * 512)) / (8 * 512)
#     plt.figure()
#     plt.plot(freq_x[0:400], np.mean(tmp1[:, :, 0:400].T, axis=2))
#     return epochs


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return new_cmap


def fc_plot_spectral(epochs, low, high, fs, ax1, ax2, t0, t1):
    """Plot functional connectivity graph in specific band using spectral_connectivity package

    :param epochs: mne epoch data
    :param low: low freq
    :param high: high freq
    :param fs: sampling freq
    :param ax1: axes, plot left hand
    :param ax2: axes, plot right hand
    :param t0: MI start time
    :param t1: MI end time
    :return: functional connectivity matrix
    """
    # plt.ioff() # if not showing figure
    cmap = plt.get_cmap('YlGnBu')
    new_cmap = truncate_colormap(cmap, 0., 1)
    coords = np.array([[-0.309, 0.95, -0.0349], [0.309, 0.95, -0.0349], [-0, 0.719, 0.695], [-0.445, 0.673, 0.5],
                       [0.445, 0.673, 0.5], [-0.809, 0.587, -0.0349], [0.809, 0.587, -0.0349], [-0, 0.391, 0.921],
                       [-0.576, 0.36, 0.643], [0.576, 0.36, 0.643], [-0.95, 0.309, 0.0349], [0.95, 0.309, -0.0349],
                       [0, 0, 1], [-0.619, 0, 0.695], [0.619, 0, 0.695], [-0.999, 6.12e-17, -0.0349],
                       [0.999, 6.12e-17, -0.0349], [-0.576, -0.36, 0.643], [0.576, -0.36, 0.643],
                       [-0.95, -0.309, -0.0349],
                       [0.95, -0.309, -0.0349], [-8.81e-17, -0.719, 0.695], [-0.445, -0.673, 0.5], [0.445, -0.673, 0.5],
                       [-0.809, -0.587, -0.0349], [0.809, -0.587, -0.0349], [-1.22e-16, -0.999, -0.0349],
                       [-0.309, -0.95, -0.0349],
                       [0.309, -0.95, -0.0349]])
    labels = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
              "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]
    # epochs = epochs - np.expand_dims(np.mean(epochs, axis=1), axis=1)     # CAR

    # calculate connectivity using spectral domain method
    epochs_permute = epochs.transpose(2, 0, 1)
    multitaper = Multitaper(epochs_permute[t0:t1, :, :], sampling_frequency=500)
    connectivity = Connectivity.from_multitaper(multitaper, expectation_type="time_trials_tapers", blocks=1)
    fc = connectivity.imaginary_coherence()  # method: imagery coherence
    fc = np.abs(fc)
    f_range = np.where(np.logical_and(connectivity.frequencies >= low, connectivity.frequencies <= high))[0]
    connect = np.squeeze(np.mean(fc[f_range[0]:f_range[-1] + 1, :, :], axis=0))

    # # baseline correction
    # fc_base = multitaper_connectivity(epochs_permute[t0-2*fs:t0, :, :], sampling_frequency=500, method='imaginary_coherence')
    # fc_base = np.squeeze(fc_base.data[0, f_range[0]:f_range[-1] + 1, :, :])
    # fc_base = np.mean(np.abs(fc_base), axis=0)
    # connect = np.log(connect/fc_base)
    # row, col = np.diag_indices_from(connect)
    # connect[row, col] = 0
    # connect[row, col] = np.min(connect)

    # plot functional connectivity matrix and corresponding connectome
    plot_matrix(connect, colorbar=False, labels=labels, cmap=new_cmap, axes=ax1)
    plt.subplots_adjust(left=0.1)
    # plt.figure()
    coords[:, 0] = coords[:, 0] * 0.9
    coords[:, 1] = coords[:, 1] * 1.2
    coords = coords * 70
    coords[:, 1] = coords[:, 1] - 15
    plot_connectome(connect, coords, edge_threshold='50%', edge_vmax=np.max(connect),
                    edge_vmin=np.percentile(connect, 50), axes=ax2, display_mode='z', edge_cmap=new_cmap, colorbar=True)

    # figure 3 left column for paper
    fig, ax = plt.subplots(1, 1, figsize=(5.5 / 2.54, 5.5 / 2.54), layout='constrained')
    colormap = 'Blues'
    pm = plot_matrix(connect, colorbar=False, labels=labels, cmap=colormap, axes=ax)
    ax.set_yticklabels([item.get_text() for item in ax.get_yticklabels()], rotation=0)
    ax.set_xticklabels([item.get_text() for item in ax.get_xticklabels()], rotation=90)
    fig, ax = plt.subplots(1, 1, figsize=(6.2 / 2.54, 5.8 / 2.54), layout='constrained')
    pc = plot_connectome(connect, coords, edge_threshold='50%', edge_vmax=np.max(connect), node_size=20,
                         node_color='cornflowerblue',
                         edge_vmin=np.percentile(connect, 50), axes=ax, display_mode='z', edge_cmap=colormap,
                         colorbar=False, alpha=0.2)
    fig.colorbar(pm, ax=ax, shrink=0.6, pad=0.1)
    # ax[-1].set_axis_off()
    coords_proj = [(0.24, 0.86), (0.41, 0.86), (0.33, 0.76), (0.21, 0.75), (0.46, 0.75), (0.1, 0.72), (0.56, 0.72),
                   (0.33, 0.65),
                   (0.17, 0.64), (0.5, 0.64), (0.06, 0.62), (0.6, 0.62), (0.32, 0.5), (0.15, 0.5), (0.5, 0.5),
                   (0.04, 0.5), (0.6, 0.5),
                   (0.16, 0.36), (0.48, 0.36), (0.05, 0.38), (0.59, 0.38), (0.33, 0.23), (0.2, 0.25), (0.45, 0.25),
                   (0.09, 0.28),
                   (0.55, 0.28), (0.32, 0.13), (0.23, 0.15), (0.41, 0.15)]
    # for i, coord in enumerate(coords_proj):
    #     ax[1].annotate(labels[i], (coords_proj[i][0], coords_proj[i][1]), fontsize=14)
    # plt.subplots_adjust(left=0.1)

    # # flowchart example for fc plot
    # fig, ax = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.4 / 2.54), layout='constrained')
    # pc = plot_connectome(connect, coords, edge_threshold='50%', edge_vmax=np.max(connect), node_size=10,
    #                      node_color='cornflowerblue', edge_kwargs={'lw': 0},
    #                      edge_vmin=np.percentile(connect, 50), axes=ax, display_mode='z', edge_cmap=colormap,
    #                      colorbar=False, alpha=0.6)
    # fig.colorbar(pm, ax=ax, shrink=0.6, pad=0.1)

    return connect


# def fc_plot_mne(epochs, low, high, fs, ax1, ax2, t0, t1):
#     cmap = plt.get_cmap('YlGnBu')
#     new_cmap = truncate_colormap(cmap, 0., 1)
#     coords = np.array([[-0.309, 0.95, -0.0349], [0.309, 0.95, -0.0349], [-0, 0.719, 0.695], [-0.445, 0.673, 0.5],
#                        [0.445, 0.673, 0.5], [-0.809, 0.587, -0.0349], [0.809, 0.587, -0.0349], [-0, 0.391, 0.921],
#                        [-0.576, 0.36, 0.643], [0.576, 0.36, 0.643], [-0.95, 0.309, 0.0349], [0.95, 0.309, -0.0349],
#                        [0, 0, 1], [-0.619, 0, 0.695], [0.619, 0, 0.695], [-0.999, 6.12e-17, -0.0349],
#                        [0.999, 6.12e-17, -0.0349], [-0.576, -0.36, 0.643], [0.576, -0.36, 0.643],
#                        [-0.95, -0.309, -0.0349],
#                        [0.95, -0.309, -0.0349], [-8.81e-17, -0.719, 0.695], [-0.445, -0.673, 0.5], [0.445, -0.673, 0.5],
#                        [-0.809, -0.587, -0.0349], [0.809, -0.587, -0.0349], [-1.22e-16, -0.999, -0.0349],
#                        [-0.309, -0.95, -0.0349],
#                        [0.309, -0.95, -0.0349]])
#     labels = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
#               "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]
#     freqs = np.linspace(low, high, int((high - low) * 2 + 1))
#     # epochs = epochs - np.expand_dims(np.mean(epochs, axis=1), axis=1)     # CAR
#     connect_matrix = spectral_connectivity_time(epochs[:, :, t0 + 1:t1], freqs=freqs, faverage=True, sfreq=fs,
#                                                 average=True, method='pli', fmin=low, fmax=high)
#     connect = connect_matrix.get_data(output='dense')
#     connect = np.squeeze(np.mean(connect, axis=-1))
#     connect = connect + connect.T
#     plot_matrix(connect, colorbar=True, labels=labels, cmap=new_cmap, axes=ax1)
#     plt.subplots_adjust(left=0.1)
#     # plt.figure()
#     coords = coords * 70
#     coords[:, 1] = coords[:, 1] - 10
#     plot_connectome(connect, coords, edge_threshold='50%', edge_vmax=np.max(connect),
#                     edge_vmin=np.percentile(connect, 50), axes=ax2, display_mode='z', edge_cmap=new_cmap, colorbar=True)
#     plt.subplots_adjust(left=0.1)
#
#     return connect


if __name__ == "__main__":
    plt.ion()  # if not showing figure: plt.ioff()
    # adjust figure format
    plt.rc('font', size=9, family='Arial', weight='normal')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['axes.labelweight'] = 'normal'
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['axes.titlesize'] = 11
    matplotlib.rcParams['axes.titleweight'] = 'normal'
    matplotlib.rcParams['axes.linewidth'] = 1.0
    matplotlib.rcParams['svg.fonttype'] = 'none'

    # result variable of  functional connectivity: 50 subjects, 29 channels, 3 frequency bands
    connect_left_all = np.zeros((50, 29, 29, 3))
    connect_right_all = np.zeros((50, 29, 29, 3))

    for s in range(1, 51):  # s:subject num(1-50)
        # load data
        if s >= 10:
            data = sio.loadmat('dataset/sourcedata/sub-' + str(s) + '/sub-' + str(s) + '_task-motor-imagery_eeg.mat')
        else:
            data = sio.loadmat('dataset/sourcedata/sub-0' + str(s) + '/sub-0' + str(s) + '_task-motor-imagery_eeg.mat')

        # read raw data and label
        eeg_data = data['eeg']
        eeg_raw_data = eeg_data['rawdata']
        labels = eeg_data['label']
        epochs = eeg_raw_data.tolist()[0][0]
        labels = np.squeeze(labels.tolist()[0][0])
        trigger = np.squeeze(np.where(epochs[:, -1, :] == 2))[1, :]    # "2": beginning of motor imagery

        # preprocess: 8-30 Hz bandpass
        low = 8
        high = 30
        fs = 500
        chan = np.concatenate((np.arange(0, 17), np.arange(18, 30)), axis=-1)
        b, a = signal.butter(4, [low * 2 / fs, high * 2 / fs], 'bandpass')
        epochs_filt = signal.filtfilt(b, a, epochs[:, chan, :], axis=-1)
        plt.figure()
        plt.plot(range(epochs_filt.shape[-1]), epochs_filt[1, :, :].T)

        # make mne epochs data
        epochs_train = epochs_make(epochs_filt, labels, trigger)  # t:[-2,5]s
        epochs_left = epochs_train.get_data()
        tmp = fft(epochs_left[:, :, 2 * fs:6 * fs], axis=-1)
        tmp1 = np.array(np.abs(tmp) / 1000)
        freq_x = fs * np.array(range(4 * fs)) / (4 * fs)
        plt.figure()
        plt.plot(freq_x[0:200], np.mean(tmp1[:, :, 0:200].T, axis=2))

        # visulization example:channel signal
        fig, ax = plt.subplots(10, 1, figsize=(5.65 / 2.54, 4.45 / 2.54), sharex=True, gridspec_kw={'hspace': 0})
        chans = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
                 "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]
        epochs_colors = plt.get_cmap('Dark2', 29)
        for i, k in enumerate(range(7, 17)):
            ax[i].plot(range(1000), epochs_filt[0, k, :1000].T, color='cornflowerblue')
            ax[i].plot(range(1000, 3000), epochs_filt[0, k, 1000:3000].T, color='coral')
            ax[i].plot(range(3000, 4000), epochs_filt[0, k, 3000:].T, color='cornflowerblue')
            if i != 9:
                ax[i].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
            else:
                ax[i].spines[['right', 'top', 'left', 'bottom']].set_visible(False)
            ax[i].set_ylabel(chans[k], rotation=0, fontdict={'fontsize': 9})
            ax[i].set_yticks([])
            ax[i].set_xticks([])

        # functional connectivity: MI for left and right hand, 3 frequency bands
        fig, ax = plt.subplots(3, 4, figsize=(20, 15),
                               gridspec_kw={'width_ratios': [0.1, 0.1, 0.1, 0.1], 'height_ratios': [0.2, 0.2, 0.2]})
        plt.figtext(0.3, 0.98, "Left hand MI", va="center", ha="center", size=20, fontname='sans-serif')
        plt.figtext(0.75, 0.98, "Right hand MI", va="center", ha="center", size=20, fontname='sans-serif')
        plt.figtext(0.05, 0.85, "8-13 Hz", va="center", ha="center", size=20, fontname='sans-serif')
        plt.figtext(0.05, 0.5, "14-20 Hz", va="center", ha="center", size=20, fontname='sans-serif')
        plt.figtext(0.05, 0.2, "21-29 Hz", va="center", ha="center", size=20, fontname='sans-serif')
        connect_freq1_l = fc_plot_spectral(epochs_train['left_hand'].get_data(), 8, 13, fs, ax[0, 0], ax[0, 1], 2 * fs,
                                           6 * fs)
        connect_freq1_r = fc_plot_spectral(epochs_train['right_hand'].get_data(), 8, 13, fs, ax[0, 2], ax[0, 3], 2 * fs,
                                           6 * fs)
        connect_freq2_l = fc_plot_spectral(epochs_train['left_hand'].get_data(), 14, 20, fs, ax[1, 0], ax[1, 1], 2 * fs,
                                           6 * fs)
        connect_freq2_r = fc_plot_spectral(epochs_train['right_hand'].get_data(), 14, 20, fs, ax[1, 2], ax[1, 3],
                                           2 * fs, 6 * fs)
        connect_freq3_l = fc_plot_spectral(epochs_train['left_hand'].get_data(), 21, 29, fs, ax[2, 0], ax[2, 1], 2 * fs,
                                           6 * fs)
        connect_freq3_r = fc_plot_spectral(epochs_train['right_hand'].get_data(), 21, 29, fs, ax[2, 2], ax[2, 3],
                                           2 * fs, 6 * fs)
        connect_left = np.stack((connect_freq1_l, connect_freq2_l, connect_freq3_l), axis=-1)
        connect_right = np.stack((connect_freq1_r, connect_freq2_r, connect_freq3_r), axis=-1)
        connect_left_all[s - 1, :, :, :] = connect_left
        connect_right_all[s - 1, :, :, :] = connect_right

    # save fc data (further used to calculate MST)
    np.save('data_load/ImCoh_data/alpha_beta12/imcoh_left.npy', connect_left_all)
    np.save('data_load/ImCoh_data/alpha_beta12/imcoh_right.npy', connect_right_all)
    plt.ioff()
    plt.show()
