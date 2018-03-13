# -*- coding: utf-8 -*-
"""some plot function mainly for debugging,
used in proposed_method_pipeline.py"""
import matplotlib
matplotlib.use('TkAgg')

from general.parameters import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def figure_plot_joint(mfcc_line,
                      syllable_gt_onsets_0start,
                      phoneme_gt_onsets_0start_without_syllable_onsets,
                      obs_syllable,
                      boundaries_syllable_start_time,
                      obs_phoneme,
                      boundaries_phoneme_start_time,
                      syllable_score_durs,
                      phoneme_score_durs):
    # plot Error analysis figures
    plt.figure(figsize=(16, 8))
    # class weight
    ax1 = plt.subplot(411)
    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 7:80 * 8]))
    for gso in syllable_gt_onsets_0start:
        plt.axvline(gso, color='r', linewidth=2)
    for gpo in phoneme_gt_onsets_0start_without_syllable_onsets:
        plt.axvline(gpo, color='k', linewidth=2)
        # for i_gs, gs in enumerate(groundtruth_onset):
        #     plt.axvline(gs, color='r', linewidth=2)
        # plt.text(gs, ax1.get_ylim()[1], groundtruth_syllables[i_gs])

    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')

    ax2 = plt.subplot(412, sharex=ax1)
    plt.plot(np.arange(0, len(obs_syllable)) * hopsize_t, obs_syllable)
    for bsst in boundaries_syllable_start_time:
        plt.axvline(bsst, color='r', linewidth=2)

    ax2.set_ylabel('ODF syllable', fontsize=12)
    ax2.axis('tight')

    ax3 = plt.subplot(413, sharex=ax1)
    plt.plot(np.arange(0, len(obs_phoneme)) * hopsize_t, obs_phoneme)
    for bpst in boundaries_phoneme_start_time:
        plt.axvline(bpst, color='k', linewidth=1)
    for bsst in boundaries_syllable_start_time:
        plt.axvline(bsst, color='r', linewidth=2)
    # for i_ib in range(len(i_boundary)-1):
    #     plt.axvline(i_boundary[i_ib] * hopsize_t, color='r', linewidth=2)
    # plt.text(i_boundary[i_ib] * hopsize_t, ax2.get_ylim()[1], syllables[i_line][i_ib])

    ax3.set_ylabel('ODF phoneme', fontsize=12)
    ax3.axis('tight')

    ax4 = plt.subplot(414, sharex=ax1)
    # print(duration_score)
    time_start = 0
    for ii_ds, ds in enumerate(syllable_score_durs):
        ax4.add_patch(
            patches.Rectangle(
                (time_start, 0),  # (x,y)
                ds,  # width
                3,  # height
                alpha=0.5
            ))
        time_start += ds

    time_start = 0
    for psd in phoneme_score_durs:
        ax4.add_patch(
            patches.Rectangle(
                (time_start, 0.5),  # (x,y)
                psd,  # width
                2,  # height
                color='r',
                alpha=0.5
            ))
        time_start += psd
    ax4.set_ylim((0, 3))
    ax4.set_ylabel('Score duration', fontsize=12)
    plt.xlabel('Time (s)')
    # plt.tight_layout()

    plt.show()


def figure_plot_hsmm(mfcc_line,
                     syllable_gt_onsets_0start,
                     phoneme_gt_onsets_0start_without_syllable_onsets,
                     hsmm,
                     phoneme_score_labels,
                     path,
                     boundaries_phoneme_start_time,
                     boundaries_syllable_start_time,
                     syllable_score_durs,
                     phoneme_score_durs,
                     obs_joint_phn):

    # plot Error analysis figures
    tPlot, axes = plt.subplots(
        nrows=4, ncols=1, sharex=True, sharey=False,
        gridspec_kw={'height_ratios': [1, 2, 1, 1]})
    ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]

    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    ax1.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 10:80 * 11]))
    for gso in syllable_gt_onsets_0start:
        ax1.axvline(gso, color='r', linewidth=2)
    for gpo in phoneme_gt_onsets_0start_without_syllable_onsets:
        ax1.axvline(gpo, color='k', linewidth=2)

    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')

    # plot observation proba matrix
    B_map = hsmm._getBmap()
    n_states = B_map.shape[0]
    n_frame = B_map.shape[1]
    y = np.arange(n_states + 1)
    x = np.arange(n_frame) * hopsize / float(fs)
    ax2.pcolormesh(x, y, B_map)
    ax2.set_yticks(y)
    ax2.set_yticklabels(phoneme_score_labels, fontdict={'fontsize': 6})
    ax2.plot(x, path, 'b', linewidth=2)     # plot the decoding path
    for bpst in boundaries_phoneme_start_time:
        ax2.axvline(bpst, color='k', linewidth=1)
    for bsst in boundaries_syllable_start_time:
        ax2.axvline(bsst, color='r', linewidth=2)

    ax2.axis('tight')

    # # ax3 = plt.subplot(313, sharex=ax1)
    # # print(duration_score)
    # time_start = 0
    # # print(syllable_score_durs)
    # for ii_ds, ds in enumerate(syllable_score_durs):
    #     ax3.add_patch(
    #         patches.Rectangle(
    #             (time_start, 0),  # (x,y)
    #             ds,  # width
    #             3,  # height
    #             alpha=0.5
    #         ))
    #     time_start += ds
    #
    # time_start = 0
    # for psd in phoneme_score_durs:
    #     ax3.add_patch(
    #         patches.Rectangle(
    #             (time_start, 0.5),  # (x,y)
    #             psd,  # width
    #             2,  # height
    #             color='r',
    #             alpha=0.5
    #         ))
    #     time_start += psd
    # ax3.set_ylim((0, 3))
    # ax3.set_ylabel('Score duration', fontsize=12)
    # ax3.axis('tight')

    if obs_joint_phn is not None:
        ax4.plot(x, obs_joint_phn)
        ax4.axis('tight')

    plt.xlabel('Time (s)')

    plt.show()
