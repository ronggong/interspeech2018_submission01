# -*- coding: utf-8 -*-

import pickle
from os import makedirs
from os.path import exists

import numpy as np
import soundfile as sf
import pyximport

pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

import viterbiDecodingPhonemeSeg
from general.filePathShared import *
from general.filePathJoint import *
from general.parameters import *
from general.trainTestSeparation import get_train_test_recordings_joint
from general.utilFunctions import textgrid_syllable_phoneme_parser
from audio_preprocessing import feature_reshape
from audio_preprocessing import get_log_mel_madmom
from helper_code import gt_score_preparation_helper
from helper_code import results_aggregation_save_helper
from helper_code import findShiftOffset
from helper_code import score_variations_phn
from plot_code import figure_plot_joint
from eval.onsetSegmentEval.runEval import run_eval_onset
from eval.onsetSegmentEval.runEval import run_eval_segment

from keras.models import load_model


def smooth_obs(obs):
    """
    hanning window smooth the onset observation function
    :param obs: syllable/phoneme onset function
    :return:
    """
    hann = np.hanning(5)
    hann /= np.sum(hann)

    obs = np.convolve(hann, obs, mode='same')

    return obs


def onset_function_all_recordings(wav_path,
                                  textgrid_path,
                                  scaler,
                                  test_recordings,
                                  model_keras_cnn_0,
                                  cnnModel_name,
                                  eval_results_path,
                                  obs_cal='tocal',
                                  plot=False,
                                  save_data=False,
                                  missing_phn=False):
    """
    ODF and viterbi decoding
    :param wav_path: string, the path of the .wav files
    :param textgrid_path: string, the path of the .textgrid files
    :param scaler: sklearn scaler object
    :param test_recordings: list, the test recording names
    :param model_keras_cnn_0: loaded keras CNN model name
    :param eval_results_path: string, where to put the evaluation results
    :param obs_cal: string, tocal or toload, if to calculate the observation function
    :param plot: bool
    :param save_data: bool, whether to save the wav, duration and label data
    :param missing_phn: bool, whether to consider the missing phonemes in actual singing, for experiment (not in the paper)
    :return:
    """

    for artist_path, fn in test_recordings:

        print('Calculating for artist:', artist_path, 'filename:', fn)

        # use the teacher's text grid as the score
        score_text_grid_file = join(textgrid_path, artist_path, 'teacher.TextGrid')
        # student text grid
        ground_truth_text_grid_file = join(textgrid_path, artist_path, fn+'.TextGrid')

        # student .wav
        wav_file = join(wav_path, artist_path, fn + '.wav')

        # parse teacher (score) and student (ground truth) text grid file
        score_syllable_lists, score_phoneme_lists = \
            textgrid_syllable_phoneme_parser(score_text_grid_file, 'dianSilence', 'detailsSilence')
        gt_syllable_lists, gt_phoneme_lists = \
            textgrid_syllable_phoneme_parser(ground_truth_text_grid_file, 'dianSilence', 'details')

        # do audio precessing
        if obs_cal == 'tocal' or plot:
            mfcc = get_log_mel_madmom(wav_file, fs, hopsize_t, channel=1)
            mfcc_scaled = scaler.transform(mfcc)
            mfcc_reshaped = feature_reshape(mfcc_scaled, nlen=7)

        for ii_line in range(len(gt_syllable_lists)):
            print('line:', ii_line)

            # observation path, save the onset function for the next time calculation
            obs_path = join('./obs', cnnModel_name, artist_path)
            obs_syllable_filename = fn + '_syllable_' + str(ii_line + 1) + '.pkl'
            obs_phoneme_filename = fn + '_phoneme_' + str(ii_line + 1) + '.pkl'

            # sometimes the score and ground truth text grids are not started from the same phrase,
            # ii_aug is the offset
            ii_aug = findShiftOffset(gt_syllable_lists, score_syllable_lists, ii_line)

            # calculate necessary information from the text grid
            frame_start, frame_end, \
            time_start, time_end, \
            syllable_gt_onsets, syllable_gt_labels, \
            phoneme_gt_onsets, phoneme_gt_labels, \
            syllable_score_onsets, syllable_score_labels, \
            phoneme_score_onsets, phoneme_score_labels, \
            syllable_score_durs, phoneme_list_score = \
                gt_score_preparation_helper(gt_syllable_lists,
                                            score_syllable_lists,
                                            gt_phoneme_lists,
                                            score_phoneme_lists,
                                            ii_line,
                                            ii_aug)

            # collect phoneme durations and labels
            phoneme_score_durs_grouped_by_syllables = []
            phoneme_score_labels_grouped_by_syllables = []
            phoneme_score_durs_syllable = []
            phoneme_score_labels_syllable = []
            for pls in phoneme_list_score:

                # when the phoneme onset time is also the syllable onset time
                if pls[0] in syllable_score_onsets[1:]:
                    phoneme_score_durs_grouped_by_syllables.append(phoneme_score_durs_syllable)
                    phoneme_score_labels_grouped_by_syllables.append(phoneme_score_labels_syllable)
                    phoneme_score_durs_syllable = []
                    phoneme_score_labels_syllable = []

                phoneme_score_durs_syllable.append(pls[1] - pls[0])
                phoneme_score_labels_syllable.append(pls[2])

                if pls == phoneme_list_score[-1]:
                    phoneme_score_durs_grouped_by_syllables.append(phoneme_score_durs_syllable)
                    phoneme_score_labels_grouped_by_syllables.append(phoneme_score_labels_syllable)

            # onsets start from time 0
            syllable_gt_onsets_0start = np.array(syllable_gt_onsets) - syllable_gt_onsets[0]
            phoneme_gt_onsets_0start = np.array(phoneme_gt_onsets) - phoneme_gt_onsets[0]
            phoneme_gt_onsets_0start_without_syllable_onsets = \
                np.setdiff1d(phoneme_gt_onsets_0start, syllable_gt_onsets_0start)

            if not set(syllable_gt_onsets).issubset(set(phoneme_gt_onsets)):
                raise
            if not set(syllable_score_onsets).issubset(set(phoneme_score_onsets)):
                raise

            frame_start = int(round(time_start / hopsize_t))
            frame_end = int(round(time_end / hopsize_t))

            syllable_score_durs = np.array(syllable_score_durs)
            syllable_score_durs *= (time_end - time_start) / np.sum(syllable_score_durs)

            if obs_cal == 'tocal' or plot:
                mfcc_line = mfcc[frame_start:frame_end]
                mfcc_reshaped_line = mfcc_reshaped[frame_start:frame_end]
                mfcc_reshaped_line = np.expand_dims(mfcc_reshaped_line, axis=1)

                # calculate syllable and phoneme onset functions
                obs_syllable, obs_phoneme = model_keras_cnn_0.predict(mfcc_reshaped_line, batch_size=128, verbose=2)

                # save onset functions into obs_path
                print('save onset curve ... ...')
                if not exists(obs_path):
                    makedirs(obs_path)
                pickle.dump(obs_syllable, open(join(obs_path, obs_syllable_filename), 'w'))
                pickle.dump(obs_phoneme, open(join(obs_path, obs_phoneme_filename), 'w'))

            else:
                obs_syllable = pickle.load(open(join(obs_path, obs_syllable_filename), 'r'))
                obs_phoneme = pickle.load(open(join(obs_path, obs_phoneme_filename), 'r'))

            obs_syllable = np.squeeze(obs_syllable)
            obs_phoneme = np.squeeze(obs_phoneme)

            obs_syllable = smooth_obs(obs_syllable)
            obs_phoneme = smooth_obs(obs_phoneme)

            # decoding syllable boundaries
            obs_syllable[0] = 1.0
            obs_syllable[-1] = 1.0
            boundaries_syllable = viterbiDecodingPhonemeSeg.viterbiSegmental2(obs_syllable, syllable_score_durs, varin)

            # syllable boundaries
            boundaries_syllable_start_time = np.array(boundaries_syllable[:-1])*hopsize_t
            boundaries_syllable_end_time = np.array(boundaries_syllable[1:])*hopsize_t

            # initialize phoneme boundaries arrays
            boundaries_phoneme_start_time = np.array([])
            boundaries_phoneme_end_time = np.array([])

            # array of the phoneme durations to be concatenated
            phoneme_score_durs = np.array([])
            phoneme_score_labels = []

            # decode phoneme onsets
            for ii_syl_boundary in range(len(boundaries_syllable)-1):

                dur_syl = boundaries_syllable_end_time[ii_syl_boundary] - boundaries_syllable_start_time[ii_syl_boundary]

                frame_start_syl = boundaries_syllable[ii_syl_boundary]
                frame_end_syl = boundaries_syllable[ii_syl_boundary+1]

                obs_phoneme_syl = obs_phoneme[frame_start_syl: frame_end_syl]
                obs_phoneme_syl[0] = 1.0
                obs_phoneme_syl[-1] = 1.0

                # phoneme score durs and labels for the current syllable, used in the decoding
                phoneme_score_durs_syl = np.array(phoneme_score_durs_grouped_by_syllables[ii_syl_boundary])

                if len(phoneme_score_durs_syl) < 2:
                    continue

                phoneme_score_durs_syl_vars = [phoneme_score_durs_syl] # init the durs_syl_vars
                if missing_phn:
                    phoneme_score_labels_syl = phoneme_score_labels_grouped_by_syllables[ii_syl_boundary]
                    phoneme_score_labels_syl_vars, phoneme_score_durs_syl_vars = \
                        score_variations_phn(phoneme_score_labels_syl, phoneme_score_durs_syl)

                # missing phoneme decoding, only for experiment, not included in the paper
                if missing_phn and len(phoneme_score_durs_syl_vars) > 1:
                    boundaries_phoneme_syl_vars = []
                    phoneme_score_durs_syl_vars_norm = []
                    posterior_vars =[]
                    for ii in range(len(phoneme_score_durs_syl_vars)):
                        phoneme_score_labels_syl_vars_ii = phoneme_score_labels_syl_vars[ii]
                        phoneme_score_durs_syl_vars_ii = np.array(phoneme_score_durs_syl_vars[ii])
                        phoneme_score_durs_syl_vars_ii *= dur_syl/np.sum(phoneme_score_durs_syl_vars_ii)
                        boundaries_phoneme_syl_vars_ii, pp_ii = \
                            viterbiDecodingPhonemeSeg.viterbiSegmentalPenalized(obs_phoneme_syl,
                                                                                phoneme_score_durs_syl_vars_ii,
                                                                                varin)
                        posterior = pp_ii / np.power(len(phoneme_score_labels_syl_vars_ii), varin['posterior_norm'])

                        boundaries_phoneme_syl_vars.append(boundaries_phoneme_syl_vars_ii)
                        phoneme_score_durs_syl_vars_norm.append(phoneme_score_durs_syl_vars_ii)
                        posterior_vars.append(posterior)

                    # posterior vars either contain inf or nan
                    if len(posterior_vars) and np.all(np.isinf(posterior_vars)+np.isnan(posterior_vars)):
                        continue

                    idx_max_posterior = np.argmax(posterior_vars)
                    boundaries_phoneme_syl = boundaries_phoneme_syl_vars[idx_max_posterior]
                    phoneme_score_labels += phoneme_score_labels_syl_vars[idx_max_posterior]
                    phoneme_score_durs_syl = phoneme_score_durs_syl_vars_norm[idx_max_posterior]
                    # print(idx_max_posterior)

                else:
                    phoneme_score_durs_syl *= dur_syl/np.sum(phoneme_score_durs_syl)
                    phoneme_score_labels += phoneme_score_labels_grouped_by_syllables[ii_syl_boundary]
                    boundaries_phoneme_syl = \
                        viterbiDecodingPhonemeSeg.viterbiSegmental2(obs_phoneme_syl, phoneme_score_durs_syl, varin)

                # phoneme boundaries
                boundaries_phoneme_syl_start_time = \
                    (np.array(boundaries_phoneme_syl[:-1]) + frame_start_syl) * hopsize_t
                boundaries_phoneme_syl_end_time = (np.array(boundaries_phoneme_syl[1:]) + frame_start_syl) * hopsize_t

                boundaries_phoneme_start_time = \
                    np.concatenate((boundaries_phoneme_start_time, boundaries_phoneme_syl_start_time))
                boundaries_phoneme_end_time = \
                    np.concatenate((boundaries_phoneme_end_time, boundaries_phoneme_syl_end_time))

                phoneme_score_durs = np.concatenate((phoneme_score_durs, phoneme_score_durs_syl))

            phoneme_score_durs *= (time_end-time_start)/np.sum(phoneme_score_durs)

            # save the results
            results_aggregation_save_helper(syllable_gt_onsets_0start,
                                            syllable_gt_labels,
                                            boundaries_syllable_start_time,
                                            syllable_score_labels,
                                            phoneme_gt_onsets_0start,
                                            phoneme_gt_labels,
                                            boundaries_phoneme_start_time,
                                            phoneme_score_labels,
                                            eval_results_path,
                                            artist_path,
                                            fn,
                                            ii_line,
                                            time_end-time_start)

            if plot:
                figure_plot_joint(mfcc_line,
                                  syllable_gt_onsets_0start,
                                  phoneme_gt_onsets_0start_without_syllable_onsets,
                                  obs_syllable,
                                  boundaries_syllable_start_time,
                                  obs_phoneme,
                                  boundaries_phoneme_start_time,
                                  syllable_score_durs,
                                  phoneme_score_durs)

            if save_data:
                # save wav line
                data_wav, fs_wav = sf.read(wav_file)
                sf.write('./temp/wav_line_'+str(ii_line)+'.wav', data_wav, fs_wav)

                # save durations:
                pickle.dump(syllable_score_durs,
                            open('./temp/syllable_score_durs_'+str(ii_line)+'.pkl', 'w'), protocol=2)
                pickle.dump(phoneme_score_durs_grouped_by_syllables,
                            open('./temp/phoneme_score_durs_grouped_by_syllables_' + str(ii_line) + '.pkl', 'w'),
                            protocol=2)
                print(syllable_score_durs)
                print(phoneme_score_durs_grouped_by_syllables)

                # save labels:
                pickle.dump(syllable_score_labels,
                            open('./temp/syllable_score_labels_' + str(ii_line) + '.pkl', 'w'), protocol=2)
                pickle.dump(phoneme_score_labels_grouped_by_syllables,
                            open('./temp/phoneme_score_labels_grouped_by_syllables_' + str(ii_line) + '.pkl', 'w'),
                            protocol=2)
                print(syllable_score_labels)
                print(phoneme_score_labels_grouped_by_syllables)


def main():
    plot = False
    save_data = False

    # missing phoneme experiment parameters
    missing_phn = False
    missing_phn_str = 'missing_phn' if missing_phn else ''
    varin['posterior_norm'] = 1.0

    # validation and test recording names, in the paper, we use both validation and test recordings
    primarySchool_val_recordings, primarySchool_test_recordings, _, _, _, _ = get_train_test_recordings_joint()

    # load scaler
    scaler = pickle.load(open(scaler_joint_model_path, 'rb'))

    # repeat the experiment 5 times
    for ii in range(0, 5):
        model_keras_cnn_0 = load_model(full_path_keras_cnn_0 + str(ii) + '.h5')

        onset_function_all_recordings(wav_path=primarySchool_wav_path,
                                      textgrid_path=primarySchool_textgrid_path,
                                      scaler=scaler,
                                      test_recordings=primarySchool_val_recordings+primarySchool_test_recordings,
                                      model_keras_cnn_0=model_keras_cnn_0,
                                      cnnModel_name=cnnModel_name+'_'+str(ii),
                                      eval_results_path=eval_results_path+'_'+str(ii),
                                      obs_cal='tocal',
                                      plot=plot,
                                      save_data=save_data,
                                      missing_phn=missing_phn)

        # calculate the evaluation results
        run_eval_onset('joint', missing_phn_str+'_'+str(ii), 'test')
        run_eval_segment('joint', missing_phn_str+'_'+str(ii), 'test')

        run_eval_onset('joint', missing_phn_str + '_' + str(ii), 'all')
        run_eval_segment('joint', missing_phn_str + '_' + str(ii), 'all')


if __name__ == '__main__':
    main()