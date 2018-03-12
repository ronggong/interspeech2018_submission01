import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pyximport
pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

from lyricsRecognizer.LRHSMM import LRHSMM
from lyricsRecognizer.makeHSMMNet import singleTransMatBuild
from audio_preprocessing import get_log_mel_madmom
from audio_preprocessing import feature_reshape

from general.utilFunctions import remove_silence
from general.filePathHsmm import *
from general.parameters import *
from general.utilFunctions import textgrid_syllable_phoneme_parser
from general.trainTestSeparation import get_train_test_recordings_joint
from general.phonemeMap import dic_pho_map
from helper_code import results_aggregation_save_helper
from helper_code import gt_score_preparation_helper
from helper_code import findShiftOffset
from plot_code import figure_plot_hsmm
from onsetSegmentEval.runEval import run_eval_onset
from onsetSegmentEval.runEval import run_eval_segment


def phoneme_seg_all_recordings(wav_path,
                               textgrid_path,
                               scaler,
                               scaler_joint,
                               test_recordings,
                               model_keras_cnn_0,
                               model_joint,
                               eval_results_path,
                               use_joint_obs=False,
                               plot=False,
                               debug_mode=False):
    """
    :param wav_path:
    :param textgrid_path:
    :param scaler:
    :param scaler_joint: onset detection joing model scaler, for experiment, not included in the paper
    :param test_recordings:
    :param model_keras_cnn_0:
    :param model_joint: onset detection joint model, for experiment, not included in the paper
    :param eval_results_path:
    :param use_joint_obs: bool
    :param plot: bool
    :param debug_mode: bool
    :return:
    """

    for artist_path, fn in test_recordings:

        print('Calculating for artist:', artist_path, 'filename:', fn)

        score_textgrid_file = join(textgrid_path, artist_path, 'teacher.TextGrid')
        groundtruth_textgrid_file   = join(textgrid_path, artist_path, fn+'.TextGrid')
        wav_file = join(wav_path, artist_path, fn + '.wav')
        scoreSyllableLists, scorePhonemeLists = textgrid_syllable_phoneme_parser(score_textgrid_file,
                                                                                 'dianSilence',
                                                                                 'details')
        gtSyllableLists, gtPhonemeLists = textgrid_syllable_phoneme_parser(groundtruth_textgrid_file,
                                                                           'dianSilence',
                                                                           'details')

        # calculate mfcc
        mfcc = get_log_mel_madmom(wav_file, fs, hopsize_t, channel=1)
        mfcc_scaled = scaler.transform(mfcc)
        mfcc_reshaped = feature_reshape(mfcc_scaled, nlen=7)

        if use_joint_obs:
            mfcc_scaled_joint = scaler_joint.transform(mfcc)
            mfcc_reshaped_joint = feature_reshape(mfcc_scaled_joint, nlen=7)

        for ii_line in range(len(gtSyllableLists)):
            print('line:', ii_line)

            # search for the corresponding score line
            ii_aug = findShiftOffset(gtSyllableLists, scoreSyllableLists, ii_line)

            frame_start, frame_end, \
            time_start, time_end, \
            syllable_gt_onsets, syllable_gt_labels, \
            phoneme_gt_onsets, phoneme_gt_labels, \
            syllable_score_onsets, syllable_score_labels, \
            phoneme_score_onsets, phoneme_score_labels, \
            syllable_score_durs, phoneme_list_score = \
                            gt_score_preparation_helper(gtSyllableLists,
                                                        scoreSyllableLists,
                                                        gtPhonemeLists,
                                                        scorePhonemeLists,
                                                        ii_line,
                                                        ii_aug)

            # phoneme durations and labels
            phoneme_score_durs = []
            # index of syllable onsets in phoneme onsets list
            idx_syllable_score_phoneme = []
            for ii_pls, pls in enumerate(phoneme_list_score):
                # when the phoneme onset time is also the syllable onset time
                phoneme_score_durs.append(pls[1] - pls[0])

                if pls[0] in syllable_score_onsets:
                    idx_syllable_score_phoneme.append(ii_pls)

            # map the phone labels
            phoneme_score_labels_mapped = [dic_pho_map[l] for l in phoneme_score_labels]

            # normalize phoneme score durations
            phoneme_score_durs = np.array(phoneme_score_durs)
            phoneme_score_durs *= (time_end - time_start) / np.sum(phoneme_score_durs)

            # onsets start from time 0, syllable and phoneme onsets
            syllable_gt_onsets_0start = np.array(syllable_gt_onsets) - syllable_gt_onsets[0]
            phoneme_gt_onsets_0start = np.array(phoneme_gt_onsets) - phoneme_gt_onsets[0]
            phoneme_gt_onsets_0start_without_syllable_onsets = \
                np.setdiff1d(phoneme_gt_onsets_0start, syllable_gt_onsets_0start)

            # check the annotations, if syllable onset are also phoneme onsets
            if not set(syllable_gt_onsets).issubset(set(phoneme_gt_onsets)):
                raise
            if not set(syllable_score_onsets).issubset(set(phoneme_score_onsets)):
                raise

            # line level mfcc
            mfcc_line = mfcc[frame_start:frame_end]
            mfcc_reshaped_line = mfcc_reshaped[frame_start:frame_end]
            mfcc_reshaped_line = np.expand_dims(mfcc_reshaped_line, axis=1)

            if use_joint_obs:
                mfcc_reshaped_line_joint = mfcc_reshaped_joint[frame_start:frame_end]
                mfcc_reshaped_line_joint = np.expand_dims(mfcc_reshaped_line_joint, axis=1)
                _, obs_joint_phoneme = model_joint.predict(mfcc_reshaped_line_joint, batch_size=128, verbose=2)

                obs_joint_phoneme = obs_joint_phoneme[:, 0]
                # obs_joint_phoneme[:20] = 0.0
            else:
                obs_joint_phoneme = None

            # transition matrix
            mat_tran = singleTransMatBuild(phoneme_score_labels_mapped)

            # initialize the the HSMM
            # set proportionality to 0.2 in some sample will break
            hsmm = LRHSMM(mat_tran,
                          phoneme_score_labels_mapped,
                          phoneme_score_durs,
                          proportionality_std=0.2)

            # calculate observation
            hsmm.mapBKeras(observations=mfcc_reshaped_line,
                           kerasModel=model_keras_cnn_0,
                           obs_onset_phn=obs_joint_phoneme,
                           use_joint_obs=use_joint_obs,
                           debug_mode=debug_mode)

            forwardDelta, \
            previousState, \
            state, \
            stateIn, \
            occupancy, \
            tau = hsmm._inferenceInit(observations=mfcc_reshaped_line)

            path, posteri_proba = hsmm._viterbiHSMM(forwardDelta,
                                                    previousState,
                                                    state,
                                                    stateIn,
                                                    occupancy,
                                                    tau,
                                                    obsOnsetPhn=obs_joint_phoneme)

            # construct ground truth path
            phoneme_gt_onsets_0start_frame = list(np.floor(phoneme_gt_onsets_0start * (len(path)/(time_end-time_start))))
            path_gt = np.zeros((len(path),), dtype='int')
            state_num = 0
            for ii_path in range(len(path)):
                if ii_path in phoneme_gt_onsets_0start_frame[1:]:
                    state_num += 1
                path_gt[ii_path] = state_num

            # detected phoneme onsets
            phoneme_start_frame = [0]
            for ii_path in range(len(path)-1):
                if path[ii_path] != path[ii_path+1]:
                    phoneme_start_frame.append(ii_path+1)

            boundaries_phoneme_start_time = list(np.array(phoneme_start_frame)*(time_end-time_start)/len(path))
            boundaries_syllable_start_time = [boundaries_phoneme_start_time[ii_bpst]
                                              for ii_bpst in range(len(boundaries_phoneme_start_time))
                                              if ii_bpst in idx_syllable_score_phoneme]

            # remove the silence from the score and the ground truth onset time
            if u'' in phoneme_gt_labels:
                phoneme_gt_onsets_0start, phoneme_gt_labels = remove_silence(phoneme_gt_onsets_0start, phoneme_gt_labels)

            if u'' in phoneme_score_labels:
                boundaries_phoneme_start_time, phoneme_score_labels = remove_silence(boundaries_phoneme_start_time, phoneme_score_labels)

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
                figure_plot_hsmm(mfcc_line,
                                 syllable_gt_onsets_0start,
                                 phoneme_gt_onsets_0start_without_syllable_onsets,
                                 hsmm,
                                 phoneme_score_labels_mapped,
                                 path,
                                 boundaries_phoneme_start_time,
                                 boundaries_syllable_start_time,
                                 syllable_score_durs,
                                 phoneme_score_durs,
                                 obs_joint_phoneme)


def main():
    import pickle

    use_joint_obs = False

    joint_obs_str = '_joint_obs_e6' if use_joint_obs else ''

    plot = False

    debug_mode = False  # in debug mode, you will have the plots of the observation matrix

    primarySchool_val_recordings, primarySchool_test_recordings, _, _, _, _ = get_train_test_recordings_joint()

    scaler = pickle.load(open(kerasScaler_path, 'rb'))

    if use_joint_obs:
        # load joint models
        from general.filePathJoint import scaler_joint_model_path
        from general.filePathJoint import full_path_keras_cnn_0
        scaler_joint = pickle.load(open(scaler_joint_model_path, 'rb'))
        model_joint = LRHSMM.kerasModel(full_path_keras_cnn_0 + str(0) + '.h5')
    else:
        scaler_joint = None
        model_joint = None

    for ii in range(0, 5):
        model_keras_cnn_0 = LRHSMM.kerasModel(kerasModels_path+'_'+str(ii)+'.h5')

        phoneme_seg_all_recordings(wav_path=primarySchool_wav_path,
                                   textgrid_path=primarySchool_textgrid_path,
                                   scaler=scaler,
                                   scaler_joint=scaler_joint,
                                   test_recordings=primarySchool_test_recordings,
                                   model_keras_cnn_0=model_keras_cnn_0,
                                   model_joint=model_joint,
                                   eval_results_path=eval_results_path+joint_obs_str+'_'+str(ii),
                                   use_joint_obs=use_joint_obs,
                                   plot=plot,
                                   debug_mode=debug_mode)

        run_eval_onset('hsmm', joint_obs_str+'_'+str(ii), 'test')
        run_eval_segment('hsmm', joint_obs_str+'_'+str(ii), 'test')
        run_eval_onset('hsmm', joint_obs_str + '_' + str(ii), 'all')
        run_eval_segment('hsmm', joint_obs_str + '_' + str(ii), 'all')


if __name__ == '__main__':
    main()
