"""
Run this script to extract features for training joint model,
You need to set up all the paths in filePathJoint.py
"""
import cPickle
import gzip
import os
import pickle

import h5py
import numpy as np
from sklearn import preprocessing

from filePathJoint import *
from parameters import *
from audio_preprocessing import feature_reshape
from audio_preprocessing import get_log_mel_madmom
from textgridParser import textGrid2WordList, wordListsParseByLines
from trainTestSeparation import get_train_test_recordings_joint
from trainTestSeparation import get_train_test_recordings_joint_subset


def remove_out_of_range(frames, frame_start, frame_end):
    return frames[np.all([frames <= frame_end, frames >= frame_start], axis=0)]


def simple_sample_weighting(mfcc, frames_onset_s_p, frames_onset_p, frame_start, frame_end):
    """
    simple weighing strategy used in Schluter's paper
    :param mfcc:
    :param frames_onset:
    :param frame_start:
    :param frame_end:
    :return:
    """

    frames_onset_s_p25 = np.hstack((frames_onset_s_p - 1, frames_onset_s_p + 1))
    frames_onset_s_p25 = remove_out_of_range(frames_onset_s_p25, frame_start, frame_end)

    frames_onset_p25 = np.hstack((frames_onset_p - 1, frames_onset_p + 1))
    frames_onset_p25 = remove_out_of_range(frames_onset_p25, frame_start, frame_end)

    # mfcc positive
    mfcc_s_p100 = mfcc[frames_onset_s_p, :]
    mfcc_s_p25 = mfcc[frames_onset_s_p25, :]

    mfcc_p100 = mfcc[frames_onset_p, :]
    mfcc_p25 = mfcc[frames_onset_p25, :]

    frames_all = np.arange(frame_start, frame_end)
    frames_n100 = np.setdiff1d(frames_all, np.hstack((frames_onset_s_p,
                                                      frames_onset_s_p25,
                                                      frames_onset_p,
                                                      frames_onset_p25)))
    mfcc_n100 = mfcc[frames_n100, :]

    mfcc_s_p = np.concatenate((mfcc_s_p100, mfcc_s_p25), axis=0)
    sample_weights_s_p = np.concatenate((np.ones((mfcc_s_p100.shape[0],)),
                                       np.ones((mfcc_s_p25.shape[0],)) * 0.25))

    mfcc_p = np.concatenate((mfcc_p100, mfcc_p25), axis=0)
    sample_weights_p_syllable = np.concatenate((np.ones((mfcc_p100.shape[0],)),
                                                np.ones((mfcc_p25.shape[0],))))

    sample_weights_p_phoneme = np.concatenate((np.ones((mfcc_p100.shape[0],)),
                                                np.ones((mfcc_p25.shape[0],)) * 0.25))

    mfcc_n = mfcc_n100
    sample_weights_n = np.ones((mfcc_n100.shape[0],))

    return mfcc_s_p, mfcc_p, mfcc_n, sample_weights_s_p, \
           sample_weights_p_syllable, sample_weights_p_phoneme, sample_weights_n


def feature_label_onset_h5py(filename_mfcc_s_p, filename_mfcc_p, filename_mfcc_n, scaling=True):
    """
    organize the training feature and label
    :param
    :return:
    """

    f_mfcc_s_p = h5py.File(filename_mfcc_s_p, 'a')
    f_mfcc_p = h5py.File(filename_mfcc_p, 'r')
    f_mfcc_n = h5py.File(filename_mfcc_n, 'r')

    dim_s_p_0 = f_mfcc_s_p['mfcc_s_p'].shape[0]
    dim_p_0 = f_mfcc_p['mfcc_p'].shape[0]
    dim_n_0 = f_mfcc_n['mfcc_n'].shape[0]
    dim_1 = f_mfcc_p['mfcc_p'].shape[1]

    label_s_p = [1] * dim_s_p_0
    label_p_syllable = [0] * dim_p_0  # phoneme onset label for syllable detection
    label_p_phoneme = [1] * dim_p_0  # phoneme onset label for phoneme detection
    label_n = [0] * dim_n_0

    label_all_syllable = label_s_p + label_p_syllable + label_n
    label_all_phoneme = label_s_p + label_p_phoneme + label_n

    label_all_syllable = np.array(label_all_syllable,dtype='int64')
    label_all_phoneme = np.array(label_all_phoneme,dtype='int64')

    feature_all = np.zeros((dim_s_p_0+dim_p_0+dim_n_0, dim_1), dtype='float32')

    print('concatenate features... ...')

    feature_all[:dim_s_p_0, :] = f_mfcc_s_p['mfcc_s_p']
    feature_all[dim_s_p_0:dim_s_p_0+dim_p_0, :] = f_mfcc_p['mfcc_p']
    feature_all[dim_s_p_0+dim_p_0:, :] = f_mfcc_n['mfcc_n']

    f_mfcc_s_p.flush()
    f_mfcc_s_p.close()
    f_mfcc_p.flush()
    f_mfcc_p.close()
    f_mfcc_n.flush()
    f_mfcc_n.close()

    print('scaling features... ... ')

    scaler = preprocessing.StandardScaler()
    scaler.fit(feature_all)
    if scaling:
        feature_all = scaler.transform(feature_all)

    return feature_all, label_all_syllable, label_all_phoneme, scaler


def dump_feature_onset_helper(wav_path,
                              textgrid_path,
                              artist_name,
                              recording_name):

    groundtruth_textgrid_file = os.path.join(textgrid_path, artist_name, recording_name + '.TextGrid')
    wav_file = os.path.join(wav_path, artist_name, recording_name + '.wav')

    lineList = textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
    utteranceList = textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')
    phonemeList = textGrid2WordList(groundtruth_textgrid_file, whichTier='details')

    # parse lines of groundtruth
    nestedUtteranceLists, numLines, numUtterances = wordListsParseByLines(lineList, utteranceList)
    nestedPhonemeLists, _, _ = wordListsParseByLines(lineList, phonemeList)

    # load audio
    mfcc = get_log_mel_madmom(wav_file, fs, hopsize_t, channel=1)

    return nestedUtteranceLists, nestedPhonemeLists, mfcc, phonemeList


def get_frame_onset(u_list):

    times_onset = [u[0] for u in u_list[1]]

    # syllable onset frames
    frames_onset = np.array(np.around(np.array(times_onset) / hopsize_t), dtype=int)

    # line start and end frames
    frame_start = frames_onset[0]
    frame_end = int(u_list[0][1] / hopsize_t)

    return frames_onset, frame_start, frame_end


def dump_feature_onset(wav_path,
                       textgrid_path,
                       recordings):
    """
    :param wav_path:
    :param textgrid_path:
    :param recordings:
    :return:
    """

    # p: position, n: negative, 75: 0.75 sample_weight
    mfcc_s_p_all = []
    mfcc_p_all = []
    mfcc_n_all = []
    sample_weights_s_p_all = []
    sample_weights_p_syllable_all = []
    sample_weights_p_phoneme_all = []
    sample_weights_n_all = []

    for artist_name, recording_name in recordings:

        nestedUtteranceLists, nestedPhonemeLists, mfcc, phonemeList = \
            dump_feature_onset_helper(wav_path, textgrid_path, artist_name, recording_name)

        for ii_line, line in enumerate(nestedUtteranceLists):
            list_syllable = nestedUtteranceLists[ii_line]
            list_phoneme = nestedPhonemeLists[ii_line]

            # pass if no syllable in this line
            if not len(list_syllable[1]):
                continue

            onsets_syllable, frame_start_line_syllable, frame_end_line_syllable = get_frame_onset(list_syllable)
            onsets_phoneme, frame_start_line_phoneme, frame_end_line_phoneme = get_frame_onset(list_phoneme)

            if not set(onsets_syllable).issubset(onsets_phoneme) or \
               frame_start_line_syllable != frame_start_line_phoneme or\
               frame_end_line_syllable != frame_end_line_phoneme:
                raise

            frames_onset_s_p = onsets_syllable # simultaneously syllable and phoneme onsets
            # only phoneme onsets
            frames_onset_p = np.array([o for o in onsets_phoneme if o not in onsets_syllable], dtype=int)

            mfcc_s_p, \
            mfcc_p, \
            mfcc_n, \
            sample_weights_s_p, \
            sample_weights_p_syllable, \
            sample_weights_p_phoneme, \
            sample_weights_n = \
                simple_sample_weighting(mfcc, frames_onset_s_p, frames_onset_p,
                                        frame_start_line_syllable, frame_end_line_syllable)

            mfcc_s_p_all.append(mfcc_s_p)
            mfcc_p_all.append(mfcc_p)
            mfcc_n_all.append(mfcc_n)
            sample_weights_s_p_all.append(sample_weights_s_p)
            sample_weights_p_syllable_all.append(sample_weights_p_syllable)
            sample_weights_p_phoneme_all.append(sample_weights_p_phoneme)
            sample_weights_n_all.append(sample_weights_n)

            # print(len(mfcc_p_all), len(mfcc_n_all), len(sample_weights_p_all), len(sample_weights_n_all))

    return np.concatenate(mfcc_s_p_all), \
           np.concatenate(mfcc_p_all), \
           np.concatenate(mfcc_n_all), \
           np.concatenate(sample_weights_s_p_all), \
           np.concatenate(sample_weights_p_syllable_all), \
           np.concatenate(sample_weights_p_phoneme_all), \
           np.concatenate(sample_weights_n_all)


def dump_feature_batch_onset_subset():
    """
    dump features for onset detection a subset
    :return:
    """

    trainNacta2017, trainNacta = get_train_test_recordings_joint_subset()

    nacta_data = trainNacta
    nacta_data_2017 = trainNacta2017
    scaling = True

    mfcc_s_p_nacta2017, \
    mfcc_p_nacta2017,\
    mfcc_n_nacta2017, \
    sample_weights_s_p_nacta2017, \
    sample_weights_p_syllable_nacta2017, \
    sample_weights_p_phoneme_nacta2017, \
    sample_weights_n_nacta2017 = \
        dump_feature_onset(wav_path=nacta2017_wav_path,
                           textgrid_path=nacta2017_textgrid_path,
                           recordings=nacta_data_2017)

    mfcc_s_p_nacta, \
    mfcc_p_nacta, \
    mfcc_n_nacta, \
    sample_weights_s_p_nacta, \
    sample_weights_p_syllable_nacta, \
    sample_weights_p_phoneme_nacta, \
    sample_weights_n_nacta =    \
        dump_feature_onset(wav_path=nacta_wav_path,
                           textgrid_path=nacta_textgrid_path,
                           recordings=nacta_data)

    print('finished feature extraction.')

    mfcc_s_p = np.concatenate((mfcc_s_p_nacta2017, mfcc_s_p_nacta))
    mfcc_p = np.concatenate((mfcc_p_nacta2017, mfcc_p_nacta))
    mfcc_n = np.concatenate((mfcc_n_nacta2017, mfcc_n_nacta))

    sample_weights_s_p = np.concatenate((sample_weights_s_p_nacta2017, sample_weights_s_p_nacta))
    sample_weights_p_syllable = np.concatenate((sample_weights_p_syllable_nacta2017, sample_weights_p_syllable_nacta))
    sample_weights_p_phoneme = np.concatenate((sample_weights_p_phoneme_nacta2017, sample_weights_p_phoneme_nacta))
    sample_weights_n = np.concatenate((sample_weights_n_nacta2017, sample_weights_n_nacta))

    sample_weights_syllable = np.concatenate((sample_weights_s_p, sample_weights_p_syllable, sample_weights_n))
    sample_weights_phoneme = np.concatenate((sample_weights_s_p, sample_weights_p_phoneme, sample_weights_n))

    filename_mfcc_s_p = join(training_data_joint_path, 'mfcc_s_p_joint.h5')
    h5f = h5py.File(filename_mfcc_s_p, 'w')
    h5f.create_dataset('mfcc_s_p', data=mfcc_s_p)
    h5f.close()

    filename_mfcc_p = join(training_data_joint_path, 'mfcc_p_joint.h5')
    h5f = h5py.File(filename_mfcc_p, 'w')
    h5f.create_dataset('mfcc_p', data=mfcc_p)
    h5f.close()

    filename_mfcc_n = join(training_data_joint_path, 'mfcc_n_joint.h5')
    h5f = h5py.File(filename_mfcc_n, 'w')
    h5f.create_dataset('mfcc_n', data=mfcc_n)
    h5f.close()

    del mfcc_s_p
    del mfcc_p
    del mfcc_n

    feature_all, label_all_syllable, label_all_phoneme, scaler = \
        feature_label_onset_h5py(filename_mfcc_s_p, filename_mfcc_p, filename_mfcc_n, scaling=scaling)

    os.remove(filename_mfcc_s_p)
    os.remove(filename_mfcc_p)
    os.remove(filename_mfcc_n)

    nlen = 7
    feature_all = feature_reshape(feature_all, nlen=nlen)

    print('feature shape:', feature_all.shape)

    filename_feature_all = join(training_data_joint_path, 'feature_all_joint_subset.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    print('finished feature concatenation.')

    cPickle.dump(label_all_syllable,
                 gzip.open(
                     './training_data/labels_joint_syllable_subset.pickle.gz',
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(label_all_phoneme,
                 gzip.open(
                     './training_data/labels_joint_phoneme_subset.pickle.gz',
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights_syllable,
                 gzip.open('./training_data/sample_weights_joint_syllable_subset.pickle.gz',
                           'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights_phoneme,
                 gzip.open('./training_data/sample_weights_joint_phoneme_subset.pickle.gz',
                           'wb'), cPickle.HIGHEST_PROTOCOL)

    print(feature_all.shape)
    print(label_all_syllable.shape, label_all_phoneme.shape)
    print(sample_weights_syllable.shape, sample_weights_phoneme.shape)

    pickle.dump(scaler,
                open('./cnnModels/scaler_joint_subset.pkl', 'wb'))


def dump_Feature_batch_onset():
    """
    dump features for all the dataset for onset detection
    :return:
    """

    valPrimarySchool, testPrimarySchool, trainNacta2017, trainNacta, trainPrimarySchool, _ \
        = get_train_test_recordings_joint()

    nacta_data = trainNacta
    nacta_data_2017 = trainNacta2017
    scaling = True

    mfcc_s_p_nacta2017, \
    mfcc_p_nacta2017,\
    mfcc_n_nacta2017, \
    sample_weights_s_p_nacta2017, \
    sample_weights_p_syllable_nacta2017, \
    sample_weights_p_phoneme_nacta2017, \
    sample_weights_n_nacta2017 = \
        dump_feature_onset(wav_path=nacta2017_wav_path,
                           textgrid_path=nacta2017_textgrid_path,
                           recordings=nacta_data_2017)

    mfcc_s_p_nacta, \
    mfcc_p_nacta, \
    mfcc_n_nacta, \
    sample_weights_s_p_nacta, \
    sample_weights_p_syllable_nacta, \
    sample_weights_p_phoneme_nacta, \
    sample_weights_n_nacta =    \
        dump_feature_onset(wav_path=nacta_wav_path,
                           textgrid_path=nacta_textgrid_path,
                           recordings=nacta_data)

    mfcc_s_p_primary, \
    mfcc_p_primary, \
    mfcc_n_primary, \
    sample_weights_s_p_primary, \
    sample_weights_p_syllable_primary, \
    sample_weights_p_phoneme_primary, \
    sample_weights_n_primary = \
        dump_feature_onset(wav_path=primarySchool_wav_path,
                           textgrid_path=primarySchool_textgrid_path,
                           recordings=trainPrimarySchool)

    print('finished feature extraction.')

    mfcc_s_p = np.concatenate((mfcc_s_p_nacta2017, mfcc_s_p_nacta, mfcc_s_p_primary))
    mfcc_p = np.concatenate((mfcc_p_nacta2017, mfcc_p_nacta, mfcc_p_primary))
    mfcc_n = np.concatenate((mfcc_n_nacta2017, mfcc_n_nacta, mfcc_n_primary))

    sample_weights_s_p = \
        np.concatenate((sample_weights_s_p_nacta2017, sample_weights_s_p_nacta, sample_weights_s_p_primary))
    sample_weights_p_syllable = \
        np.concatenate((sample_weights_p_syllable_nacta2017,
                        sample_weights_p_syllable_nacta, sample_weights_p_syllable_primary))
    sample_weights_p_phoneme = \
        np.concatenate((sample_weights_p_phoneme_nacta2017,
                        sample_weights_p_phoneme_nacta, sample_weights_p_phoneme_primary))
    sample_weights_n = \
        np.concatenate((sample_weights_n_nacta2017, sample_weights_n_nacta, sample_weights_n_primary))

    sample_weights_syllable = np.concatenate((sample_weights_s_p, sample_weights_p_syllable, sample_weights_n))
    sample_weights_phoneme = np.concatenate((sample_weights_s_p, sample_weights_p_phoneme, sample_weights_n))

    filename_mfcc_s_p = join(training_data_joint_path, 'mfcc_s_p_joint.h5')
    h5f = h5py.File(filename_mfcc_s_p, 'w')
    h5f.create_dataset('mfcc_s_p', data=mfcc_s_p)
    h5f.close()

    filename_mfcc_p = join(training_data_joint_path, 'mfcc_p_joint.h5')
    h5f = h5py.File(filename_mfcc_p, 'w')
    h5f.create_dataset('mfcc_p', data=mfcc_p)
    h5f.close()

    filename_mfcc_n = join(training_data_joint_path, 'mfcc_n_joint.h5')
    h5f = h5py.File(filename_mfcc_n, 'w')
    h5f.create_dataset('mfcc_n', data=mfcc_n)
    h5f.close()

    del mfcc_s_p
    del mfcc_p
    del mfcc_n

    feature_all, label_all_syllable, label_all_phoneme, scaler = \
        feature_label_onset_h5py(filename_mfcc_s_p, filename_mfcc_p, filename_mfcc_n, scaling=scaling)

    os.remove(filename_mfcc_s_p)
    os.remove(filename_mfcc_p)
    os.remove(filename_mfcc_n)

    nlen = 7
    feature_all = feature_reshape(feature_all, nlen=nlen)

    print('feature shape:', feature_all.shape)

    filename_feature_all = join(training_data_joint_path, 'feature_all_joint.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    print('finished feature concatenation.')

    cPickle.dump(label_all_syllable,
                 gzip.open(
                     './training_data/labels_joint_syllable.pickle.gz',
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(label_all_phoneme,
                 gzip.open(
                     './training_data/labels_joint_phoneme.pickle.gz',
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights_syllable,
                 gzip.open('./training_data/sample_weights_joint_syllable.pickle.gz',
                           'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights_phoneme,
                 gzip.open('./training_data/sample_weights_joint_phoneme.pickle.gz',
                           'wb'), cPickle.HIGHEST_PROTOCOL)

    print(feature_all.shape)
    print(label_all_syllable.shape, label_all_phoneme.shape)
    print(sample_weights_syllable.shape, sample_weights_phoneme.shape)

    pickle.dump(scaler,
                open('./cnnModels/scaler_joint.pkl', 'wb'))


if __name__ == '__main__':
    dump_Feature_batch_onset()