"""
Run this script to extract features for training hsmm model,
You need to set up all the paths in filePathHsmm.py
"""

import h5py
import cPickle
import gzip
import pickle
from sklearn import preprocessing
import numpy as np

from filePathHsmm import *
from parameters import *
from audio_preprocessing import feature_reshape
from audio_preprocessing import get_log_mel_madmom
from phonemeMap import dic_pho_map, dic_pho_label
from textgridParser import syllableTextgridExtraction

from trainTestSeparation import get_train_test_recordings_joint


def dump_feature_phn(wav_path,
                     textgrid_path,
                     recordings,
                     syllableTierName,
                     phonemeTierName):
    """
    Dump feature for each phoneme
    :param wav_path:
    :param textgrid_path:
    :param recordings:
    :param syllableTierName:
    :param phonemeTierName:
    :return:
    """

    dic_pho_feature = {}

    for _,pho in enumerate(set(dic_pho_map.values())):
        dic_pho_feature[pho] = np.array([])

    for artist_path, recording in recordings:
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path=textgrid_path,
                                         recording=join(artist_path,recording),
                                         tier0=syllableTierName,
                                         tier1=phonemeTierName)

        # audio
        wav_full_filename   = join(wav_path,artist_path,recording+'.wav')

        log_mel = get_log_mel_madmom(wav_full_filename, fs, hopsize_t, channel=1)

        for ii,pho in enumerate(nestedPhonemeLists):
            print 'calculating ', recording, ' and phoneme ', str(ii), ' of ', str(len(nestedPhonemeLists))
            for p in pho[1]:
                # map from annotated xsampa to readable notation
                try:
                    key = dic_pho_map[p[2]]
                except KeyError:
                    print(artist_path, recording)
                    print(ii, p[2])
                    raise

                sf = int(round(p[0] * fs / float(hopsize))) # starting frame
                ef = int(round(p[1] * fs / float(hopsize))) # ending frame

                log_mel_phn = log_mel[sf:ef,:]  # phoneme syllable

                if not len(dic_pho_feature[key]):
                    dic_pho_feature[key] = log_mel_phn
                else:
                    dic_pho_feature[key] = np.vstack((dic_pho_feature[key],log_mel_phn))

    return dic_pho_feature


def feature_aggregator(dic_pho_feature_train):
    """
    aggregate feature dictionary into numpy feature, label lists,
    reshape the feature
    :param dic_pho_feature_train:
    :return:
    """
    feature_all = np.array([], dtype='float32')
    label_all = []
    for key in dic_pho_feature_train:
        feature = dic_pho_feature_train[key]
        label = [dic_pho_label[key]] * len(feature)

        if len(feature):
            if not len(feature_all):
                feature_all = feature
            else:
                feature_all = np.vstack((feature_all, feature))
            label_all += label
    label_all = np.array(label_all, dtype='int64')

    scaler = preprocessing.StandardScaler().fit(feature_all)
    feature_all = scaler.transform(feature_all)
    feature_all = feature_reshape(feature_all, nlen=7)

    return feature_all, label_all, scaler


def batch_dump():
    _, testPrimarySchool, trainNacta2017, trainNacta, trainPrimarySchool, trainSepa = get_train_test_recordings_joint()

    dic_pho_feature_nacta2017 = dump_feature_phn(wav_path=nacta2017_wav_path,
                                                 textgrid_path=nacta2017_textgrid_path,
                                                 recordings=trainNacta2017,
                                                 syllableTierName='line',
                                                 phonemeTierName='details')

    dic_pho_feature_nacta = dump_feature_phn(wav_path=nacta_wav_path,
                                             textgrid_path=nacta_textgrid_path,
                                             recordings=trainNacta,
                                             syllableTierName='line',
                                             phonemeTierName='details')

    dic_pho_feature_primarySchool = dump_feature_phn(wav_path=primarySchool_wav_path,
                                                     textgrid_path=primarySchool_textgrid_path,
                                                     recordings=trainPrimarySchool,
                                                     syllableTierName='line',
                                                     phonemeTierName='details')

    dic_pho_feature_sepa = dump_feature_phn(wav_path=nacta_wav_path,
                                            textgrid_path=nacta_textgrid_path,
                                            recordings=trainSepa,
                                            syllableTierName='line',
                                            phonemeTierName='details')

    # fuse two dictionaries
    list_key = list(set(dic_pho_feature_nacta.keys() + dic_pho_feature_nacta2017.keys() +
                        dic_pho_feature_primarySchool.keys() + dic_pho_feature_sepa.keys()))

    dic_pho_feature_all = {}
    for key in list_key:
        if not len(dic_pho_feature_nacta2017[key]):
            dic_pho_feature_nacta2017[key] = np.empty((0, 1200), dtype='float32')

        if not len(dic_pho_feature_nacta[key]):
            dic_pho_feature_nacta[key] = np.empty((0, 1200), dtype='float32')

        if not len(dic_pho_feature_primarySchool[key]):
            dic_pho_feature_primarySchool[key] = np.empty((0, 1200), dtype='float32')

        if not len(dic_pho_feature_sepa[key]):
            dic_pho_feature_sepa[key] = np.empty((0, 1200), dtype='float32')

        dic_pho_feature_all[key] = np.vstack((dic_pho_feature_nacta[key], dic_pho_feature_nacta2017[key],
                                              dic_pho_feature_primarySchool[key], dic_pho_feature_sepa[key]))

    feature_all, label_all, scaler = feature_aggregator(dic_pho_feature_all)

    # save feature label scaler
    filename_feature_all = join(training_data_hsmm_path, 'feature_hsmm_am.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    cPickle.dump(label_all,
                 gzip.open(
                     join(training_data_hsmm_path, 'labels_hsmm_am.pickle.gz'),
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    pickle.dump(scaler,
                open(join(training_data_hsmm_path, 'scaler_hsmm_am.pkl'), 'wb'))


if __name__ == '__main__':
    batch_dump()
