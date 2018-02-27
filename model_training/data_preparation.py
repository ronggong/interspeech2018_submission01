import cPickle
import gzip
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(filename_labels_train_validation_set):

    """load training and validation data"""

    with gzip.open(filename_labels_train_validation_set, 'rb') as f:
        Y_train_validation = cPickle.load(f)

    # this is the filename indices
    indices_features = range(len(Y_train_validation))

    indices_train, indices_validation, Y_train, Y_validation = \
        train_test_split(indices_features, Y_train_validation,
                         test_size=0.1, stratify=Y_train_validation)



    return indices_train, Y_train, \
           indices_validation, Y_validation, \
           indices_features, Y_train_validation


def load_data_joint(filename_labels_syllable_train_validation_set,
                    filename_labels_phoneme_train_validation_set,
                    filename_sample_weights_syllable,
                    filename_sample_weights_phoneme):

    # load training and validation data

    with gzip.open(filename_labels_syllable_train_validation_set, 'rb') as f:
        Y_syllable_train_validation = cPickle.load(f)

    with gzip.open(filename_labels_phoneme_train_validation_set, 'rb') as f:
        Y_phoneme_train_validation = cPickle.load(f)

    with gzip.open(filename_sample_weights_syllable, 'rb') as f:
        sample_weights_syllable = cPickle.load(f)

    with gzip.open(filename_sample_weights_phoneme, 'rb') as f:
        sample_weights_phoneme = cPickle.load(f)

    indices_all = range(len(Y_syllable_train_validation))

    # split indices to 3 parts
    idx_s_p = np.where(Y_syllable_train_validation == 1)[0]
    idx_n = np.where(Y_phoneme_train_validation == 0)[0]
    idx_p = np.setdiff1d(indices_all, np.concatenate((idx_s_p, idx_n), axis=0))

    # split each part into train and val sets
    idx_s_p_val = np.random.choice(idx_s_p, int(round(0.1*len(idx_s_p))), replace=False)
    idx_s_p_train = np.setdiff1d(idx_s_p, idx_s_p_val)

    idx_p_val = np.random.choice(idx_p, int(round(0.1*len(idx_p))), replace=False)
    idx_p_train = np.setdiff1d(idx_p, idx_p_val)

    idx_n_val = np.random.choice(idx_n, int(round(0.1 * len(idx_n))), replace=False)
    idx_n_train = np.setdiff1d(idx_n, idx_n_val)

    idx_val = np.concatenate((idx_s_p_val, idx_p_val, idx_n_val))
    idx_train = np.concatenate((idx_s_p_train, idx_p_train, idx_n_train))

    Y_syllable_train = Y_syllable_train_validation[idx_train]
    Y_syllable_val = Y_syllable_train_validation[idx_val]

    Y_phoneme_train = Y_phoneme_train_validation[idx_train]
    Y_phoneme_val = Y_phoneme_train_validation[idx_val]

    sample_weights_syllable_train = sample_weights_syllable[idx_train]
    sample_weights_syllable_val = sample_weights_syllable[idx_val]

    sample_weights_phoneme_train = sample_weights_phoneme[idx_train]
    sample_weights_phoneme_val = sample_weights_phoneme[idx_val]

    #
    return idx_train, Y_syllable_train, Y_phoneme_train, sample_weights_syllable_train, sample_weights_phoneme_train, \
            idx_val, Y_syllable_val, Y_phoneme_val, sample_weights_syllable_val, sample_weights_phoneme_val, \
            indices_all, Y_syllable_train_validation, Y_phoneme_train_validation