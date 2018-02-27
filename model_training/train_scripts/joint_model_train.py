import sys, os
import cPickle
import gzip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models_joint import train_model_validation


if __name__ == '__main__':

    nlen = 15
    input_dim = (80, nlen)

    # change these folders
    filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/syllableSeg/feature_all_joint.h5'
    filename_labels_syllable_train_validation_set = '../../training_data/labels_joint_syllable.pickle.gz'
    filename_labels_phoneme_train_validation_set = '../../training_data/labels_joint_phoneme.pickle.gz'
    filename_sample_weights_syllable = '../../training_data/sample_weights_joint_syllable.pickle.gz'
    filename_sample_weights_phoneme = '../../training_data/sample_weights_joint_phoneme.pickle.gz'

    labels_syllable = cPickle.load(gzip.open(filename_labels_syllable_train_validation_set, 'rb'))
    labels_pho = cPickle.load(gzip.open(filename_labels_phoneme_train_validation_set, 'rb'))

    for running_time in range(1, 5):
        # change these folders
        file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/jan_joint_subset_syllable_val_loss_weighted'+str(running_time)+'.h5'
        file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/jan_joint_subset_syllable_val_loss_weighted'+str(running_time)+'.csv'

        train_model_validation(filename_train_validation_set,
                               filename_labels_syllable_train_validation_set,
                               filename_labels_phoneme_train_validation_set,
                               filename_sample_weights_syllable,
                               filename_sample_weights_phoneme,
                               filter_density=1,
                               dropout=0.5,
                               input_shape=input_dim,
                               file_path_model=file_path_model,
                               filename_log=file_path_log,
                               channel=1)

