import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
from models import train_model_validation

if __name__ == '__main__':

    nlen = 15
    input_dim = (80, nlen)

    # change these two paths
    filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/acousticModels/feature_hsmm_am.h5'
    filename_labels_train_validation_set = '/Users/gong/Documents/MTG document/dataset/acousticModels/labels_hsmm_am.pickle.gz'

    for ii in range(1, 5):
        # change these two paths
        file_path_model = '/homedtic/rgong/acousticModelsTraining/out/hsmm_am_timbral_'+str(ii)+'.h5'
        file_path_log = '/homedtic/rgong/acousticModelsTraining/out/log/hsmm_am_timbral_'+str(ii)+'.csv'

        train_model_validation(filename_train_validation_set,
                               filename_labels_train_validation_set,
                               filter_density=4,
                               dropout=0.32,
                               input_shape=input_dim,
                               file_path_model = file_path_model,
                               filename_log = file_path_log,
                               channel=1)
