"""
This is not working, so don' waste time to look at it :)

If you really want to know what was going on here, ok...

I built something which I called the phoneme detection with the consideration of the missing phones,
you know, I wanted to consider the situation that the student were missing some phones when singing,
I used a rule-based method to generate multi decoding paths, where each path has some missing phones,
then I choose one path with the maximum posterior probabilities after the viterbi decoding.

The first result showed that it improved the phoneme overlapping metric, then it turned out just an error.

So nothing improved...
"""
from general.trainTestSeparation import get_train_test_recordings_joint
from general.filePathJoint import *
from general.parameters import varin
from proposed_method_pipeline import onset_function_all_recordings
from onsetSegmentEval.runEval import run_eval_onset
from onsetSegmentEval.runEval import run_eval_segment

import pickle
import random
from keras.models import load_model

def randomSearch(bounds,
                 rounds,
                 primarySchool_val_recordings,
                 scaler,
                 model_keras_cnn_0):
    """
    Random search the penalizing parameters alpha and beta
    :param bounds:
    :param rounds:
    :param primarySchool_val_recordings:
    :param scaler:
    :param model_keras_cnn_0:
    :return:
    """

    max_acc_segment = -float('inf')
    history_acc_segment = []
    best_hp_acc_segment = None

    max_f1_onset = -float('inf')
    history_f1_onset = []
    best_hp_f1_onset = None

    # D = 2

    for ii in range(rounds):

        # generate random parameters
        # params = [bounds[d][0] + random.random() * (bounds[d][1] - bounds[d][0]) for d in range(D)]
        # varin['penalized_alpha'] = params[0]
        # varin['penalized_beta'] = params[1]
        #
        # print('alpha', varin['penalized_alpha'])
        # print('beta', varin['penalized_beta'])

        varin['posterior_norm'] = ii * 0.1

        print('round', ii)
        print('posterior_norm', varin['posterior_norm'])

        onset_function_all_recordings(wav_path=primarySchool_wav_path,
                                      textgrid_path=primarySchool_textgrid_path,
                                      scaler=scaler,
                                      test_recordings=primarySchool_val_recordings,
                                      model_keras_cnn_0=model_keras_cnn_0,
                                      cnnModel_name=cnnModel_name,
                                      eval_results_path=eval_results_path,
                                      obs_cal='toload',
                                      plot=False,
                                      missing_phn=True)

        nolabel_phoneme_25, label_phoneme_25 = run_eval_onset('joint', '', 'val')
        _, acc_phoneme_segment = run_eval_segment('joint', '', 'val')

        history_acc_segment.append(acc_phoneme_segment)
        history_f1_onset.append(nolabel_phoneme_25)

        if acc_phoneme_segment > max_acc_segment:
            max_acc_segment = acc_phoneme_segment
            # best_hp_acc_segment = [varin['penalized_alpha'], varin['penalized_beta']]
            best_hp_acc_segment = varin['posterior_norm']

        if nolabel_phoneme_25 > max_f1_onset:
            max_f1_onset = nolabel_phoneme_25
            best_hp_f1_onset = varin['posterior_norm']

        print('accuracy segment', acc_phoneme_segment)
        print('f1 onset', nolabel_phoneme_25)
        print('\n')

    return max_acc_segment, best_hp_acc_segment, history_acc_segment, \
           max_f1_onset, best_hp_f1_onset, history_f1_onset

if __name__ == '__main__':

    primarySchool_val_recordings, _, _, _, _, _ = get_train_test_recordings_joint()
    scaler = pickle.load(open(scaler_joint_model_path, 'rb'))
    model_keras_cnn_0 = load_model(full_path_keras_cnn_0 + str(0) + '.h5')

    bounds = [[0.0, 100.0], [0.0, 100.0]]

    max_acc_segment, best_hp_acc_segment, history_acc_segment, max_f1_onset, best_hp_f1_onset, history_f1_onset =\
        randomSearch(bounds, 21, primarySchool_val_recordings, scaler, model_keras_cnn_0)

    print('best accuracy segment', max_acc_segment)
    print('best posterior norm segment', best_hp_acc_segment)
    print('accuracy history segment', history_acc_segment)

    print('best accuracy onset', max_f1_onset)
    print('best posterior norm onset', best_hp_f1_onset)
    print('accuracy history onset', history_f1_onset)
    # print('best alpha', best_hp[0])
    # print('best beta', best_hp[1])

    """
    ('best accuracy segment', 57.19141603365809)
    ('best posterior norm segment', 1.3)
    ('accuracy history segment', [55.98989928069389, 55.90353319542177, 55.96111058560318, 55.90353319542177, 55.90353319542177, 55.95391341183051, 55.91381487223989, 55.97283169717583, 56.07544283182055, 56.14391879942916, 56.52434084169919, 56.404456490000044, 56.81305032675169, 57.19141603365809, 57.10648938314051, 57.175993518430936, 57.08736546425883, 57.00511204971397, 56.76842784936109, 56.813667227360774, 56.73634901768859])
    ('best accuracy onset', 77.054814469621959)
    ('best posterior norm onset', 1.8)
    ('accuracy history onset', [73.136222320918733, 73.07985273581312, 73.052030456852805, 73.07985273581312, 73.108043788672063, 73.164388340892515, 73.240775960441226, 73.309067890489004, 73.796319637005297, 74.543575611436452, 75.309025405304666, 75.859095947776822, 76.336017569546115, 76.627335003168668, 76.70987396212891, 76.797957911608449, 76.920358152686148, 76.894227951948423, 77.054814469621959, 77.040905134899916, 76.944919305768835])
    swig/python detected a memory leak of type 'int64_t *', no destructor found.
    """