"""run evaluation on the onset detection results, output results in the eval directory"""

from filePathHsmm import eval_results_path as eval_results_path_hsmm
from filePathJoint import eval_results_path as eval_results_path_joint
from filePathHsmm import cnn_file_name
from filePathJoint import cnnModel_name
from general.trainTestSeparation import get_train_test_recordings_joint
from general.utilFunctions import get_recordings
from evaluation import onsetEval
from evaluation import segmentEval
from evaluation import metrics
from parameters import *
import os
import pickle
import numpy as np


def write_results_2_txt(filename,
                        decoding_method,
                        results):
    """write the evaluation results to a text file"""

    with open(filename, 'w') as f:

        # no label 0.025
        f.write(str(results[0]))
        f.write('\n')
        f.write(str(results[1]))
        f.write('\n')
        f.write(str(results[2]))
        f.write('\n')

        # no label 0.05
        f.write(str(results[3]))
        f.write('\n')
        f.write(str(results[4]))
        f.write('\n')
        f.write(str(results[5]))
        f.write('\n')

        # label 0.025
        f.write(str(results[6]))
        f.write('\n')
        f.write(str(results[7]))
        f.write('\n')
        f.write(str(results[8]))
        f.write('\n')

        # label 0.05
        f.write(str(results[9]))
        f.write('\n')
        f.write(str(results[10]))
        f.write('\n')
        f.write(str(results[11]))


def batch_run_metrics_calculation(sumStat, gt_onsets, detected_onsets):
    """
    Batch run the metric calculation
    :param sumStat:
    :param gt_onsets:
    :param detected_onsets:
    :return:
    """
    counter = 0
    for l in [False, True]:
        for t in [0.025, 0.05]:
            numDetectedOnsets, numGroundtruthOnsets, \
            numOnsetCorrect, _, _ = onsetEval(gt_onsets, detected_onsets, t, l)

            sumStat[counter, 0] += numDetectedOnsets
            sumStat[counter, 1] += numGroundtruthOnsets
            sumStat[counter, 2] += numOnsetCorrect

            counter += 1


def metrics_aggregation(sumStat):

    recall_nolabel_25, precision_nolabel_25, F1_nolabel_25 = metrics(sumStat[0, 0], sumStat[0, 1], sumStat[0, 2])
    recall_nolabel_5, precision_nolabel_5, F1_nolabel_5 = metrics(sumStat[1, 0], sumStat[1, 1], sumStat[1, 2])
    recall_label_25, precision_label_25, F1_label_25 = metrics(sumStat[2, 0], sumStat[2, 1], sumStat[2, 2])
    recall_label_5, precision_label_5, F1_label_5 = metrics(sumStat[3, 0], sumStat[3, 1], sumStat[3, 2])

    return precision_nolabel_25, recall_nolabel_25, F1_nolabel_25, \
           precision_nolabel_5, recall_nolabel_5, F1_nolabel_5, \
           precision_label_25, recall_label_25, F1_label_25, \
           precision_label_5, recall_label_5, F1_label_5


def run_eval_onset(method='hsmm', param_str='', test_val='test'):
    """
    run evaluation for onset detection
    :param method: hsmm or joint:
    :param param_str different configurations and save the results to different txt files
    :param test_val: string, val or test evaluate for validation or test file
    :return:
    """
    if method == 'hsmm':
        eval_results_path = eval_results_path_hsmm+param_str
        eval_filename = cnn_file_name
    else:
        eval_results_path = eval_results_path_joint+param_str
        eval_filename = cnnModel_name

    primarySchool_val_recordings, primarySchool_test_recordings, _, _, _, _ = get_train_test_recordings_joint()

    if test_val == 'test':
        recordings = primarySchool_test_recordings
    elif test_val == 'val':
        recordings = primarySchool_val_recordings
    else:
        recordings = primarySchool_test_recordings + primarySchool_val_recordings

    sumStat_syllable = np.zeros((4, 3), dtype='int')
    sumStat_phoneme = np.zeros((4, 3), dtype='int')

    for artist, rn in recordings:
        results_path = os.path.join(eval_results_path, artist)
        result_files = get_recordings(results_path)

        for rf in result_files:
            result_filename = os.path.join(results_path, rf+'.pkl')
            syllable_gt_onsets, syllable_detected_onsets, \
            phoneme_gt_onsets, phoneme_detected_onsets, _ = pickle.load(open(result_filename, 'r'))

            batch_run_metrics_calculation(sumStat_syllable, syllable_gt_onsets, syllable_detected_onsets)
            batch_run_metrics_calculation(sumStat_phoneme, phoneme_gt_onsets, phoneme_detected_onsets)

    result_syllable = metrics_aggregation(sumStat_syllable)
    result_phoneme = metrics_aggregation(sumStat_phoneme)

    if test_val != 'val':
        current_path = os.path.dirname(os.path.abspath(__file__))

        write_results_2_txt(os.path.join(current_path, '../' + method, eval_filename
                                         + '_syllable_onset' + '_' + test_val + param_str + '.txt'),
                            method,
                            result_syllable)

        write_results_2_txt(os.path.join(current_path, '../' + method, eval_filename
                                         + '_phoneme_onset' + '_' + test_val + param_str + '.txt'),
                            method,
                            result_phoneme)

    return result_phoneme[2], result_phoneme[8]


def segment_eval_helper(onsets, line_time):
    onsets_frame = np.round(np.array([sgo[0] for sgo in onsets]) / hopsize_t)

    resample = [onsets[0][1]]

    current = onsets[0][1]

    for ii_sample in range(1, int(round(line_time / hopsize_t))):

        if ii_sample in onsets_frame:
            idx_onset = np.where(onsets_frame == ii_sample)
            idx_onset = idx_onset[0][0]
            current = onsets[idx_onset][1]
        resample.append(current)

    return resample


def run_eval_segment(method='hsmm', param_str='', test_val='test'):
    """segment level evaluation"""
    if method == 'hsmm':
        eval_results_path = eval_results_path_hsmm+param_str
        eval_filename = cnn_file_name
    else:
        eval_results_path = eval_results_path_joint+param_str
        eval_filename = cnnModel_name

    primarySchool_val_recordings, primarySchool_test_recordings, _, _, _, _ = get_train_test_recordings_joint()

    if test_val == 'val':
        recordings = primarySchool_val_recordings
    elif test_val == 'test':
        recordings = primarySchool_test_recordings
    else:
        recordings = primarySchool_test_recordings + primarySchool_val_recordings

    sumSampleCorrect_syllable, sumSampleCorrect_phoneme, \
    sumSample_syllable, sumSample_phoneme = 0,0,0,0

    for artist, rn in recordings:
        results_path = os.path.join(eval_results_path, artist)
        result_files = get_recordings(results_path)

        for rf in result_files:
            result_filename = os.path.join(results_path, rf+'.pkl')
            syllable_gt_onsets, syllable_detected_onsets, \
            phoneme_gt_onsets, phoneme_detected_onsets, line_time \
                                        = pickle.load(open(result_filename, 'r'))

            syllable_gt_onsets_resample = segment_eval_helper(syllable_gt_onsets, line_time)
            syllable_detected_onsets_resample = segment_eval_helper(syllable_detected_onsets, line_time)
            phoneme_gt_onsets_resample = segment_eval_helper(phoneme_gt_onsets, line_time)
            phoneme_detected_onsets_resample = segment_eval_helper(phoneme_detected_onsets, line_time)

            sample_correct_syllable, sample_syllable = \
                segmentEval(syllable_gt_onsets_resample, syllable_detected_onsets_resample)
            sample_correct_phoneme, sample_phoneme = \
                segmentEval(phoneme_gt_onsets_resample, phoneme_detected_onsets_resample)

            sumSampleCorrect_syllable += sample_correct_syllable
            sumSampleCorrect_phoneme += sample_correct_phoneme
            sumSample_syllable += sample_syllable
            sumSample_phoneme += sample_phoneme

    acc_syllable = sumSampleCorrect_syllable/float(sumSample_syllable)
    acc_phoneme = sumSampleCorrect_phoneme/float(sumSample_phoneme)

    acc_syllable *= 100
    acc_phoneme *= 100

    if test_val != 'val':
        # write the results to txt only in test mode
        current_path = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(current_path, '../'+method, eval_filename
                +'_syllable_segment'+'_'+test_val+param_str+'.txt'), 'w') as f:
            f.write(str(acc_syllable))

        with open(os.path.join(current_path, '../'+method, eval_filename
                +'_phoneme_segment'+'_'+test_val+param_str+'.txt'), 'w') as f:
            f.write(str(acc_phoneme))

    return acc_syllable, acc_phoneme

if __name__ == '__main__':
    run_eval_onset('hsmm')
    run_eval_onset('joint')
    run_eval_segment('hsmm')
    run_eval_segment('joint')