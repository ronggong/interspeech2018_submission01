"""use this script to calculate the mean evaluation statistics for 5 run times"""

import os
import csv
import numpy as np


def parseCsv(filename):
    list_out = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            list_out.append(float(row[0]))
    return list_out


def aggregate_run_times_eval_results(eval_str='joint', phn_syl='phoneme', onset_seg='onset', joint_obs=False):
    """aggragate the results of 5 run times, write the mean and std to csv"""

    root_path = os.path.join(os.path.dirname(__file__), '..')

    if eval_str == 'hsmm':
        eval_results_path = os.path.join(root_path, 'hsmm', 'hsmm_am_timbral')
    elif eval_str == 'joint':
        eval_results_path = os.path.join(root_path, 'joint', 'jan_joint')
    else:
        raise ValueError(eval_str, 'is not a valid eval string.')

    joint_obs_str = '_joint_obs_e6' if joint_obs else ''

    for val_test in ['test, all']:

        array_run_time = []

        for ii in range(5):
            fn_results = os.path.join(
                eval_results_path+'_'+phn_syl+'_'+onset_seg+'_'+val_test+joint_obs_str+'_'+str(ii)+'.txt')

            list_run_time = parseCsv(fn_results)

            array_run_time.append(list_run_time)

        mat_run_time = np.array(array_run_time)

        mean_run_time = np.mean(mat_run_time, axis=0)
        std_run_time = np.std(mat_run_time, axis=0)

        fn_run_time_out = os.path.join(eval_results_path+'_'+phn_syl+'_'+onset_seg+'_' + val_test + joint_obs_str + '.txt')

        with open(fn_run_time_out, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for ii in range(len(mean_run_time)):
                writer.writerow([mean_run_time[ii], std_run_time[ii]])


if __name__ == '__main__':
    joint_obs = False
    eval_str = 'joint'
    aggregate_run_times_eval_results(eval_str=eval_str, phn_syl='phoneme', onset_seg='onset', joint_obs=joint_obs)
    aggregate_run_times_eval_results(eval_str=eval_str, phn_syl='syllable', onset_seg='onset', joint_obs=joint_obs)
    aggregate_run_times_eval_results(eval_str=eval_str, phn_syl='phoneme', onset_seg='segment', joint_obs=joint_obs)
    aggregate_run_times_eval_results(eval_str=eval_str, phn_syl='syllable', onset_seg='segment', joint_obs=joint_obs)
