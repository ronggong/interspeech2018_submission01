import os
import pickle
from general.parameters import *
from zhon.hanzi import punctuation as puncChinese
import re


def removePunctuation(char):
    if len(re.findall(ur'[\u4e00-\u9fff]+', char)):
        char = re.sub(ur"[%s]+" % puncChinese, "", char)
    return char


def findShiftOffset(gtSyllableLists, scoreSyllableLists, ii_line):
    """
    find the shifting Offset
    :param gtSyllableLists:
    :param scoreSyllableLists:
    :param ii_line:
    :return:
    """
    ii_aug = 0
    text_gt = removePunctuation(gtSyllableLists[ii_line][0][2].rstrip())
    text_score = removePunctuation(scoreSyllableLists[ii_line + ii_aug][0][2].rstrip())

    while text_gt != text_score:
        ii_aug += 1
        text_score = removePunctuation(scoreSyllableLists[ii_line + ii_aug][0][2].rstrip())
    return ii_aug

def gt_score_preparation_helper(gtSyllableLists,
                                scoreSyllableLists,
                                gtPhonemeLists,
                                scorePhonemeLists,
                                ii_line,
                                ii_aug=0):
    """
    Prepare the onset times labels for syllable or phoneme ground truth or score
    :param gtSyllableLists:
    :param scoreSyllableLists:
    :param gtPhonemeLists:
    :param scorePhonemeLists:
    :param ii_line:
    :param ii_aug;
    :return:
    """

    ###--- groundtruth, score preparation
    lineList_gt_syllable = gtSyllableLists[ii_line]
    lineList_score_syllable = scoreSyllableLists[ii_line+ii_aug]
    lineList_gt_phoneme = gtPhonemeLists[ii_line]
    lineList_score_phoneme = scorePhonemeLists[ii_line+ii_aug]

    time_start = lineList_gt_syllable[0][0]
    time_end = lineList_gt_syllable[0][1]

    frame_start = int(round(time_start / hopsize_t))
    frame_end = int(round(time_end / hopsize_t))

    # list has syllable and phoneme information
    syllable_list_gt = lineList_gt_syllable[1]
    phoneme_list_gt = lineList_gt_phoneme[1]

    syllable_list_score = lineList_score_syllable[1]
    phoneme_list_score = lineList_score_phoneme[1]

    # list only has onsets
    syllable_gt_onsets = [s[0] for s in syllable_list_gt]
    syllable_gt_labels = [s[2] for s in syllable_list_gt]

    phoneme_gt_onsets = [p[0] for p in phoneme_list_gt]
    phoneme_gt_labels = [p[2] for p in phoneme_list_gt]

    syllable_score_onsets = [s[0] for s in syllable_list_score]
    phoneme_score_onsets = [p[0] for p in phoneme_list_score]
    phoneme_score_labels = [p[2] for p in phoneme_list_score]

    # syllable score durations and labels
    syllable_score_durs = [sls[1] - sls[0] for sls in syllable_list_score]
    syllable_score_labels = [sls[2] for sls in syllable_list_score]

    return frame_start, frame_end, \
           time_start, time_end, \
           syllable_gt_onsets, syllable_gt_labels, \
           phoneme_gt_onsets, phoneme_gt_labels, \
           syllable_score_onsets, syllable_score_labels, \
           phoneme_score_onsets, phoneme_score_labels, \
           syllable_score_durs, phoneme_list_score


def results_aggregation_save_helper(syllable_gt_onsets_0start,
                                    syllable_gt_labels,
                                    boundaries_syllable_start_time,
                                    syllable_score_labels,
                                    phoneme_gt_onsets_0start,
                                    phoneme_gt_labels,
                                    boundaries_phoneme_start_time,
                                    phoneme_score_labels,
                                    eval_results_path,
                                    artist_path,
                                    rn,
                                    ii_line,
                                    line_time):
    """
    Aggregate the ground truth and detected results into a list, and dump
    :param syllable_gt_onsets_0start:
    :param syllable_gt_labels:
    :param boundaries_syllable_start_time:
    :param syllable_score_labels:
    :param phoneme_gt_onsets_0start:
    :param phoneme_gt_labels:
    :param boundaries_phoneme_start_time:
    :param phoneme_score_labels:
    :param eval_results_path:
    :param artist_path:
    :param rn:
    :param ii_line:
    :param line_time:
    :return:
    """

    # aggregate the results
    syllable_gt_onsets_to_save = [[syllable_gt_onsets_0start[ii_sgo], syllable_gt_labels[ii_sgo]]
                                  for ii_sgo in range(len(syllable_gt_onsets_0start))]
    syllable_detected_onsets_to_save = [[boundaries_syllable_start_time[ii_bsst], syllable_score_labels[ii_bsst]]
                                        for ii_bsst in range(len(boundaries_syllable_start_time))]

    phoneme_gt_onsets_to_save = [[phoneme_gt_onsets_0start[ii_pgo], phoneme_gt_labels[ii_pgo]]
                                 for ii_pgo in range(len(phoneme_gt_onsets_0start))]
    phoneme_detected_onsets_to_save = [[boundaries_phoneme_start_time[ii_bpst], phoneme_score_labels[ii_bpst]]
                                       for ii_bpst in range(len(boundaries_phoneme_start_time))]

    gt_detected_to_save = [syllable_gt_onsets_to_save, syllable_detected_onsets_to_save,
                           phoneme_gt_onsets_to_save, phoneme_detected_onsets_to_save, line_time]

    # save to pickle
    path_gt_detected_to_save = os.path.join(eval_results_path, artist_path)
    filename_gt_detected_to_save = rn + '_' + str(ii_line) + '.pkl'

    if not os.path.exists(path_gt_detected_to_save):
        os.makedirs(path_gt_detected_to_save)

    pickle.dump(gt_detected_to_save, open(os.path.join(path_gt_detected_to_save, filename_gt_detected_to_save), 'w'))


def score_variations_semivowel_helper(phn_labels, phn_durs, semivowel):
    """
    generate score variations when semivowel appears
    example: ['c', 'j', 'aI^'] will become ['c', 'aI^'], because 'j' is a semivowel
    the duration array [1, 2, 3] will become [1, 5] as we add the dur of 'j' to the next phn
    :param phn_labels:
    :param phn_durs:
    :param semivowel:
    :return:
    """
    phn_labels_var = phn_labels[:] # copy the list
    phn_durs_var = phn_durs[:]

    if semivowel in phn_labels_var:
        idx = phn_labels_var.index(semivowel)
        if idx != 0 and idx != len(phn_labels_var)-1:
            phn_labels_var.pop(idx)

            phn_durs_var[idx+1] += phn_durs_var[idx]
            phn_durs_var.pop(idx)

    return phn_labels_var, phn_durs_var


def score_variations_termination_helper(phn_labels, phn_durs, termination):
    """
    generate score variations when termination appears
    example: ['c', 'AN', 'N'] will become ['c', 'AN'], because 'N' is a semivowel
    the duration array [1, 2, 3] will become [1, 5] as we add the dur of 'N' to the previous phn
    :param phn_labels:
    :param phn_durs:
    :param termination:
    :return:
    """
    phn_labels_var = phn_labels[:]  # copy the list
    phn_durs_var = phn_durs[:]

    if termination in phn_labels_var:
        idx = phn_labels_var.index(termination)
        if idx == len(phn_labels_var)-1:
            phn_labels_var.pop(idx)

            phn_durs_var[idx-1] += phn_durs_var[idx]
            phn_durs_var.pop(idx)

    return phn_labels_var, phn_durs_var


def score_variations_phn(phn_labels, phn_durs):

    phn_durs = list(phn_durs)

    # deal with semivowels
    phn_labels_vars = [phn_labels]
    phn_durs_vars = [phn_durs]

    for semi_vowel in [u'j', u'H', u'w']:
        if semi_vowel in phn_labels and phn_labels.index(semi_vowel) != 0 and phn_labels.index(semi_vowel) != len(phn_labels)-1:
            phn_labels_var_semi_vowel, phn_durs_var_semi_vowel = score_variations_semivowel_helper(phn_labels, phn_durs, semi_vowel)
            phn_labels_vars.append(phn_labels_var_semi_vowel)
            phn_durs_vars.append(phn_durs_var_semi_vowel)

    # ok, after the above, we need to end up with something in labels or durs
    phn_labels_vars_copy = phn_labels_vars[:]
    phn_durs_vars_copy = phn_durs_vars[:]

    for ii in range(len(phn_labels_vars_copy)):
        for termi in [u'i', u'u', u'n', u'N']:
            if termi in phn_labels_vars_copy[ii] and phn_labels_vars_copy[ii].index(termi) == len(phn_labels_vars_copy[ii]) - 1 and len(phn_labels_vars_copy[ii][-2]) > 1:
                phn_labels_var_termi, phn_durs_var_termi = score_variations_termination_helper(phn_labels_vars_copy[ii], phn_durs_vars_copy[ii], termi)
                phn_labels_vars.append(phn_labels_var_termi)
                phn_durs_vars.append(phn_durs_var_termi)

    return phn_labels_vars, phn_durs_vars


if __name__ == '__main__':

    # test

    # with semivowel and termination
    phn_labels = ['c', 'j', 'En', 'n']
    phn_durs = [1.0, 2.0, 3.0, 4.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # with semivowel and termination
    phn_labels = ['c', 'H', 'En', 'N']
    phn_durs = [1.0, 2.0, 3.0, 4.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # with semivowel and termination
    phn_labels = ['c', 'H', 'En']
    phn_durs = [1.0, 2.0, 3.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # with semivowel
    phn_labels = ['c', 'j', 'En']
    phn_durs = [1.0, 2.0, 3.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # with semivowel
    phn_labels = ['c', 'H', 'En']
    phn_durs = [1.0, 2.0, 3.0, 4.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # with termination
    phn_labels = ['c', 'AN', 'N']
    phn_durs = [1.0, 2.0, 3.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # with termination
    phn_labels = ['c', 'En', 'n']
    phn_durs = [1.0, 2.0, 3.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # semivowel in the beginning
    phn_labels = ['j', 'En']
    phn_durs = [1.0, 2.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # termination in the beginning
    phn_labels = ['N', 'O']
    phn_durs = [1.0, 2.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # no semivowel and termination
    phn_labels = ['c', 'En']
    phn_durs = [1.0, 2.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # termination
    phn_labels = ['c', 'AU', 'u']
    phn_durs = [1.0, 2.0, 3.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # termination
    phn_labels = ['c', 'u']
    phn_durs = [1.0, 2.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # termination
    phn_labels = ['c', 'u', 'u']
    phn_durs = [1.0, 2.0, 3.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)

    # termination
    phn_labels = ['c', 'i']
    phn_durs = [1.0, 2.0]
    phn_labels_vars, phn_durs_vars = score_variations_phn(phn_labels, phn_durs)
    print(phn_labels_vars)
    print(phn_durs_vars)