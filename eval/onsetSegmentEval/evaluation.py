# -*- coding: utf-8 -*-

"""
Syllable segmentation evaluation: landmark and boundary evaluations
Only evaluate boundary onset

[1] A new hybrid approach for automatic speech signal segmentation
using silence signal detection, energy convex hull, and spectral variation

[2] Syll-O-Matic: An adaptive time-frequency representation
for the automatic segmentation of speech into syllables

[3] EVALUATION FRAMEWORK FOR AUTOMATIC SINGING
TRANSCRIPTION
"""

from general.phonemeMap import misMatchIgnorePhn
from general.phonemeMap import misMatchIgnoreSyl


def onsetEval(groundtruthOnsets, detectedOnsets, tolerance, label):
    """
    :param groundtruthOnsets: [[onset time, onset label], ...]
    :param detectedOnsets: [[onset time, onset label], ...]
    :param tolerance: 0.025 or 0.05
    :param label: True or False, if we want to evaluate the label
    :return:
    """

    numDetectedOnsets       = len(detectedOnsets)
    numGroundtruthOnsets    = len(groundtruthOnsets)

    onsetCorrectlist            = [0]*numDetectedOnsets

    for gtb in groundtruthOnsets:
        for idx, db in enumerate(detectedOnsets):
            onsetTh     = tolerance                                          # onset threshold

            if abs(db[0]-gtb[0])<onsetTh:
                if label:
                    if db[1] == gtb[1]:
                        onsetCorrectlist[idx] = 1
                else:
                    onsetCorrectlist[idx] = 1


    numOnsetCorrect     = sum(onsetCorrectlist)
    numInsertion        = numDetectedOnsets - numOnsetCorrect
    numDeletion         = numGroundtruthOnsets - numOnsetCorrect

    return numDetectedOnsets, numGroundtruthOnsets, \
           numOnsetCorrect, numInsertion, numDeletion


def metrics(numDetected, numGroundtruth, numCorrect):
    recall = (numCorrect/float(numGroundtruth))*100
    precision = (numCorrect/float(numDetected))*100
    if precision == 0 and recall == 0:
        F1 = 0
    else:
        F1 = 2*(precision*recall)/(precision+recall)

    return recall, precision, F1


def segmentEval(gt_resample, detected_resample):

    sampleCorrect = 0
    for ii in range(len(gt_resample)):
        if gt_resample[ii] == detected_resample[ii] or \
                        [gt_resample[ii], detected_resample[ii]] in misMatchIgnorePhn or \
                        [gt_resample[ii], detected_resample[ii]] in misMatchIgnoreSyl:
            sampleCorrect += 1

    return sampleCorrect, len(gt_resample)
