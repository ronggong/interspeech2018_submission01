import os
from textgridParser import textGrid2WordList
from textgridParser import wordListsParseByLines

def get_recordings(a_path):
    """
    retrieve the filename from a path
    :param a_path:
    :return:
    """
    recordings = []
    for root, subFolders, files in os.walk(a_path):
        for f in files:
            file_prefix, file_extension = os.path.splitext(f)
            if file_prefix != '.DS_Store' and file_prefix != '_DS_Store':
                recordings.append(file_prefix)

    return recordings


def remove_silence(phn_onsets, phn_labels):
    """
    Remove silence phoneme onsets and labels
    :param phn_onsets:
    :param phn_labels:
    :return:
    """
    phn_onsets = list(phn_onsets)
    phn_labels = list(phn_labels)

    for ii in reversed(range(len(phn_labels))):
        if phn_labels[ii] == u'':
            phn_onsets.pop(ii)
            phn_labels.pop(ii)
    return phn_onsets, phn_labels


def textgrid_syllable_phoneme_parser(textgrid_file, tier1, tier2):
    """
    Parse the textgrid file,
    :param textgrid_file: filename
    :param tier1: syllable tier
    :param tier2: phoneme tier
    :return: syllable and phoneme lists
    """
    line_list = textGrid2WordList(textgrid_file, whichTier='line')
    syllable_list = textGrid2WordList(textgrid_file, whichTier=tier1)
    phoneme_list = textGrid2WordList(textgrid_file, whichTier=tier2)

    # parse lines of groundtruth
    nested_syllable_lists, _, _ = wordListsParseByLines(line_list, syllable_list)
    nested_phoneme_lists, _, _ = wordListsParseByLines(line_list, phoneme_list)

    return nested_syllable_lists, nested_phoneme_lists