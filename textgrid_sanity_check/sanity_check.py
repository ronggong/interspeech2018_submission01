"""
check the annotation errors in textgrid
"""

from general.trainTestSeparation import get_train_test_recordings_joint
from general.textgridParser import syllableTextgridExtraction
from general.filePathJoint import *

def s_check(textgrid_path,
            recordings,
            parentTierName,
            childTierName):

    num_lines = 0
    num_units = 0
    for artist_path, recording in recordings:
        nestedLists, _, _   \
            = syllableTextgridExtraction(textgrid_path=textgrid_path,
                                         recording=join(artist_path,recording),
                                         tier0=parentTierName,
                                         tier1=childTierName)

        for ii, line_list in enumerate(nestedLists):
            # last one is the num of syllables or phonemes
            print(artist_path, recording ,ii, len(line_list[1]))
            num_lines += 1
            num_units += len(line_list[1])

    return num_lines, num_units

if __name__ == '__main__':
    # check line contains a reasonable syllable or phoneme number
    valPrimarySchool, testPrimarySchool, trainNacta2017, trainNacta, trainPrimarySchool, trainSepa \
        = get_train_test_recordings_joint()


    num_lines, num_syllables = s_check(textgrid_path=primarySchool_textgrid_path,
                                        parentTierName='line',
                                        childTierName='dianSilence',
                                        recordings=valPrimarySchool)

    num_lines, num_phns = s_check(textgrid_path=primarySchool_textgrid_path,
                                    parentTierName='line',
                                    childTierName='details',
                                    recordings=valPrimarySchool)

    print(num_lines, num_syllables)
    print(num_lines, num_phns)



