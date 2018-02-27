'''
 * Copyright (C) 2017  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuSingingPhraseMatching
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 *
 * If you want to refer this code, please use this article:
 *
'''

import numpy as np

TRANS_PROB_SELF = 0.0                       # transition probability for HSMM is 0
TRANS_PROB_NEXT = 1.0-TRANS_PROB_SELF


def singleTransMatBuild(state_phns):
    """
    Construct the topology of the HSMM
    :param state_phns:
    :return:
    """
    # transition matrix
    num_state = len(state_phns)
    mat_trans = np.zeros((num_state, num_state))
    for ii in range(num_state - 1):
        mat_trans[ii][ii] = TRANS_PROB_SELF
        mat_trans[ii][ii + 1] = TRANS_PROB_NEXT
    mat_trans[-1][-1] = TRANS_PROB_SELF
    return mat_trans

