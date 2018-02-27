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

fs = 44100
framesize_t = 0.025  # in second
hopsize_t = 0.010

framesize = int(round(framesize_t * fs))
hopsize = int(round(hopsize_t * fs))

highFrequencyBound = fs/2 if fs/2<11000 else 11000

# acoustic model
am = 'cnn'

varin = {}
# parameters of viterbi
varin['delta_mode'] = 'proportion'
varin['delta']      = 0.35

# parameters of the penalized viterbi
# varin['penalized_alpha'] = 1.0
# varin['penalized_beta'] = 1.0

# parameters of posterior normalization
varin['posterior_norm'] = 1.0