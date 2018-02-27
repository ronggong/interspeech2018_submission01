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
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

cimport numpy as np
cimport cython

from LRHMM import _LRHMM
# from general.phonemeMap import *
from general.parameters import *

# import os,sys
# import json


class LRHSMM(_LRHMM):

    def __init__(self, mat_tran, state_phns, phns_durs, proportionality_std):
        _LRHMM.__init__(self)

        self.A              = mat_tran
        self.transcription  = state_phns
        self.phns_durs      = phns_durs        # duration of each state
        self.proportionality_std = proportionality_std
        self.n              = len(self.transcription)
        self.tau            = None
        self._initialStateDist()

    def _initialStateDist(self):
        '''
        explicitly set the initial state distribution
        '''
        # list_forced_beginning = [u'nvc', u'vc', u'w']
        self.pi     = np.zeros((self.n), dtype=self.precision)

        # set the initial start state to 0
        self.pi[0] = 1.0


    def _inferenceInit(self, observations):
        '''
        HSMM inference initialization
        :param observations:
        :return:
        '''

        tau = len(observations)
        self.tau = tau

        # Forward quantities
        forwardDelta        = np.ones((self.n,tau),dtype=self.precision)
        forwardDelta        *= -float('inf')
        previousState       = np.zeros((self.n,tau),dtype=np.intc)
        state               = np.zeros((self.n,tau),dtype=np.intc)
        occupancy           = np.zeros((self.n,tau),dtype=np.intc)

        # State-in
        # REMARK : state-in a time t is StateIn(:,t+1), such that StateIn(:1) is
        # the initial distribution
        stateIn             = np.ones((self.n,tau),dtype=self.precision)
        stateIn             *= -float('inf')

        # Set initial states distribution \pi %%%%            % PARAMETERs

        return forwardDelta,\
               previousState,\
                state,\
               stateIn,\
                occupancy,\
                tau

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _viterbiHSMM(self,
                     forwardDelta,
                     previousState,
                     state,
                     stateIn,
                     occupancy,
                     tau,
                     obsOnsetPhn):
        """
        HSMMs Viterbi decoding, from Guedon 2007 paper
        :param observations:
        :param am:
        :return:
        """
        # tried having observation onset function in the decoding, not working,
        # tried to insert the term cObsOnsetPhn[t] in calculating prod_occupancy
        # cdef double [::1] cObsOnsetPhn      = np.log(obsOnsetPhn, dtype='float64')

        cdef double [:, ::1] cA             = np.log(self.A)
        cdef double [:, ::1] cforwardDelta  = forwardDelta
        cdef int [:, ::1] cpreviousState    = previousState
        cdef int [:, ::1] cstate            = state
        cdef double [:, ::1] cstateIn       = stateIn
        cdef int [:, ::1] coccupancy        = occupancy

        # cdef double [:, ::1] cobs   = np.log(obs)
        cdef double [::1] cpi       = np.log(self.pi)
        # cdef double cmaxForward     = -float('inf') # max value in time T (max)

        # print pi
        # print self.net.getStates()
        # print self.transcription

        # predefine M,d,D
        M = []
        max_mean_dur    = max(self.phns_durs)
        max_std_dur     = self.proportionality_std*max_mean_dur
        # x is generated as the index of the largest phoneme duration
        x               = np.arange(0,max_mean_dur+10*max_std_dur,hopsize_t)
        d = np.zeros((self.n,len(x)),dtype=self.precision)
        D = np.zeros((self.n,len(x)),dtype=self.precision)

        for j in xrange(self.n):
            mean_j          = self.phns_durs[j]
            std_j           = self.proportionality_std*mean_j
            M.append(int((mean_j+10*std_j)/hopsize_t)-1)
            d[j,:]          = norm.logpdf(x, mean_j, std_j)
            D[j,:]          = norm.logsf(x, mean_j, std_j)

        cdef int [::1] cM   = np.array(M,dtype=np.intc)
        cdef double[:,::1] cd = d
        cdef double[:,::1] cD = D

        # version in guedon 2003 paper
        for t in range(0,tau):
            # print t
            # if t > 20:
            #     raise
            for j in xrange(self.n):

                xsampa_state = self.transcription[j]

                observ          = 0.0

                if t<tau-1:
                    for u in range(1,min(t+1,cM[j])+1):
                        observ += self.B_map[xsampa_state][t-u+1]
                        if u < t+1:
                            # prod_occupancy = observ+cd[j][u]+cstateIn[j,t-u+1]+cObsOnsetPhn[t]
                            prod_occupancy = observ+cd[j][u]+cstateIn[j,t-u+1]
                            # print t, j, u, prod_occupancy, observ, cd[j][u], cstateIn[j,t-u+1]
                            if prod_occupancy > cforwardDelta[j,t]:
                                cforwardDelta[j,t]   = prod_occupancy
                                cpreviousState[j,t]  = cstate[j,t-u+1]
                                coccupancy[j,t]      = u
                        else:
                            # print u, len(occupancies_j)
                            # prod_occupancy  = observ+cd[j][t+1]+cpi[j]+cObsOnsetPhn[t]
                            prod_occupancy  = observ+cd[j][t+1]+cpi[j]
                            # print t, j, prod_occupancy, observ, d[j][u], cpi[j]
                            if prod_occupancy > cforwardDelta[j,t]:
                                cforwardDelta[j,t]   = prod_occupancy
                                coccupancy[j,t]      = t+1

                else:
                    for u in range(1,min(tau,cM[j])+1):
                        observ += self.B_map[xsampa_state][tau-u]
                        if u < tau:
                            # prod_survivor = observ+cD[j][u]+cstateIn[j,tau-u]+cObsOnsetPhn[tau-1]
                            prod_survivor = observ+cD[j][u]+cstateIn[j,tau-u]
                            # print t, j, prod_survivor, observ, cD[j][u], cstateIn[j,tau-u]
                            if prod_survivor > cforwardDelta[j,tau-1]:
                                cforwardDelta[j,tau-1]   = prod_survivor
                                cpreviousState[j,t]      = cstate[j,tau-u]
                                coccupancy[j,tau-1]      = u

                        else:
                            # prod_survivor = observ+cD[j][tau]+cpi[j]+cObsOnsetPhn[tau-1]
                            prod_survivor = observ+cD[j][tau]+cpi[j]
                            # print t, j, prod_survivor, observ, cD[j][u], cpi[j]
                            if prod_survivor > cforwardDelta[j,tau-1]:
                                cforwardDelta[j,tau-1]   = prod_survivor
                                coccupancy[j,tau-1]      = tau

            # ignore normalization

            if t<tau-1:
                for j in range(self.n):
                    for i in range(self.n):

                        if cstateIn[j,t+1] < cA[i][j] + cforwardDelta[i,t]:
                            cstateIn[j,t+1]        = cA[i][j] + cforwardDelta[i,t]
                            cstate[j,t+1]          = i


        # termination: find the maximum probability for the entire sequence (=highest prob path)

        # posteri_prob   = 0

        # print self.idx_final_tail
        # print i
        # endingProb = 0.0

        posteri_prob = cforwardDelta[self.n-1][tau-1]

        # tracking all parallel paths
        path            = np.zeros((tau),dtype=np.intc)
        path[tau-1]     = i
        t = tau-1

        while t>=0:
            j = path[t]
            u = coccupancy[j,t]
            if j == 0 and u == 0:
                # this is the case that poster_probs is -INFINITY
                # dead loop
                path[:] = j
                break
            for v in xrange(1,u):
                path[t-v] = j
            if t >= u:
                path[t-u] = cpreviousState[j,t]
            t = t-u


        # avoid memory leaks
        # cObsOnsetPhn    = None
        cA              = None
        cforwardDelta   = None
        cpreviousState  = None
        cstate          = None
        cstateIn        = None
        coccupancy      = None

        cpi             = None

        cM              = None
        cd              = None
        cD              = None

        return path, posteri_prob

    def _pathStateDur(self,path):
        '''
        path states in phoneme and duration
        :param path:
        :return:
        '''
        dur_frame = 1
        state_dur_path = []
        for ii in xrange(1,len(path)):
            if path[ii] != path[ii-1]:
                state_dur_path.append([self.transcription[int(path[ii-1])], dur_frame * hopsize_phoneticSimilarity / float(fs)])
                dur_frame = 1
            else:
                dur_frame += 1
        state_dur_path.append([self.transcription[int(path[-1])], dur_frame * hopsize_phoneticSimilarity / float(fs)])
        return state_dur_path

    def _plotNetwork(self,path):
        self.net.plotNetwork(path)

    def _getBmap(self):
        """
        from the transcription dictionary obtain the emission matrix
        :return:
        """
        B_map = np.zeros((len(self.transcription), self.tau))
        for ii_trans, trans in enumerate(self.transcription):
            B_map[ii_trans, :] = self.B_map[trans]
        return B_map

    def _pathPlot(self, transcription_gt, path_gt, path):
        '''
        plot ground truth path and decoded path
        :return:
        '''

        # ##-- unique transcription and path
        # transcription_unique = []
        # transcription_number_unique = []
        # B_map_unique = np.array([])
        # for ii,t in enumerate(self.transcription):
        #     if t not in transcription_unique:
        #         transcription_unique.append(t)
        #         transcription_number_unique.append(ii)
        #         if not len(B_map_unique):
        #             B_map_unique = self.B_map[t]
        #         else:
        #             B_map_unique = np.vstack((B_map_unique,self.B_map[t]))
        #
        # trans2transUniqueMapping = {}
        # for ii in range(len(self.transcription)):
        #     trans2transUniqueMapping[ii] = transcription_unique.index(self.transcription[ii])
        #
        # path_unique = []
        # for ii in range(len(path)):
        #     path_unique.append(trans2transUniqueMapping[path[ii]])
        #

        # ##-- figure plot
        plt.figure()
        # n_states = B_map_unique.shape[0]
        # n_frame  = B_map_unique.shape[1]
        B_map = self._getBmap()

        n_states = B_map.shape[0]
        n_frame = B_map.shape[1]
        y = np.arange(n_states+1)
        x = np.arange(n_frame) * hopsize / float(fs)

        # plt.pcolormesh(x,y,B_map_unique)
        # plt.plot(x,path_unique,'b',linewidth=3)
        # plt.xlabel('time (s)')
        # plt.ylabel('states')
        # plt.yticks(y, transcription_unique, rotation='horizontal')
        # plt.show()
        plt.pcolormesh(x, y, B_map)
        plt.plot(x, path, 'b', linewidth=3)
        # for onset in path_gt:
        #     plt.axvline(onset, color='r', linewidth=2)
        plt.plot(x, path_gt, 'r', linewidth=2)
        plt.xlabel('time (s)')
        plt.ylabel('states')
        plt.yticks(y, self.transcription, rotation='horizontal')
        plt.show()
