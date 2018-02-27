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

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

from general.parameters import *
from general.phonemeMap import dic_pho_label_inv, dic_pho_label

current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_folder,"..","gmmModels"))

class _LRHMM(object):
    '''
    make left to right HMM network
    '''

    def __init__(self,precision=np.double):
        self.n              = 2                                                 # hidden state number
        self.pi             = np.zeros((self.n), dtype=precision)                   # initial state distribution
        self.pi[0]          = 1.0
        self.A              = np.zeros((self.n,self.n), dtype=object)            # transition matrix
        self.gmmModel       = {}

        self.transcription  = []                                         # the phonetic transcription in X-SAMPA
        self.precision      = precision

    def getTranscription(self):
        return self.transcription

    def _makeNet(self):
        '''
        make a left to right HMM network
        :return:
        '''

        ##-- transition matrix
        probSelfTrans   = 0.9
        probNextTrans   = 1-probSelfTrans

        for ii in xrange(self.n):
            self.A[ii][ii] = probSelfTrans

        for ii in xrange(self.n-1):
            self.A[ii][ii+1] = probNextTrans

    def _gmmModel(self,model_path):
        '''
        load gmmModel
        :return:
        '''
        for state in list(set(self.transcription)):
            pkl_file = open(os.path.join(model_path,state+'.pkl'), 'rb')
            self.gmmModel[state] = pickle.load(pkl_file)
            pkl_file.close()

    def _mapBGMM(self, observations):
        '''
        gmm observation probability
        :param observations:
        :return:
        '''
        dim_t       = observations.shape[0]
        self.B_map  = {}
        # print self.transcription, self.B_map.shape
        for ii,state in enumerate(self.transcription):
            self.B_map[state] = self.gmmModel[state].score_samples(observations)


    @staticmethod
    def kerasModel(kerasModels_path):
        kerasModel = load_model(kerasModels_path)
        return kerasModel

    def mapBKeras(self,
                  observations,
                  kerasModel,
                  log=True,
                  obs_onset_phn=None,
                  use_joint_obs=False,
                  debug_mode=False):
        '''
        dnn observation probability
        :param observations:
        :return:
        '''
        ##-- set environment of the pdnn


        ##-- call pdnn to calculate the observation from the features
        obs = kerasModel.predict(observations, batch_size=128, verbose=0)

        if use_joint_obs: # use obs_joint_phn to adjust the observation matrix
            obs_mask = np.ones(obs.shape, dtype = np.float64) # init a mask matrix
            obs_onset_mean = np.mean(obs_onset_phn)
            penalizing_factor = obs_onset_mean * 1e-6

            bigN = min(3, obs.shape[1]) # choose N biggest observation values in each time stamps to augment
            idx_states = range(obs.shape[1])

            for ii_obs in range(obs.shape[0]):
                if obs_onset_phn[ii_obs] <= obs_onset_mean:
                    obs_mask[ii_obs, :] *= penalizing_factor # penalize the small obs_joint_phn values
                else:
                    idx_bigN = np.argpartition(obs[ii_obs, :], -bigN)[-bigN:] # find the first N biggest values idx
                    idx_small = np.setdiff1d(idx_states, idx_bigN)
                    obs_mask[ii_obs, :][idx_bigN] *= obs_onset_phn[ii_obs]
                    obs_mask[ii_obs, :][idx_small] *= penalizing_factor # penalize the small values

            if debug_mode:
                self._plotObs(obs.T, obs_onset_phn)
                self._plotObs(obs_mask.T, obs_onset_phn)

            obs *= obs_mask

            if debug_mode:
                self._plotObs(obs.T, obs_onset_phn)

        ##-- read the observation from the output
        if log:
            obs = np.log(obs+1e-32) # prevent 0

        if debug_mode:
            self._plotObs(obs.T, obs_onset_phn)

        # print obs.shape, observations.shape

        # dim_t       = obs.shape[0]
        # self.B_map  = np.zeros((self.n, dim_t), dtype=self.precision)
        # print self.transcription, self.B_map.shape
        # for ii,state in enumerate(self.transcription):
        #     self.B_map[ii,:] = obs[:, dic_pho_label[state]]

        self.B_map = {}
        # print self.transcription, self.B_map.shape
        for ii in range(obs.shape[1]):
            # print(dic_pho_label_inv[ii])
            # print(obs[:, ii])
            self.B_map[dic_pho_label_inv[ii]] = obs[:, ii]

    def _viterbi(self, observations):
        '''
        Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
        very similar to the forward-backward algorithm, with the added step of maximization and eventual
        backtracing.

        delta[t][i] = max(P[q1..qt=i,O1...Ot|model] - the path ending in Si and until time t,
        that generates the highest probability.

        psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1),
        i.e: the previous state.
        '''
        # similar to the forward-backward algorithm, we need to make sure that we're using fresh data for the given observations.
        self._mapBGMM(observations)

        delta = np.zeros((len(observations),self.n),dtype=self.precision)
        psi = np.zeros((len(observations),self.n),dtype=self.precision)

        # init
        for x in xrange(self.n):
            delta[0][x] = self.pi[x]*self.B_map[x][0]
            psi[0][x] = 0

        # induction
        for t in xrange(1,len(observations)):
            for j in xrange(self.n):
                for i in xrange(self.n):
                    if (delta[t][j] < delta[t-1][i]*self.A[i][j]):
                        delta[t][j] = delta[t-1][i]*self.A[i][j]
                        psi[t][j] = i
                delta[t][j] *= self.B_map[j][t]

        # termination: find the maximum probability for the entire sequence (=highest prob path)
        p_max = 0 # max value in time T (max)
        path = np.zeros((len(observations)),dtype=self.precision)
        for i in xrange(self.n):
            if (p_max < delta[len(observations)-1][i]):
                p_max = delta[len(observations)-1][i]
                path[len(observations)-1] = i

        # path backtracing
#        path = np.zeros((len(observations)),dtype=self.precision) ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
        for i in xrange(1, len(observations)):
            path[len(observations)-i-1] = psi[len(observations)-i][ path[len(observations)-i] ]
        return path

    def _viterbiLog(self, observations):
        '''
        Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
        very similar to the forward-backward algorithm, with the added step of maximization and eventual
        backtracing.

        delta[t][i] = max(P[q1..qt=i,O1...Ot|model] - the path ending in Si and until time t,
        that generates the highest probability.

        psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1),
        i.e: the previous state.
        '''
        # similar to the forward-backward algorithm, we need to make sure that we're using fresh data for the given observations.
        self._mapBGMM(observations)
        pi_log  = np.log(self.pi)
        A_log   = np.log(self.A)

        delta   = np.ones((len(observations),self.n),dtype=self.precision)
        delta   *= -float('Inf')
        psi     = np.zeros((len(observations),self.n),dtype=self.precision)

        # init
        for x in xrange(self.n):
            delta[0][x] = pi_log[x]+self.B_map[x][0]
            psi[0][x] = 0
        # print delta[0][:]

        # induction
        for t in xrange(1,len(observations)):
            for j in xrange(self.n):
                for i in xrange(self.n):
                    if (delta[t][j] < delta[t-1][i] + A_log[i][j]):
                        delta[t][j] = delta[t-1][i] + A_log[i][j]
                        psi[t][j] = i
                delta[t][j] += self.B_map[j][t]

        # termination: find the maximum probability for the entire sequence (=highest prob path)
        p_max = -float("inf") # max value in time T (max)
        path = np.zeros((len(observations)),dtype=self.precision)

        # last path is self.n-1
        # path[len(observations)-1] = self.n-1
        for i in xrange(self.n):
            if (p_max < delta[len(observations)-1][i]):
                p_max = delta[len(observations)-1][i]
                path[len(observations)-1] = i

        # path backtracing
#        path = np.zeros((len(observations)),dtype=self.precision) ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
        for i in xrange(1, len(observations)):
            path[len(observations)-i-1] = psi[len(observations)-i][ path[len(observations)-i] ]
        return path

    def _viterbiLogTranstionVarying(self,observations):
        '''
        loop entering transition proba varies on the transition time
        :param observations:
        :return:
        '''
        pass

    def _postProcessing(self,path,pho_duration_threshold=0.02):
        '''
        post processing of the decoded path
        set a duration threshold to jump
        :param path:
        :return:
        '''

        pdt_frame = pho_duration_threshold*fs/float(hopsize_phoneticSimilarity)

        path_post = [path[0]]
        counter = 1
        for ii in range(1,len(path)):
            if path[ii] != path[ii-1]:

                if counter <= pdt_frame:
                    for jj in xrange(ii-counter-1,ii):
                        path_post[jj] = path[ii-counter-1]

                counter = 1
            else:
                counter += 1
            path_post.append(path[ii])
        return path_post

    def _pathPlot(self,transcription_gt,path_gt,path):
        '''
        plot ground truth path and decoded path
        :return:
        '''
        plt.figure()
        print self.B_map.shape
        y = np.arange(self.B_map.shape[0]+1)
        x = np.arange(self.B_map.shape[1]) * hopsize_t / float(fs)
        plt.pcolormesh(x,y,self.B_map)
        plt.plot(x,path,'b',linewidth=3)
        plt.plot(x,path_gt,'k',linewidth=3)
        plt.xlabel('time (s)')
        plt.ylabel('states')
        plt.show()

    def _plotObs(self, obs, obs_joint_phn = None):

        tPlot, axes = plt.subplots(
            nrows=2, ncols=1, sharex=True, sharey=False,
            gridspec_kw={'height_ratios': [1, 3]})
        ax1, ax2 = axes[0], axes[1]

        # obs phns onset array
        x = np.arange(obs.shape[1]) * hopsize_t
        if obs_joint_phn is not None:
            ax1.plot(x, obs_joint_phn)

        # obs matrix plot
        y = np.arange(obs.shape[0])
        ax2.pcolormesh(x, y, obs)
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('states')
        plt.show()

