'''
 * Copyright (C) 2016  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuPhoneticSegmentation
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

####---- plot for aversano and hoang results ----####

##--instruction start--##
# make sure these two files exist in eval_path: hoa/hoa_mfccBands2_TRAIN.csv and ave/ave_bark_train.csv
# change eval_path to its location
##--end--##

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

eval_path = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/eval'

# activate latex text rendering
rc('text', usetex=True)

linestyles = ['--', '-.', '-', ':']
markerstyles = ['.','*','^','v']
markersize    = 5
labelFontsize = 15
tickLabelsize = 15
figsize       = (6,4)
legendsize    = 9

f, axarr = plt.subplots(2, 2, sharex='col', sharey=False, figsize=figsize)

####---- plot hoang
csv_filename = eval_path+'/hoa/hoa_mfccBands2_TRAIN.csv'

data         = np.loadtxt(csv_filename,delimiter=',')

fn  = 1     # figure number l
ln  = 2     # line number h1
x  = 3     # h2

HR  = 4
OS  = 5
FAR = 6
F   = 7
R   = 8
deletion    = 9
insertion   = 10

l_to_choose = [2]#[2,4,6,8,10]
h1_to_choose = [0.6,0.8,1.0]

for l_chosen in l_to_choose:
    data_l  = data[data[:,fn]==l_chosen,:]

    data_h1 = []
    for h1 in h1_to_choose:
        h1_mat = data_l[data_l[:,ln]==h1,:]
        data_h1.append(h1_mat)

    # f, axarr = plt.subplots(2, sharex=True, figsize=figsize)
    # for ii, h1_mat in enumerate(data_h1):
    #     axarr[0].plot(h1_mat[:,x],h1_mat[:,HR],label='h1='+str(0.6+ii*0.2))
    #
    # axarr[0].set_ylabel('HR')
    # axarr[0].legend()
    # axarr[0].set_title('l='+str(l_chosen))
    #
    # for ii, h1_mat in enumerate(data_h1):
    #     axarr[1].plot(h1_mat[:,x],h1_mat[:,OS])
    #
    # axarr[1].set_ylabel('OS')

    for ii, h1_mat in enumerate(data_h1):
        y = np.array(h1_mat[:,FAR])*100.0
        axarr[0,1].plot(h1_mat[:,x],y,linestyle=linestyles[ii],marker=markerstyles[ii],markersize=markersize,label='h1='+str(h1_to_choose[ii]))

    # axarr[1,0].set_ylabel(r'FAR (\%)',fontsize=labelFontsize)
    axarr[0,1].tick_params(axis='y', labelsize=tickLabelsize)
    axarr[0,1].legend(loc='upper right',prop={'size':legendsize})

    for ii, h1_mat in enumerate(data_h1):
        y = np.array(h1_mat[:,F])*100.0
        axarr[1,1].plot(h1_mat[:,x],y,linestyle=linestyles[ii],marker=markerstyles[ii],markersize=markersize,label='h1='+str(h1_to_choose[ii]))

    # axarr[1,1].set_ylabel(r'\textit{F}-value (\%)',fontsize=labelFontsize)
    axarr[1,1].tick_params(axis='x', labelsize=tickLabelsize)
    axarr[1,1].tick_params(axis='y', labelsize=tickLabelsize)
    axarr[1,1].legend(loc='upper right',prop={'size':legendsize})

    # for ii, h1_mat in enumerate(data_h1):
    #     axarr[3].plot(h1_mat[:,x],h1_mat[:,R])
    # axarr[3].set_ylabel('R')
    axarr[1,1].set_xlabel('h2',fontsize=labelFontsize)
    # plt.tight_layout()

    # plt.show()


####---- plot avesano
csv_filename = eval_path+'/ave/ave_bark_train.csv'

data         = np.loadtxt(csv_filename,delimiter=',')

fn  = 1 # a figure number
x   = 2 # b
ln  = 3 # c line number

HR  = 4
OS  = 5
FAR = 6
F   = 7
R   = 8
deletion    = 9
insertion   = 10

a_to_choose = [2]#[1,2,3,4]
c_to_choose = [3,5,7,9]

for a_chosen in a_to_choose:
    data_a  = data[data[:,fn]==a_chosen,:]
    data_c  = []
    for c in c_to_choose:
        c_mat = data_a[data_a[:,ln]==c,:]
        data_c.append(c_mat)

    # f, axarr = plt.subplots(2, sharex=True,figsize=figsize)

    # for ii, c_mat in enumerate(data_c):
    #     axarr[0].plot(c_mat[:,x],c_mat[:,HR],label=('c='+str((ii+1)*2+1)))
    #
    # axarr[0].set_ylabel('HR')
    # axarr[0].legend()
    # axarr[0].set_title('a='+str(a_chosen))

    # for ii, c_mat in enumerate(data_c):
    #     axarr[1].plot(c_mat[:,x],c_mat[:,OS])
    #
    #
    # axarr[1].set_ylabel('OS')

    for ii, c_mat in enumerate(data_c):
        y = np.array(c_mat[:,FAR])*100.0
        axarr[0,0].plot(c_mat[:,x],y,linestyle=linestyles[ii],marker=markerstyles[ii],markersize=markersize,label='c='+str(c_to_choose[ii]))
    axarr[0,0].tick_params(axis='y', labelsize=tickLabelsize)
    axarr[0,0].set_ylabel('FAR (\%)',fontsize=labelFontsize)
    axarr[0,0].legend(loc='lower right',prop={'size':legendsize})

    for ii, c_mat in enumerate(data_c):
        y = np.array(c_mat[:,F])*100.0
        axarr[1,0].plot(c_mat[:,x],y,linestyle=linestyles[ii],marker=markerstyles[ii],markersize=markersize,label='c='+str(c_to_choose[ii]))

    axarr[1,0].tick_params(axis='y', labelsize=tickLabelsize)
    axarr[1,0].tick_params(axis='x', labelsize=tickLabelsize)
    axarr[1,0].set_ylabel(r'\textit{F}-value (\%)',fontsize=labelFontsize)
    axarr[1,0].legend(loc='lower right',prop={'size':legendsize})


    # for ii, c_mat in enumerate(data_c):
    #     axarr[3].plot(c_mat[:,x],c_mat[:,R])
    #
    # axarr[3].set_ylabel('R')
    axarr[1,0].set_xlabel('b',fontsize=labelFontsize)

    plt.tight_layout()
    plt.show()

'''
####---- plot Win
csv_filename = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/eval/ave/ave_bark_train.csv'

data         = np.loadtxt(csv_filename,delimiter=',')

ln          = 0 # h2
fn          = 1 # alpha
x           = 2 # lambda

HR  = 3
OS  = 4
FAR = 5
F   = 6
R   = 7
deletion    = 8
insertion   = 9

h2_to_choose    = [0.0,0.02,0.04,0.06]
alpha_to_choose = [0.2,0.4,0.6,0.8,1.0]

for a_chosen in alpha_to_choose:
    data_a  = data[data[:,fn]==a_chosen,:]

    data_h2   = []
    for h2 in h2_to_choose:
        h2_mat = data_a[data_a[:,ln]==h2,:]
        data_h2.append(h2_mat)

    f, axarr = plt.subplots(4, sharex=True)
    for ii,h2_mat in enumerate(data_h2):
        axarr[0].plot(h2_mat[:,x],h2_mat[:,HR],label=('h2='+str(ii*0.02)))

    axarr[0].set_ylabel('HR')
    axarr[0].legend()
    axarr[0].set_title('alpha='+str(a_chosen))

    for ii, h2_mat in enumerate(data_h2):
        axarr[1].plot(h2_mat[:,x],h2_mat[:,OS])
    axarr[1].set_ylabel('OS')

    for ii, h2_mat in enumerate(data_h2):
        axarr[2].plot(h2_mat[:,x],h2_mat[:,F])
    axarr[2].set_ylabel('F')

    for ii, h2_mat in enumerate(data_h2):
        axarr[3].plot(h2_mat[:,x],h2_mat[:,R])

    axarr[3].set_ylabel('R')
    axarr[3].set_xlabel('lambda')


    plt.show()
'''