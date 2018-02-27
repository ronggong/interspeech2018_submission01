from general.parameters import *
from general.filePathHsmm import kerasScaler_path
from general.Fprev_sub import Fprev_sub
from audio_preprocessing import feature_reshape
import essentia.standard as ess
import pickle
import numpy as np

winAnalysis = 'hann'
N = 2 * framesize # padding 1 time framesize
SPECTRUM = ess.Spectrum(size=N)
MFCC = ess.MFCC(sampleRate=fs,
                highFrequencyBound=highFrequencyBound,
                inputSize=framesize + 1,
                numberBands=80)
WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize)


def getMFCCBands2D(audio, framesize, hopsize, nbf=False, nlen=10):

    """
    mel bands feature [p[0],p[1]]
    output feature for each time stamp is a 2D matrix
    it needs the array format float32
    :param audio:
    :param p:
    :param nbf: bool, if we need to neighbor frames
    :return:
    """

    mfcc = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame = WINDOW(frame)
        mXFrame = SPECTRUM(frame)
        bands,mfccFrame = MFCC(mXFrame)
        mfcc.append(bands)

    if nbf:
        mfcc = np.array(mfcc).transpose()
        mfcc_out = np.array(mfcc, copy=True)
        for ii in range(1,nlen+1):
            mfcc_right_shift = Fprev_sub(mfcc, w=ii)
            mfcc_left_shift = Fprev_sub(mfcc, w=-ii)
            mfcc_out = np.vstack((mfcc_right_shift, mfcc_out, mfcc_left_shift))
        feature = mfcc_out.transpose()
    else:
        feature = mfcc
    # the mel bands features
    feature = np.array(feature,dtype='float32')

    return feature


def mfccFeature_pipeline(filename_wav):
    audio = ess.MonoLoader(downmix = 'left', filename = filename_wav, sampleRate = fs)()
    scaler = pickle.load(open(kerasScaler_path,'rb'))

    feature = getMFCCBands2D(audio,framesize, hopsize, nbf=True)
    mfcc = np.log(100000 * feature + 1)
    feature = scaler.transform(mfcc)
    feature = feature_reshape(feature)

    return feature, mfcc