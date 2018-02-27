from madmom.processors import SequentialProcessor
from general.Fprev_sub import Fprev_sub
import numpy as np

EPSILON = np.spacing(1)


def _nbf_2D(log_mel, nlen):
    """shift the feature and concatenate it in both left and right sides for nlen"""

    log_mel = np.array(log_mel).transpose()
    log_mel_out = np.array(log_mel, copy=True)
    for ii in range(1, nlen + 1):
        log_mel_right_shift = Fprev_sub(log_mel, w=ii)
        log_mel_left_shift = Fprev_sub(log_mel, w=-ii)
        log_mel_out = np.vstack((log_mel_right_shift, log_mel_out, log_mel_left_shift))
    feature = log_mel_out.transpose()
    return feature


class MadmomMelbankProcessor(SequentialProcessor):

    def __init__(self, fs, hopsize_t):
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.audio.filters import MelFilterbank
        from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                              LogarithmicSpectrogramProcessor)

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=fs)
        frames = FramedSignalProcessor(frame_size=2048, hopsize=int(fs*hopsize_t))
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
            norm_filters=True, unique_filters=False)
        spec = LogarithmicSpectrogramProcessor(log=np.log, add=EPSILON)

        single = SequentialProcessor([frames, stft, filt, spec])

        pre_processor = SequentialProcessor([sig, single])

        super(MadmomMelbankProcessor, self).__init__([pre_processor])


def get_log_mel_madmom(audio_fn, fs, hopsize_t, channel):
    """
    calculate log mel feature by madmom
    :param audio_fn:
    :param fs:
    :param hopsize_t:
    :param channel:
    :return:
    """
    madmomMelbankProc = MadmomMelbankProcessor(fs, hopsize_t)
    mfcc = madmomMelbankProc(audio_fn)

    if channel == 1:
        mfcc = _nbf_2D(mfcc, 7)
    else:
        mfcc_conc = []
        for ii in range(3):
            mfcc_conc.append(_nbf_2D(mfcc[:,:,ii], 7))
        mfcc = np.stack(mfcc_conc, axis=2)
    return mfcc


def feature_reshape(feature, nlen=10):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param feature:
    :param nlen:
    :return:
    """

    n_sample = feature.shape[0]
    n_row = 80
    n_col = nlen*2+1

    feature_reshaped = np.zeros((n_sample,n_row,n_col),dtype='float32')
    # print("reshaping feature...")
    for ii in range(n_sample):
        # print ii
        feature_frame = np.zeros((n_row,n_col),dtype='float32')
        for jj in range(n_col):
            feature_frame[:,jj] = feature[ii][n_row*jj:n_row*(jj+1)]
        feature_reshaped[ii,:,:] = feature_frame
    return feature_reshaped
