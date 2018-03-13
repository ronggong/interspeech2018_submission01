import os
import pickle
import numpy as np
import soundfile as sf
from keras.models import load_model
from audio_preprocessing import get_log_mel_madmom
from audio_preprocessing import feature_reshape

import pyximport
pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

import viterbiDecodingPhonemeSeg

from general.parameters import hopsize_t
from general.parameters import varin
from general.utilFunctions import smooth_obs

root_path = os.path.join(os.path.dirname(__file__))

joint_cnn_model_path = os.path.join(root_path, 'cnnModels', 'joint')

# load keras joint cnn model
model_joint = load_model(os.path.join(joint_cnn_model_path, 'jan_joint0.h5'))

# load log mel feature scaler
scaler_joint = pickle.load(open(os.path.join(joint_cnn_model_path, 'scaler_joint.pkl'), 'rb'))

# load wav, duration and labels
wav_file = './temp/wav_line_0.wav'
syllable_durations_file = './temp/syllable_score_durs_0.pkl'
phoneme_durations_file = './temp/phoneme_score_durs_grouped_by_syllables_0.pkl'
syllable_labels_file = './temp/syllable_score_labels_0.pkl'
phoneme_labels_file = './temp/phoneme_score_labels_grouped_by_syllables_0.pkl'

syllable_durations = pickle.load(open(syllable_durations_file, 'rb'))
phoneme_durations = np.load(open(phoneme_durations_file, 'rb'))
syllable_labels = pickle.load(open(syllable_labels_file, 'rb'))
phoneme_labels = pickle.load(open(phoneme_labels_file, 'rb'))

print('syllable durations (second):')
print(syllable_durations)
print('\n')

print('phoneme durations (second), each sublist corresponds to a syllable:')
print(phoneme_durations)
print('\n')

print('syllable labels:')
print(syllable_labels)
print('\n')

print('phoneme labels:')
print(phoneme_labels)
print('\n')

# get wav duration
data_wav, fs_wav = sf.read(wav_file)
time_wav = len(data_wav)/float(fs_wav)

# calculate log mel feature
log_mel = get_log_mel_madmom(wav_file, fs=fs_wav, hopsize_t=hopsize_t, channel=1)
log_mel = scaler_joint.transform(log_mel)
log_mel = feature_reshape(log_mel, nlen=7)
log_mel = np.expand_dims(log_mel, axis=1)

# get the onset detection function
obs_syllable, obs_phoneme = model_joint.predict(log_mel, batch_size=128, verbose=2)

# post-processing the detection function
obs_syllable = np.squeeze(obs_syllable)
obs_phoneme = np.squeeze(obs_phoneme)

obs_syllable = smooth_obs(obs_syllable)
obs_phoneme = smooth_obs(obs_phoneme)

obs_syllable[0] = 1.0
obs_syllable[-1] = 1.0

# normalize the syllable durations
syllable_durations *= time_wav / np.sum(syllable_durations)

# decoding syllable boundaries
boundaries_syllable = viterbiDecodingPhonemeSeg.viterbiSegmental2(obs_syllable, syllable_durations, varin)

# syllable boundaries
boundaries_syllable_start_time = np.array(boundaries_syllable[:-1])*hopsize_t
boundaries_syllable_end_time = np.array(boundaries_syllable[1:])*hopsize_t

# initialize phoneme boundaries arrays
boundaries_phoneme_start_time = np.array([])
boundaries_phoneme_end_time = np.array([])

for ii_syl_boundary in range(len(boundaries_syllable) - 1):
    dur_syl = boundaries_syllable_end_time[ii_syl_boundary] - \
              boundaries_syllable_start_time[ii_syl_boundary]

    frame_start_syl = boundaries_syllable[ii_syl_boundary]
    frame_end_syl = boundaries_syllable[ii_syl_boundary + 1]

    obs_phoneme_syl = obs_phoneme[frame_start_syl: frame_end_syl]

    obs_phoneme_syl[0] = 1.0
    obs_phoneme_syl[-1] = 1.0

    # phoneme score durs and labels for the current syllable, used in the decoding
    phoneme_score_durs_syl = np.array(phoneme_durations[ii_syl_boundary])

    # when there is only one phoneme contained in this syllable
    if len(phoneme_score_durs_syl) < 2:
        boundaries_phoneme_start_time = \
            np.concatenate((boundaries_phoneme_start_time, boundaries_syllable_start_time[ii_syl_boundary]))
        boundaries_phoneme_end_time = \
            np.concatenate((boundaries_phoneme_end_time, boundaries_syllable_end_time[ii_syl_boundary]))
        continue

    phoneme_score_durs_syl *= dur_syl / np.sum(phoneme_score_durs_syl)

    boundaries_phoneme_syl = \
        viterbiDecodingPhonemeSeg.viterbiSegmental2(obs_phoneme_syl, phoneme_score_durs_syl, varin)

    # phoneme boundariesUntitled Folder
    boundaries_phoneme_syl_start_time = \
        (np.array(boundaries_phoneme_syl[:-1]) + frame_start_syl) * hopsize_t
    boundaries_phoneme_syl_end_time = (np.array(boundaries_phoneme_syl[1:]) + frame_start_syl) * hopsize_t

    boundaries_phoneme_start_time = \
        np.concatenate((boundaries_phoneme_start_time, boundaries_phoneme_syl_start_time))
    boundaries_phoneme_end_time = \
        np.concatenate((boundaries_phoneme_end_time, boundaries_phoneme_syl_end_time))

print('Detected syllable onset times (second):')
print(boundaries_syllable_start_time)
print('\n')

print('Detected phoneme onset times (second)')
print(boundaries_phoneme_start_time)
print('\n')