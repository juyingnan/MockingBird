import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math
import librosa
# from sklearn import preprocessing
from python_speech_features import mfcc
from python_speech_features import logfbank


def calculate_dft_matrix(n):
    exp1, exp2 = np.meshgrid(np.arange(n), np.arange(n))
    omega = np.exp(- 2 * math.pi * 1J / n)
    w = np.power(omega, exp1 * exp2)
    return w


def get_framed_matrix(a, window_type, frame_size):
    result = []
    window = signal.get_window(window_type, frame_size)
    start = 0
    while start + frame_size <= len(a):
        result.append(a[start:start + frame_size] * window)
        start += frame_size // 2
    result = np.asarray(result, np.float32).transpose()
    return result


def stft(a, window_type, frame_size):
    a_frames = get_framed_matrix(a, window_type, frame_size)
    a_y_mat = dft_mat.dot(a_frames)
    return a_y_mat


file_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24\Actor_01\03-01-01-01-01-01-01.wav'
audio, sr_audio = librosa.load(file_path, sr=None)
N = 1024
win_type = 'hann'
dft_mat = calculate_dft_matrix(N)
spectrogram = np.absolute(stft(audio, win_type, N))
_mfcc = mfcc(audio, sr_audio, nfft=2048, numcep=26, nfilt=26)
_logfbank = logfbank(audio, sr_audio, nfft=2048, nfilt=26)

fig = plt.figure()
fig.subplots_adjust(bottom=0.05)
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(hspace=0.35)

ax = fig.add_subplot(221)
ax.plot(audio)
ax.set_aspect(700000)
ax.set_title('original')

ax = fig.add_subplot(222)
ax.imshow(spectrogram, vmax=0.1)
ax.set_aspect(0.1)
ax.set_title('STFT')

ax = fig.add_subplot(223)
ax.imshow(_mfcc.T)
ax.set_aspect(5)
ax.set_title('_mfcc')

ax = fig.add_subplot(224)
ax.imshow(_logfbank.T)
ax.set_aspect(5)
ax.set_title('_logfbank')

plt.show()
