from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

from scipy import io
import numpy as np
from python_speech_features import mfcc
# from python_speech_features import fbank
from python_speech_features import logfbank
from python_speech_features import ssc

root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
mat_path = root_path + 'raw_norm2.mat'
digits = io.loadmat(mat_path)
X, y, z, sr, lengths = digits.get('feature_matrix'), digits.get('emotion_label')[0], digits.get('intensity_label')[0], \
                       digits.get('sample_rate')[0], digits.get('actual_length')[0]  # X: nxm: n=1440//sample, m=feature
# X = X[:, ::100]
n_samples, n_features = X.shape
index = 100

frequencies, times, spectrogram = signal.spectrogram(X[index], sr[index])

plt.pcolormesh(times, frequencies, spectrogram, vmin=-0.001, vmax=0.001)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
