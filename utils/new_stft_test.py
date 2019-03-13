import numpy as np
from scipy import signal
# import matplotlib.pyplot as plt
from scipy import io

root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
mat_path = root_path + 'raw.mat'
digits = io.loadmat(mat_path)
X, y, z, sr, lengths, rep = digits.get('feature_matrix'), digits.get('emotion_label')[0], \
                            digits.get('intensity_label')[0], digits.get('sample_rate')[0], \
                            digits.get('actual_length')[0], digits.get('repetition_label')[0]
n_samples, n_features = X.shape

N = 512
win_type = 'hann'

stft_list = []

count = 0
for i in range(len(X)):
    _data = X[i]
    _sr = sr[i]
    f, t, spectrogram = signal.stft(_data, window=win_type, nperseg=N, nfft=512)
    stft_list.append(spectrogram)
    count += 1
    print("\rreading {0}/{1}".format(count, len(X)), end='')

io.savemat(root_path + 'stft_1024.mat', mdict={'feature_matrix': np.array(stft_list),
                                               'sample_rate': sr,
                                               'actual_length': lengths,
                                               'emotion_label': y,
                                               'intensity_label': z,
                                               'repetition_label': rep
                                               })
