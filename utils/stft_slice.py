import numpy as np
import math
from scipy import signal
# import matplotlib.pyplot as plt
from scipy import io


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


root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
mat_path = root_path + 'raw_slice_050_025.mat'
digits = io.loadmat(mat_path)
X, y, z, sr, file_ids, slice_ids, rep = digits.get('feature_matrix'), digits.get('emotion_label')[0], \
                                        digits.get('intensity_label')[0], digits.get('sample_rate')[0], \
                                        digits.get('file_id')[0], digits.get('slice_id')[0], \
                                        digits.get('repetition_label')[0]
n_samples, n_features = X.shape

N = 512
win_type = 'hann'

dft_mat = calculate_dft_matrix(N)

# fig = plt.figure()
#
# ax = fig.add_subplot(121)
# spectrogram = np.absolute(stft(X[1], win_type, N))
# ax.imshow(spectrogram[:len(spectrogram) // 2 + 1], vmax=0.1, vmin=-0.1)
# ax.set_aspect(0.2)
# ax.set_title('spectrogram')
#
# plt.show()

stft_list = []

count = 0
for i in range(len(X)):
    _data = X[i]
    _sr = sr[i]
    spectrogram = np.absolute(stft(X[i], win_type, N))
    stft_list.append(spectrogram[:len(spectrogram) // 2 + 1])
    count += 1
    print("\rreading {0}/{1}".format(count, len(X)), end='')

print(np.array(stft_list[0]).shape)
# 129 x 186
io.savemat(root_path + 'stft_slice_256_2.mat', mdict={'feature_matrix': np.array(stft_list),
                                                      'sample_rate': sr,
                                                      'file_id': file_ids,
                                                      'slice_id': slice_ids,
                                                      'emotion_label': y,
                                                      'intensity_label': z,
                                                      'repetition_label': rep
                                                      })
