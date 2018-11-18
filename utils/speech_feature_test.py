from scipy import io
import numpy as np
from python_speech_features import mfcc
# from python_speech_features import fbank
from python_speech_features import logfbank
from python_speech_features import ssc

root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
mat_path = root_path + 'raw.mat'
digits = io.loadmat(mat_path)
X, y, z, sr, lengths = digits.get('feature_matrix'), digits.get('emotion_label')[0], digits.get('intensity_label')[0], \
                       digits.get('sample_rate')[0], digits.get('actual_length')[0]  # X: nxm: n=1440//sample, m=feature
# X = X[:, ::100]
n_samples, n_features = X.shape

mfcc_list = []
fbank_list = []
logfbank_list = []
ssc_list = []
all_list = []

count = 0
for i in range(len(X)):
    _data = X[i]
    _sr = sr[i]

    _mfcc = mfcc(_data, _sr, nfft=2048)
    _logfbank = logfbank(_data, _sr, nfft=2048)
    _ssc = ssc(_data, _sr, nfft=2048)
    all = np.concatenate((_mfcc, _logfbank, _ssc), axis=1)

    mfcc_list.append(_mfcc)
    # fbank_list.append(fbank(_data, _sr, nfft=2048))
    logfbank_list.append(_logfbank)
    ssc_list.append(_ssc)
    all_list.append(all)

    count += 1
    print("\rreading {0}/{1}".format(count, len(X)), end='')

io.savemat(root_path + 'mfcc.mat', mdict={'feature_matrix': np.array(mfcc_list),
                                          'sample_rate': sr,
                                          'actual_length': lengths,
                                          'emotion_label': y,
                                          'intensity_label': z
                                          })
io.savemat(root_path + 'logfbank.mat', mdict={'feature_matrix': np.array(logfbank_list),
                                              'sample_rate': sr,
                                              'actual_length': lengths,
                                              'emotion_label': y,
                                              'intensity_label': z
                                              })
io.savemat(root_path + 'ssc.mat', mdict={'feature_matrix': np.array(ssc_list),
                                         'sample_rate': sr,
                                         'actual_length': lengths,
                                         'emotion_label': y,
                                         'intensity_label': z
                                         })
io.savemat(root_path + 'mfcc_logfbank_ssc.mat', mdict={'feature_matrix': np.array(all_list),
                                                       'sample_rate': sr,
                                                       'actual_length': lengths,
                                                       'emotion_label': y,
                                                       'intensity_label': z
                                                       })

print()
