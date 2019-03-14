import numpy as np
from python_speech_features import mfcc
from python_speech_features import logfbank
from scipy import io

for length in ['050', '100', '150']:
    for step in ['025', '050', '075']:
        root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
        slice_parameter = length + '_' + step
        mat_file_name = 'raw_slice_' + slice_parameter + '.mat'
        mat_path = root_path + mat_file_name
        digits = io.loadmat(mat_path)
        X, y, z, sr, file_ids, slice_ids, rep = digits.get('feature_matrix'), digits.get('emotion_label')[0], \
                                                digits.get('intensity_label')[0], digits.get('sample_rate')[0], \
                                                digits.get('file_id')[0], digits.get('slice_id')[0], \
                                                digits.get('repetition_label')[0]
        n_samples, n_features = X.shape

        mfcc_list = []
        fbank_list = []
        logfbank_list = []
        all_list = []

        count = 0
        for i in range(len(X)):
            _data = X[i]
            _sr = sr[i]

            _mfcc = mfcc(_data, _sr, nfft=2048, numcep=26, nfilt=26)
            _logfbank = logfbank(_data, _sr, nfft=2048, nfilt=26)
            _all = np.dstack((_mfcc, _logfbank))

            mfcc_list.append(_mfcc)
            logfbank_list.append(_logfbank)
            all_list.append(_all)

            count += 1
            print("\rreading {0}/{1}".format(count, len(X)), end='')

        print(np.array(all_list[0]).shape)
        # 49 x 26 x 2
        io.savemat(root_path + 'mfcc_slice_' + slice_parameter + '.mat', mdict={'feature_matrix': np.array(mfcc_list),
                                                                                'sample_rate': sr,
                                                                                'file_id': file_ids,
                                                                                'slice_id': slice_ids,
                                                                                'emotion_label': y,
                                                                                'intensity_label': z,
                                                                                'repetition_label': rep
                                                                                })

        io.savemat(root_path + 'logf_slice_' + slice_parameter + '.mat',
                   mdict={'feature_matrix': np.array(logfbank_list),
                          'sample_rate': sr,
                          'file_id': file_ids,
                          'slice_id': slice_ids,
                          'emotion_label': y,
                          'intensity_label': z,
                          'repetition_label': rep
                          })

        io.savemat(root_path + 'mfcc_logf_slice_' + slice_parameter + '.mat',
                   mdict={'feature_matrix': np.array(all_list),
                          'sample_rate': sr,
                          'file_id': file_ids,
                          'slice_id': slice_ids,
                          'emotion_label': y,
                          'intensity_label': z,
                          'repetition_label': rep
                          })
