import numpy as np
from python_speech_features import mfcc
from python_speech_features import logfbank
from scipy import io, signal
from tqdm import tqdm


def get_slice_feature(original_mat_path):
    digits = io.loadmat(original_mat_path)
    x, sr, = digits.get('feature_matrix'), digits.get('sample_rate')[0]
    n_samples, n_features = x.shape
    print(n_samples, n_features)
    win_type = 'hann'
    stft_list = []
    mfcc_list = []
    logfbank_list = []
    all_list = []
    count = 0
    for i in tqdm(range(len(x)), desc="Processing"):
        _data = x[i]
        _sr = sr[i]

        # stft
        f, t, spectrogram = signal.stft(_data, window=win_type, nperseg=512, nfft=512)

        # mfcc
        _mfcc = mfcc(_data, _sr, nfft=2048, numcep=26, nfilt=26)

        # logfbank
        _logfbank = logfbank(_data, _sr, nfft=2048, nfilt=26)

        # mfcc + logfbank
        _all = np.dstack((_mfcc, _logfbank))

        stft_list.append(np.absolute(spectrogram.T))
        mfcc_list.append(_mfcc)
        logfbank_list.append(_logfbank)
        all_list.append(_all)

        count += 1
        print("\rreading {0}/{1}".format(count, len(x)), end='')

    print('stft shape: ', np.array(stft_list[0]).shape)
    print('mfcc+logfbank shape: ', np.array(all_list[0]).shape)

    # 49 x 26 x 2
    digits['feature_matrix'] = np.array(stft_list)
    io.savemat(original_mat_path.replace('raw', 'stft'), mdict=digits)
    digits['feature_matrix'] = np.array(mfcc_list)
    io.savemat(original_mat_path.replace('raw', 'mfcc'), mdict=digits)
    digits['feature_matrix'] = np.array(logfbank_list)
    io.savemat(original_mat_path.replace('raw', 'logf'), mdict=digits)
    digits['feature_matrix'] = np.array(all_list)
    io.savemat(original_mat_path.replace('raw', 'mfcc_logf'), mdict=digits)


root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
get_slice_feature(root_path + 'raw.mat')
for length in ['050', '100', '150']:
    for step in ['025', '050', '075']:
        slice_parameter = length + '_' + step
        mat_file_name = 'raw_slice_' + slice_parameter + '.mat'
        mat_path = root_path + mat_file_name
        get_slice_feature(mat_path)
