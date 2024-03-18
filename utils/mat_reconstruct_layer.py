import os.path

import numpy as np
import sys
from scipy import io


def layer_discard(matrix, discard_list):
    result = np.copy(matrix)
    for index in discard_list:
        result[:, :, index] = 0
    return result


def layer_keep(matrix, keep_list):
    result = np.copy(matrix)
    for index in range(result.shape[-1]):
        if index not in keep_list:
            result[:, :, index] = 0
    return result


def layer_discard_overwrite(matrix, discard_list):
    result = np.copy(matrix)
    result = np.delete(result, discard_list, 2)
    zero_fill = np.zeros((result.shape[0], result.shape[1], len(discard_list)))
    result = np.append(result, zero_fill, axis=2)
    return result


if __name__ == '__main__':
    root_path = r'D:\Projects\emotion_in_speech\vis_mat/'
    file_name = 'mfcc.mat'
    to_discard_list = [1, 3, 5, 7, 8]

    if len(sys.argv) >= 3:
        file_name = sys.argv[1] + '.mat'
        to_discard_list = [int(item) for item in sys.argv[2:]]
        print(to_discard_list)

    mat_path = os.path.join(root_path, file_name)
    result_file_name_postfix = '-'.join([str(item) for item in to_discard_list])
    digits = io.loadmat(mat_path)
    X = digits.get('feature_matrix')
    print(X.shape)

    reconstruct_mat = layer_discard(X, to_discard_list)
    # reconstruct_mat_overwrite = layer_discard_overwrite(X, to_discard_list)

    digits['feature_matrix'] = np.array(reconstruct_mat)
    io.savemat(mat_path.replace('.mat', '_layer_dis_%s.mat' % result_file_name_postfix), mdict=digits)

    # digits['feature_matrix'] = np.array(reconstruct_mat_overwrite)
    # io.savemat(mat_path.replace('.mat', '_layer_%s_ow.mat' % result_file_name_postfix), mdict=digits)
