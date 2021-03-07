import numpy as np
import sys
from scipy import io


def eigen_discard(matrix, discard_list, need_reshape=False):
    if need_reshape:
        _u, _s, _vh = np.linalg.svd(matrix.reshape(matrix.shape[0], -1), full_matrices=False)
    else:
        _u, _s, _vh = np.linalg.svd(matrix, full_matrices=False)
    for index in discard_list:
        _s[index] = 0
    result = np.dot(_u * _s, _vh)
    return result

def eigen_keep(matrix, keep_list, need_reshape=False):
    if need_reshape:
        _u, _s, _vh = np.linalg.svd(matrix.reshape(matrix.shape[0], -1), full_matrices=False)
    else:
        _u, _s, _vh = np.linalg.svd(matrix, full_matrices=False)
    for index in range(len(_s)):
        if index not in keep_list:
            _s[index] = 0
    result = np.dot(_u * _s, _vh)
    return result

if __name__ == '__main__':
    root_path = r'D:\Projects\emotion_in_speech\vis_mat/'
    file_name = 'mfcc.mat'
    to_discard_list = [1]

    if len(sys.argv) >= 3:
        file_name = sys.argv[1] + '.mat'
        to_discard_list = [int(item) for item in sys.argv[2:]]
        print(to_discard_list)

    mat_path = root_path + file_name
    result_file_name_postfix = '-'.join([str(item) for item in to_discard_list])
    digits = io.loadmat(mat_path)
    X = digits.get('feature_matrix')
    original_shape = X.shape
    print("Original shape: {}".format(original_shape))

    reconstruct_mat = eigen_discard(X, to_discard_list, need_reshape=True)
    print("Reconstructed shape: {}".format(reconstruct_mat.shape))
    reconstruct_mat = reconstruct_mat.reshape(original_shape)

    # print(np.allclose(X, reconstruct_mat))
    digits['feature_matrix'] = np.array(reconstruct_mat)
    io.savemat(mat_path.replace('.mat', '_eigen_%s.mat' % result_file_name_postfix), mdict=digits)
