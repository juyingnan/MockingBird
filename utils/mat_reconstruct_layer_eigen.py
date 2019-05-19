import numpy as np
import sys
from scipy import io
import mat_reconstruct_svd

if __name__ == '__main__':
    root_path = r'D:\Projects\emotion_in_speech\vis_mat/'
    file_name = 'mfcc.mat'
    to_discard_list = ['1-2-5','2-2']

    if len(sys.argv) >= 3:
        file_name = sys.argv[1] + '.mat'
        to_discard_list = sys.argv[2:]
        print(to_discard_list)

    mat_path = root_path + file_name
    result_file_name_postfix = '-'.join([str(item) for item in to_discard_list])
    digits = io.loadmat(mat_path)
    X = digits.get('feature_matrix')

    for arg_pair in to_discard_list:
        layer_index = int(arg_pair.split('-')[0])
        eigen_index = [int(item) for item in arg_pair.split('-')[1:]]
        layer = X[:, :, layer_index]
        print('layer - eigen: {}, {}'.format(layer_index, eigen_index))
        original_shape = layer.shape
        print("Original shape: {}".format(original_shape))

        reconstruct_mat = mat_reconstruct_svd.eigen_discard(layer, eigen_index, need_reshape=True)
        print("Reconstructed shape: {}".format(reconstruct_mat.shape))
        reconstruct_mat = reconstruct_mat.reshape(original_shape)

        X[:, :, layer_index] = reconstruct_mat

    # print(np.allclose(X, reconstruct_mat))
    digits['feature_matrix'] = np.array(X)
    io.savemat(mat_path.replace('.mat', '_both_%s.mat' % result_file_name_postfix), mdict=digits)
