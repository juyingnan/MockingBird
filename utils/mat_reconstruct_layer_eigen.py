import numpy as np
import sys
from scipy import io
import mat_reconstruct_svd
import mat_reconstruct_layer

if __name__ == '__main__':
    root_path = r'D:\Projects\emotion_in_speech\vis_mat/'
    file_name = 'mfcc.mat'
    to_discard_list = ['1-4', '3-2', '5-1', '8-1', '9-2']
    is_keep_other_layers = True

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

        reconstruct_mat = mat_reconstruct_svd.eigen_keep(layer, eigen_index, need_reshape=True)
        print("Reconstructed shape: {}".format(reconstruct_mat.shape))
        reconstruct_mat = reconstruct_mat.reshape(original_shape)

        X[:, :, layer_index] = reconstruct_mat

    # layers in which linearly separable eigenvectors are discarded
    if not is_keep_other_layers:
        to_discard_layer = []
        for each in to_discard_list:
            to_discard_layer.append(int(each.split("-")[0]))

        X = mat_reconstruct_layer.layer_keep(X, to_discard_layer)
        print("discard other layers")
    else:
        print("keep other layers")

    # print(np.allclose(X, reconstruct_mat))
    digits['feature_matrix'] = np.array(X)
    io.savemat(mat_path.replace('.mat', '_keep_%s.mat' % result_file_name_postfix), mdict=digits)
