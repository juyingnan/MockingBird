import numpy as np
from bokeh.io import output_file
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


if __name__ == '__main__':
    root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
    file_name = 'mfcc.mat'
    output_file("result/svd.html")
    mat_path = root_path + file_name
    x_axis_index = 0
    y_axis_index = 1
    digits = io.loadmat(mat_path)
    # X, y = digits.get('feature_matrix'), digits.get('emotion_label')[0]  # X: nxm: n=1440//sample, m=feature
    # X = X[:, ::100]
    X = digits.get('feature_matrix')
    # t = 13
    # X = X[:, :, t:t + 1]
    X = X.reshape(X.shape[0], -1)
    n_samples, n_features = X.shape
    print("{} samples, {} features".format(n_samples, n_features))

    # eigenvalues, eigenvectors = np.linalg.eig(np.cov(X.transpose()))  # values: mx1/12x1, vectors: mxm/12x12
    U, s, Vh = np.linalg.svd(X, full_matrices=False)  # u: nxn/1440x1440, s: mx1, v:mxm
    s[1] = 0

    reconstruct_mat = np.dot(U * s, Vh)
    print(reconstruct_mat.shape)
