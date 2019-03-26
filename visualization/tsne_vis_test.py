# reference: https://github.com/DmitryUlyanov/Multicore-TSNE

import matplotlib.pyplot as plt
from matplotlib import gridspec
# from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
from scipy import io
import math

root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
file_name = 'mfcc.mat'
mat_path = root_path + file_name
digits = io.loadmat(mat_path)

# X = X[:, ::100]
N = 26
cols = 6
rows = int(math.ceil(N / cols))

gs = gridspec.GridSpec(rows, cols)
fig = plt.figure()

fig.tight_layout()
for t in range(26):
    X, y = digits.get('feature_matrix'), digits.get('actor_label')[
        0]  # X: nxm: n=500//sample, m=12,10,71,400//feature
    X = X[:, :, t:t + 1]
    X = X.reshape(X.shape[0], -1)
    n_samples, n_features = X.shape

    '''t-SNE'''
    tsne = TSNE(n_components=2, init='random', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print("After {} iter: Org data dimension is {}. Embedded data dimension is {}".format(tsne.n_iter, X.shape[-1],
                                                                                          X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    ax = fig.add_subplot(gs[t])
    # plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        # plt.plot(X_norm[i, 0], X_norm[i, 1], '.', color=plt.cm.Set1(int(y[i][0])))
        ax.text(X_norm[i, 0], X_norm[i, 1], str(y[i] % 2 + 1), color=plt.cm.tab20(int(y[i] % 2 + 1)), fontdict={'size': 8})
    # ax.xticks([])
    # ax.yticks([])
plt.title(file_name)
plt.show()
