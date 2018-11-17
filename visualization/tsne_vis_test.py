# reference: https://github.com/DmitryUlyanov/Multicore-TSNE

import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from scipy import io

mat_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24\raw.mat'
digits = io.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('emotion_label')[0]  # X: nxm: n=500//sample, m=12,10,71,400//feature
X = X[:, ::100]
n_samples, n_features = X.shape

'''t-SNE'''
tsne = TSNE(n_jobs=8, init='random', random_state=501)
X_tsne = tsne.fit_transform(X)
print("After {} iter: Org data dimension is {}. Embedded data dimension is {}".format(tsne.n_iter, X.shape[-1],
                                                                                      X_tsne.shape[-1]))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    # plt.plot(X_norm[i, 0], X_norm[i, 1], '.', color=plt.cm.Set1(int(y[i][0])))
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.tab20(int(y[i])), fontdict={'size': 8})
plt.xticks([])
plt.yticks([])
plt.show()