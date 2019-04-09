import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from sklearn import preprocessing

root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
file_name = 'mfcc.mat'
mat_path = root_path + file_name
x_axis_index = 0
y_axis_index = 1
digits = io.loadmat(mat_path)
# X, y = digits.get('feature_matrix'), digits.get('emotion_label')[0]  # X: nxm: n=1440//sample, m=feature
# X = X[:, ::100]
X, y = digits.get('feature_matrix'), digits.get('statement_label')[0]
t = 1
X = X[:, :, t:t + 1]
X = X.reshape(X.shape[0], -1)
n_samples, n_features = X.shape
print("{} samples, {} features".format(n_samples, n_features))

eigenvalues, eigenvectors = np.linalg.eig(np.cov(X))  # values: nx1/1440x1, vectors: nxn/1440x1440

U, s, Vh = np.linalg.svd(X.transpose(), full_matrices=False)  # u: mxm, s: mx1, v:nxn/1440x1440
# s[2:] = 0

fig = plt.figure()
fig.subplots_adjust(bottom=0.05)
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(hspace=0.35)

ax = fig.add_subplot(321)
ax.imshow(X.transpose())
ax.set_aspect(30)
if "raw" in mat_path:
    ax.set_aspect(0.01)
ax.set_title('original_mat')

ax = fig.add_subplot(322)
ax.bar(np.arange(len(s)), s)
ax.set_title('singular_values_feature')

small_edge_index = 0.2
ax = fig.add_subplot(323)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.xlabel('Projection on {}'.format(x_axis_index))
plt.ylabel('Projection on {}'.format(y_axis_index))
ev1 = Vh[x_axis_index]  # ev: nx1/1440x1
ev2 = Vh[y_axis_index]
xx = X.transpose().dot(ev1)  # mxn.nx1 = mx1
yy = X.transpose().dot(ev2)
small_edge = (max(yy) - min(yy)) * small_edge_index
ax.set_ylim(min(yy) - small_edge if min(yy) <= -small_edge else -small_edge,
            max(yy) + small_edge if max(yy) >= small_edge else small_edge)
small_edge = (max(xx) - min(xx)) * small_edge_index
ax.set_xlim(min(xx) - small_edge if min(xx) <= -small_edge else -small_edge,
            max(xx) + small_edge if max(xx) >= small_edge else small_edge)
for i in range(xx.shape[0]):
    if "raw" in mat_path:
        ax.text(xx[i], yy[i], '.', fontdict={'size': 10})
    else:
        ax.text(xx[i], yy[i], str(i), fontdict={'size': 8})
ax.set_title('features_projection')

ax = fig.add_subplot(324)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.xlabel('Correlation on {}'.format(x_axis_index))
plt.ylabel('Correlation on {}'.format(y_axis_index))
s_x = preprocessing.normalize(X.transpose())
normalized_vh = preprocessing.normalize(Vh)
s_ev1 = normalized_vh[x_axis_index]
s_ev2 = normalized_vh[y_axis_index]
xx = s_x.dot(s_ev1)
yy = s_x.dot(s_ev2)
small_edge = (max(yy) - min(yy)) * small_edge_index
ax.set_ylim(min(yy) - small_edge if min(yy) <= -small_edge else -small_edge,
            max(yy) + small_edge if max(yy) >= small_edge else small_edge)
small_edge = (max(xx) - min(xx)) * small_edge_index
ax.set_xlim(min(xx) - small_edge if min(xx) <= -small_edge else -small_edge,
            max(xx) + small_edge if max(xx) >= small_edge else small_edge)
for i in range(xx.shape[0]):
    if "raw" in mat_path:
        ax.text(xx[i], yy[i], '.', fontdict={'size': 10})
    else:
        ax.text(xx[i], yy[i], str(i), fontdict={'size': 8})
ax.set_title('features_correlation')

eigenvalues, eigenvectors = np.linalg.eig(np.cov(X.transpose()))  # values: mx1/12x1, vectors: mxm/12x12
U, s, Vh = np.linalg.svd(X, full_matrices=False)  # u: nxn/1440x1440, s: mx1, v:mxm
# s[2:] = 0

ax = fig.add_subplot(325)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.xlabel('Projection on {}'.format(x_axis_index))
plt.ylabel('Projection on {}'.format(y_axis_index))
ev1 = Vh[x_axis_index]  # ev: nx1/1440x1
ev2 = Vh[y_axis_index]
xx = X.dot(ev1)  # nxm.mx1=nx1
yy = X.dot(ev2)
small_edge = (max(yy) - min(yy)) * small_edge_index
ax.set_ylim(min(yy) - small_edge if min(yy) <= -small_edge else -small_edge,
            max(yy) + small_edge if max(yy) >= small_edge else small_edge)
small_edge = (max(xx) - min(xx)) * small_edge_index
ax.set_xlim(min(xx) - small_edge if min(xx) <= -small_edge else -small_edge,
            max(xx) + small_edge if max(xx) >= small_edge else small_edge)
for i in range(X.shape[0]):
    ax.text(xx[i], yy[i], str(y[i] % 2 + 1), plt.cm.get_cmap('tab10')(int(y[i] % 2 + 1)), fontdict={'size': 8})
ax.set_title('samples_projection')

ax = fig.add_subplot(326)
ax.grid(True, which='both', color='#CFCFCF')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.xlabel('Correlation on {}'.format(x_axis_index))
plt.ylabel('Correlation on {}'.format(y_axis_index))
s_x = preprocessing.normalize(X)
normalized_vh = preprocessing.normalize(Vh)
s_ev1 = normalized_vh[x_axis_index]
s_ev2 = normalized_vh[y_axis_index]
xx = s_x.dot(s_ev1)
yy = s_x.dot(s_ev2)
small_edge = (max(yy) - min(yy)) * small_edge_index
ax.set_ylim(min(yy) - small_edge if min(yy) <= -small_edge else -small_edge,
            max(yy) + small_edge if max(yy) >= small_edge else small_edge)
small_edge = (max(xx) - min(xx)) * small_edge_index
ax.set_xlim(min(xx) - small_edge if min(xx) <= -small_edge else -small_edge,
            max(xx) + small_edge if max(xx) >= small_edge else small_edge)
for i in range(X.shape[0]):
    ax.text(xx[i], yy[i], str(y[i] % 2 + 1), color=plt.cm.get_cmap('tab10')(int(y[i] % 2 + 1)), fontdict={'size': 8})
ax.set_title('samples_correlation')

plt.show()
