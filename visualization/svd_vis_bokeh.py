import numpy as np
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import factor_mark, factor_cmap
from scipy import io
from sklearn import preprocessing

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
t = 1
X = X[:, :, t:t + 1]
X = X.reshape(X.shape[0], -1)
n_samples, n_features = X.shape
print("{} samples, {} features".format(n_samples, n_features))

# eigenvalues, eigenvectors = np.linalg.eig(np.cov(X))  # values: nx1/1440x1, vectors: nxn/1440x1440
U, s, Vh = np.linalg.svd(X.transpose(), full_matrices=False)  # u: mxm, s: mx1, v:nxn/1440x1440
del U
del s
# s[2:] = 0


# ax = fig.add_subplot(321)
# ax.imshow(X.transpose())
# ax.set_aspect(30)
# if "raw" in mat_path:
#     ax.set_aspect(0.01)
# ax.set_title('original_mat')
#
# ax = fig.add_subplot(322)
# ax.bar(np.arange(len(s)), s)
# ax.set_title('singular_values_feature')

# plot tools
tools_list = "pan," \
             "hover," \
             "box_select," \
             "lasso_select," \
             "box_zoom, " \
             "wheel_zoom," \
             "reset," \
             "save," \
             "help"

# feature projection calculation
ev1 = Vh[x_axis_index]  # ev: nx1/1440x1
ev2 = Vh[y_axis_index]
xx_feature_projection = X.transpose().dot(ev1)  # mxn.nx1 = mx1
yy_feature_projection = X.transpose().dot(ev2)

# feature correlation calculation
s_x = preprocessing.normalize(X.transpose())
normalized_vh = preprocessing.normalize(Vh)
s_ev1 = normalized_vh[x_axis_index]
s_ev2 = normalized_vh[y_axis_index]
xx_feature_correlation = s_x.dot(s_ev1)
yy_feature_correlation = s_x.dot(s_ev2)

# feature data
feature_data = {'xx_feature_projection': xx_feature_projection,
                'yy_feature_projection': yy_feature_projection,
                'xx_feature_correlation': xx_feature_correlation,
                'yy_feature_correlation': yy_feature_correlation,
                }
feature_source = ColumnDataSource(data=feature_data)

# feature vis
feature_left = figure(title="features projection", tools=tools_list)
feature_left.xaxis.axis_label = 'Projection on {}'.format(x_axis_index)
feature_left.yaxis.axis_label = 'Projection on {}'.format(y_axis_index)
feature_left.scatter("xx_feature_projection", "yy_feature_projection", source=feature_source, fill_alpha=0.4, size=12)

feature_right = figure(title="features correlation", tools=tools_list)
feature_right.xaxis.axis_label = 'Correlation on {}'.format(x_axis_index)
feature_right.yaxis.axis_label = 'Correlation on {}'.format(y_axis_index)
feature_right.scatter("xx_feature_correlation", "yy_feature_correlation", source=feature_source, fill_alpha=0.4,
                      size=12)

# eigenvalues, eigenvectors = np.linalg.eig(np.cov(X.transpose()))  # values: mx1/12x1, vectors: mxm/12x12
U, s, Vh = np.linalg.svd(X, full_matrices=False)  # u: nxn/1440x1440, s: mx1, v:mxm
# s[2:] = 0

# sample projection calculation
ev1 = Vh[x_axis_index]  # ev: nx1/1440x1
ev2 = Vh[y_axis_index]
xx_sample_projection = X.dot(ev1)  # nxm.mx1=nx1
yy_sample_projection = X.dot(ev2)

# sample correlation calculation
s_x = preprocessing.normalize(X)
normalized_vh = preprocessing.normalize(Vh)
s_ev1 = normalized_vh[x_axis_index]
s_ev2 = normalized_vh[y_axis_index]
xx_sample_correlation = s_x.dot(s_ev1)
yy_sample_correlation = s_x.dot(s_ev2)

# sample data
statements = ['st 1: Kids are talking by the door', 'st 2: Dogs are sitting by the door']
genders = ['female', 'male']
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
markers = ['hex', 'triangle', 'circle', 'cross', 'diamond', 'square', 'x', 'inverted_triangle']
raw_statement_label = digits.get('statement_label')[0]
raw_actor_label = digits.get('actor_label')[0]
raw_emotion_label = digits.get('emotion_label')[0]
sample_data = {'xx_sample_projection': xx_sample_projection,
               'yy_sample_projection': yy_sample_projection,
               'xx_sample_correlation': xx_sample_correlation,
               'yy_sample_correlation': yy_sample_correlation,
               'statement_label': [statements[i - 1] for i in raw_statement_label],
               'gender_label': [genders[i % 2] for i in raw_actor_label],
               'emotion_label': [emotions[i - 1] for i in raw_emotion_label],
               }
sample_source = ColumnDataSource(data=sample_data)
custom_tooltip = [
    ("index", "$index"),
    # ("(x,y)", "($x, $y)"),
    ("label", "@emotion_label"),
]

# sample vis
sample_left = figure(title="samples projection", tools=tools_list, tooltips=custom_tooltip)
sample_left.xaxis.axis_label = 'Projection on {}'.format(x_axis_index)
sample_left.yaxis.axis_label = 'Projection on {}'.format(y_axis_index)
sample_left.scatter("xx_sample_projection", "yy_sample_projection", source=sample_source, fill_alpha=0.4, size=12,
                    marker=factor_mark('emotion_label', markers, emotions),
                    color=factor_cmap('emotion_label', 'Category10_8', emotions),
                    legend='emotion_label')

sample_right = figure(title="samples correlation", tools=tools_list, tooltips=custom_tooltip)
sample_right.xaxis.axis_label = 'Correlation on {}'.format(x_axis_index)
sample_right.yaxis.axis_label = 'Correlation on {}'.format(y_axis_index)
sample_right.scatter("xx_sample_correlation", "yy_sample_correlation", source=sample_source, fill_alpha=0.4, size=12,
                     marker=factor_mark('emotion_label', markers, emotions),
                     color=factor_cmap('emotion_label', 'Category10_8', emotions),
                     legend='emotion_label')

p = gridplot([[feature_left, feature_right],
              [sample_left, sample_right]])
show(p)
