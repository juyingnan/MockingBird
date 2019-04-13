import numpy as np
from bokeh.io import output_file, show
from bokeh.layouts import gridplot, column  # , column
from bokeh.models import ColumnDataSource, Select, CDSView, IndexFilter, Span, CustomJS  # , CustomJS
from bokeh.plotting import figure
from bokeh.transform import factor_mark, factor_cmap
from scipy import io
from sklearn import preprocessing

# def update_plot(attrname, old, new):
#     label = labels[label_select.value]
#     sample_left.scatter("xx_sample_projection", "yy_sample_projection", source=sample_source, fill_alpha=0.4, size=12,
#                         marker=factor_mark(label['real_label_list'], markers, label['standard_label_list']),
#                         color=factor_cmap(label['real_label_list'], 'Category10_8', label['standard_label_list']),
#                         legend=label['real_label_list'])


root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
file_name = 'mfcc.mat'
output_file("result/svd.html")
mat_path = root_path + file_name
statements = ['st 1: Kids are talking by the door', 'st 2: Dogs are sitting by the door']
genders = ['female', 'male']
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
markers = ['hex', 'triangle', 'circle', 'cross', 'diamond', 'square', 'x', 'inverted_triangle']
axis_threshold = 5
default_x_index = '1'
default_y_index = '2'
digits = io.loadmat(mat_path)
# X, y = digits.get('feature_matrix'), digits.get('emotion_label')[0]  # X: nxm: n=1440//sample, m=feature
# X = X[:, ::100]
X = digits.get('feature_matrix')
# t = 13
# X = X[:, :, t:t + 1]
# wanted_columns = [x for x in range(26) if x not in ([] + [6,13,16,19,20])]
# X = X[:, :, wanted_columns]
X = X.reshape(X.shape[0], -1)
n_samples, n_features = X.shape
print("{} samples, {} features".format(n_samples, n_features))

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
# highlight axis x & y
vline = Span(location=0, dimension='height', line_color='black', line_width=2)
hline = Span(location=0, dimension='width', line_color='black', line_width=2)

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


# feature_matrix = figure(title="feature matrix", tools=tools_list)
# feature_matrix.x_range.range_padding = feature_matrix.y_range.range_padding = 0
# feature_matrix.image(image=[X.transpose()], x=0, y=0, dw=20, dh=20, palette="Spectral11")

# feature projection calculation
xx_feature_projection_list = list()
xx_feature_correlation_list = list()
# eigenvalues, eigenvectors = np.linalg.eig(np.cov(X))  # values: nx1/1440x1, vectors: nxn/1440x1440
U, s, Vh = np.linalg.svd(X.transpose(), full_matrices=False)  # u: mxm, s: mx1, v:nxn/1440x1440
del U
del s
for axis_index in range(axis_threshold):
    ev1 = Vh[axis_index]
    xx_feature_projection_list.append(X.transpose().dot(ev1))

# feature correlation calculation
s_x = preprocessing.normalize(X.transpose())
normalized_vh = preprocessing.normalize(Vh)
for axis_index in range(axis_threshold):
    s_ev1 = normalized_vh[axis_index]
    xx_feature_correlation_list.append(s_x.dot(s_ev1))

# feature data
feature_data = {'current_projection_x': xx_feature_projection_list[0],
                'current_projection_y': xx_feature_projection_list[1],
                'current_correlation_x': xx_feature_correlation_list[0],
                'current_correlation_y': xx_feature_correlation_list[1],
                }
feature_source = ColumnDataSource(data=feature_data)


# feature vis
def create_feature_scatter(x_data, y_data, source, title='', x_axis_title='', y_axis_title=''):
    result_plot = figure(title=title, tools=tools_list)
    result_plot.xaxis.axis_label = x_axis_title
    result_plot.yaxis.axis_label = y_axis_title
    result_plot.scatter(x_data, y_data, source=source, fill_alpha=0.4, size=12)
    # highlight x y axes
    result_plot.renderers.extend([vline, hline])
    return result_plot


feature_left = create_feature_scatter(x_data="current_projection_x", y_data="current_projection_y",
                                      source=feature_source,
                                      title="features projection",
                                      x_axis_title='Projection on {}'.format(default_x_index),
                                      y_axis_title='Projection on {}'.format(default_y_index))
feature_right = create_feature_scatter(x_data="current_correlation_x", y_data="current_correlation_y",
                                       source=feature_source,
                                       title="features correlation",
                                       x_axis_title='Correlation on {}'.format(default_x_index),
                                       y_axis_title='Correlation on {}'.format(default_y_index))

# controls
feature_selection_dict = {}
for j in range(axis_threshold):
    feature_selection_dict[str(j + 1)] = j

feature_axis_x_select = Select(value=default_x_index, title='X-axis', options=sorted(feature_selection_dict.keys()))
feature_axis_y_select = Select(value=default_y_index, title='Y-axis', options=sorted(feature_selection_dict.keys()))
code = """
    var index = labels[cb_obj.value];
    console.log(index);
    var data = source.data;
    data['current_projection_'+ key] = projection_pool[index]
    data['current_correlation_'+ key] = correlation_pool[index]
    var ax
    for (ax of axis[0])
    {
        ax.axis_label = "Projection on " + cb_obj.value;
    }
    for (ax of axis[1])
    {
        ax.axis_label = "Correlation on " + cb_obj.value;
    }
    source.change.emit();
    """
feature_axis_x_select.js_on_change('value',
                                   CustomJS(args=dict(key='x', labels=feature_selection_dict, source=feature_source,
                                                      projection_pool=xx_feature_projection_list,
                                                      correlation_pool=xx_feature_correlation_list,
                                                      axis=[[feature_left.xaxis[0]], [feature_right.xaxis[0]]]),
                                            code=code))
feature_axis_y_select.js_on_change('value',
                                   CustomJS(args=dict(key='y', labels=feature_selection_dict, source=feature_source,
                                                      projection_pool=xx_feature_projection_list,
                                                      correlation_pool=xx_feature_correlation_list,
                                                      axis=[[feature_left.yaxis[0]], [feature_right.yaxis[0]]]),
                                            code=code))
feature_controls = column(feature_axis_x_select, feature_axis_y_select)

# SAMPLE
xx_sample_projection_list = list()
xx_sample_correlation_list = list()
# sample projection calculation
U, s, Vh = np.linalg.svd(X, full_matrices=False)
for axis_index in range(axis_threshold):
    ev1 = Vh[axis_index]
    xx_sample_projection_list.append(X.dot(ev1))

# X = np.dot(U * s, Vh)

# sample correlation calculation
s_x = preprocessing.normalize(X)
normalized_vh = preprocessing.normalize(Vh)
for axis_index in range(axis_threshold):
    s_ev1 = normalized_vh[axis_index]
    xx_sample_correlation_list.append(s_x.dot(s_ev1))

# sample data
raw_statement_label = digits.get('statement_label')[0]
raw_actor_label = digits.get('actor_label')[0]
raw_emotion_label = digits.get('emotion_label')[0]
sample_data = {'current_projection_x': xx_sample_projection_list[0],
               'current_projection_y': xx_sample_projection_list[1],
               'current_correlation_x': xx_sample_correlation_list[0],
               'current_correlation_y': xx_sample_correlation_list[1],
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

# controls
current_label = 'gender'
labels = {
    'emotion':
        {
            'real_label_list': 'emotion_label',
            'standard_label_list': emotions,
        },
    'gender':
        {
            'real_label_list': 'gender_label',
            'standard_label_list': genders,
        },
    'statement':
        {
            'real_label_list': 'statement_label',
            'standard_label_list': statements,
        },
}
label_select = Select(value=current_label, title='Label', options=sorted(labels.keys()))
current_label = labels[label_select.value]


def create_sample_scatter(x_data, y_data, source, label, title='', x_axis_title='', y_axis_title=''):
    result_plot = figure(title=title, tools=tools_list, tooltips=custom_tooltip)
    result_plot.xaxis.axis_label = x_axis_title
    result_plot.yaxis.axis_label = y_axis_title
    for cat_filter in label['standard_label_list']:
        index_list = []
        for i in range(len(source.data[label['real_label_list']])):
            if source.data[label['real_label_list']][i] == cat_filter:
                index_list.append(i)
        view = CDSView(source=source, filters=[IndexFilter(index_list)])
        result_plot.scatter(x_data, y_data, source=source, fill_alpha=0.4, size=12,
                            marker=factor_mark(label['real_label_list'], markers, label['standard_label_list']),
                            color=factor_cmap(label['real_label_list'], 'Category10_8', label['standard_label_list']),
                            # muted_color=factor_cmap(label['real_label_list'], 'Category10_8',
                            #                         label['standard_label_list']),
                            muted_alpha=0.1, view=view,
                            legend=cat_filter)
    result_plot.legend.click_policy = "mute"
    # highlight x y axes
    result_plot.renderers.extend([vline, hline])

    return result_plot


# sample vis
sample_plot_list = list()
for current_label_key in ['emotion', 'gender', 'statement']:
    current_label = labels[current_label_key]
    sample_plot_list.append(
        create_sample_scatter(x_data="current_projection_x", y_data="current_projection_y", source=sample_source,
                              label=current_label,
                              title="samples projection", x_axis_title='Projection on {}'.format(default_x_index),
                              y_axis_title='Projection on {}'.format(default_y_index)))
    sample_plot_list.append(
        create_sample_scatter(x_data="current_correlation_x", y_data="current_correlation_y", source=sample_source,
                              label=current_label,
                              title="samples correlation", x_axis_title='Correlation on {}'.format(default_x_index),
                              y_axis_title='Correlation on {}'.format(default_y_index)))

# controls
sample_axis_x_select = Select(value=default_x_index, title='X-axis', options=sorted(feature_selection_dict.keys()))
sample_axis_y_select = Select(value=default_y_index, title='Y-axis', options=sorted(feature_selection_dict.keys()))

sample_axis_x_select.js_on_change('value',
                                  CustomJS(args=dict(key='x', labels=feature_selection_dict, source=sample_source,
                                                     projection_pool=xx_sample_projection_list,
                                                     correlation_pool=xx_sample_correlation_list,
                                                     axis=[[sample_plot_list[i].xaxis[0] for i in [0, 2, 4]],
                                                           [sample_plot_list[i].xaxis[0] for i in [1, 3, 5]]]),
                                           code=code))
sample_axis_y_select.js_on_change('value',
                                  CustomJS(args=dict(key='y', labels=feature_selection_dict, source=sample_source,
                                                     projection_pool=xx_sample_projection_list,
                                                     correlation_pool=xx_sample_correlation_list,
                                                     axis=[[sample_plot_list[i].yaxis[0] for i in [0, 2, 4]],
                                                           [sample_plot_list[i].yaxis[0] for i in [1, 3, 5]]]),
                                           code=code))
sample_controls = column(sample_axis_x_select, sample_axis_y_select)

p = gridplot([[feature_left, feature_right, feature_controls],
              [sample_plot_list[0], sample_plot_list[1], sample_controls],
              [sample_plot_list[2], sample_plot_list[3]],
              [sample_plot_list[4], sample_plot_list[5]], ])

show(p)
# curdoc().add_root(column(p, controls))
