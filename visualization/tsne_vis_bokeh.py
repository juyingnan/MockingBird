from bokeh.io import output_file, show
from bokeh.layouts import gridplot, column
from bokeh.models import ColumnDataSource, CDSView, IndexFilter, Panel, Tabs
from bokeh.plotting import figure
from bokeh.transform import factor_mark, factor_cmap
from sklearn.manifold import TSNE
from scipy import io
import math


def create_sample_scatter(x_data, y_data, source, label, title='', x_axis_title='', y_axis_title=''):
    result_plot = figure(title=title, tools=tools_list, tooltips=custom_tooltip)
    result_plot.xaxis.axis_label = x_axis_title
    result_plot.yaxis.axis_label = y_axis_title
    for cat_filter in label['standard_label_list']:
        index_list = []
        for i in range(len(source.data['label'])):
            if source.data['label'][i] == cat_filter:
                index_list.append(i)
        view = CDSView(source=source, filters=[IndexFilter(index_list)])
        result_plot.scatter(x_data, y_data, source=source, fill_alpha=0.4, size=8,
                            marker=factor_mark('label', markers, label['standard_label_list']),
                            color=factor_cmap('label', 'Category10_8', label['standard_label_list']),
                            # muted_color=factor_cmap(label['real_label_list'], 'Category10_8',
                            #                         label['standard_label_list']),
                            muted_alpha=0.1, view=view,
                            legend=cat_filter)
    result_plot.legend.label_text_font_size = '8pt'
    result_plot.legend.click_policy = "mute"

    return result_plot


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
custom_tooltip = [
    ("index", "$index"),
    # ("(x,y)", "($x, $y)"),
    ("label", "@label"),
]

genders = ['female', 'male']
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
statements = ['st 1', 'st 2']
markers = ['hex', 'triangle', 'circle', 'cross', 'diamond', 'square', 'x', 'inverted_triangle']
sample_labels = {
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

root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
feature_name = 'mfcc'
file_name = feature_name + '.mat'
mat_path = root_path + file_name
digits = io.loadmat(mat_path)
output_file("result/tsne_%s.html" % feature_name)

raw_statement_label = digits.get('statement_label')[0]
raw_actor_label = digits.get('actor_label')[0]
raw_emotion_label = digits.get('emotion_label')[0]
statement_label = [statements[i - 1] for i in raw_statement_label]
gender_label = [genders[i % 2] for i in raw_actor_label]
emotion_label = [emotions[i - 1] for i in raw_emotion_label]

X = digits.get('feature_matrix')
N = X.shape[2]
cols = round(math.sqrt(N))
print('N: {}, cols: {}'.format(N, cols))

# T-SNE

# all
tsne = TSNE(n_components=2, init='random', random_state=42)
all_X = X.reshape(X.shape[0], -1)
All_tsne = tsne.fit_transform(all_X)
print(
    "ALL:\tAfter {} iter: Org data dimension is {}. Embedded data dimension is {}".format(tsne.n_iter,
                                                                                          all_X.shape[-1],
                                                                                          All_tsne.shape[-1]))

# layers
tsne_list = list()
for t in range(N):
    x_layer = X[:, :, t:t + 1]
    x_layer = x_layer.reshape(x_layer.shape[0], -1)

    '''t-SNE'''
    tsne = TSNE(n_components=2, init='random', random_state=42)
    X_tsne = tsne.fit_transform(x_layer)
    print(
        "Layer {}:\tAfter {} iter: Org data dimension is {}. Embedded data dimension is {}".format(t,
                                                                                                   tsne.n_iter,
                                                                                                   x_layer.shape[-1],
                                                                                                   X_tsne.shape[-1]))
    tsne_list.append(X_tsne)

tab_list = list()
for label_key, labels in zip(['statement', 'gender', 'emotion'], [statement_label, gender_label, emotion_label]):
    data = {'X': All_tsne.T[0],
            'Y': All_tsne.T[1],
            'label': labels
            }
    sample_source = ColumnDataSource(data=data)
    current_label = sample_labels[label_key]
    plot_all = create_sample_scatter(x_data="X", y_data="Y", source=sample_source,
                                     label=current_label,
                                     title="ALL: %s" % feature_name, )
    grid_list = list()
    for t in range(N):
        data = {'X': tsne_list[t].T[0],
                'Y': tsne_list[t].T[1],
                'label': labels
                }
        sample_source = ColumnDataSource(data=data)
        plot = create_sample_scatter(x_data="X", y_data="Y", source=sample_source,
                                     label=current_label,
                                     title="%s" % t, )
        grid_list.append(plot)

    grid = gridplot(grid_list, ncols=cols, plot_width=350, plot_height=350)
    tab_list.append(Panel(child=column(plot_all, grid), title=label_key))
tabs = Tabs(tabs=tab_list)
show(tabs)
