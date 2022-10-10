from bokeh.io import output_file, show
from bokeh.layouts import gridplot, column
from bokeh.models import ColumnDataSource, CDSView, IndexFilter, Panel, Tabs
from bokeh.plotting import figure
from bokeh.transform import factor_mark, factor_cmap
from sklearn.manifold import TSNE
from umap import UMAP
from scipy import io
import math
import sys


class MappingFunctions:
    def cal_umap(feature_mat, n=2):
        print("umap input shape:", feature_mat.shape)
        X = feature_mat.reshape(feature_mat.shape[0], -1)
        # n_samples, n_features = X.shape

        '''UMAP'''
        umap_2d = UMAP(n_components=n, init='random', random_state=42, n_neighbors=100)
        return umap_2d.fit_transform(X)

    def cal_tsne(feature_mat, n=2):
        print("tsne input shape:", feature_mat.shape)
        X = feature_mat.reshape(feature_mat.shape[0], -1)
        # n_samples, n_features = X.shape

        '''t-SNE'''
        tsne = TSNE(n_components=n, init='random', random_state=42)
        return tsne.fit_transform(X)


def create_sample_scatter(x_data, y_data, source, label, title='', x_axis_title='', y_axis_title='', unmute=''):
    result_plot = figure(title=title, tools=tools_list, tooltips=custom_tooltip)
    result_plot.xaxis.axis_label = x_axis_title
    result_plot.yaxis.axis_label = y_axis_title
    for cat_filter in label['standard_label_list']:
        index_list = []
        for i in range(len(source.data['label'])):
            if source.data['label'][i] == cat_filter:
                index_list.append(i)
        view = CDSView(source=source, filters=[IndexFilter(index_list)])
        current = result_plot.scatter(x_data, y_data, source=source, fill_alpha=0.4, size=8,
                                      marker=factor_mark('label', markers, label['standard_label_list']),
                                      color=factor_cmap('label', 'Category10_8', label['standard_label_list']),
                                      # muted_color=factor_cmap(label['real_label_list'], 'Category10_8',
                                      #                         label['standard_label_list']),
                                      muted_alpha=0.1, view=view,
                                      legend_label=cat_filter)
        if unmute != '':
            if cat_filter == unmute:
                current.muted = False
            else:
                current.muted = True
            result_plot.legend.visible = False
    # result_plot.add_layout(result_plot.legend[0], 'right')
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

DEBUG = False

root_path = r'D:\Projects\emotion_in_speech\vis_mat/'
feature_name = 'mfcc'

if len(sys.argv) >= 2:
    feature_name = sys.argv[1]
    # layer_index = sys.argv[2]

file_name = feature_name + '.mat'
mat_path = root_path + file_name
digits = io.loadmat(mat_path)

for mapping in ['tsne', 'umap']:
    output_file(f"result/{mapping}_{feature_name}.html")

    raw_statement_label = digits.get('statement_label')[0]
    raw_actor_label = digits.get('actor_label')[0]
    raw_emotion_label = digits.get('emotion_label')[0]
    statement_label = [statements[i - 1] for i in raw_statement_label]
    gender_label = [genders[i % 2] for i in raw_actor_label]
    emotion_label = [emotions[i - 1] for i in raw_emotion_label]

    X = digits.get('feature_matrix')
    N = X.shape[2] if not DEBUG else 2
    cols = round(math.sqrt(N))
    print('N: {}, cols: {}'.format(N, cols))

    # T-SNE
    mapping_function = getattr(MappingFunctions, f"cal_{mapping}")
    # all
    all_X = X.reshape(X.shape[0], -1)
    All_tsne = mapping_function(all_X)

    # layers
    tsne_list = list()
    for t in range(N):
        x_layer = X[:, :, t:t + 1]
        x_layer = x_layer.reshape(x_layer.shape[0], -1)
        X_tsne = mapping_function(x_layer)
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

        grid = gridplot(grid_list, ncols=cols, width=400, height=350)
        tab_list.append(Panel(child=column(plot_all, grid), title=label_key))

    # emotion 1/0 test
    for emo in emotions:
        # for label_key, labels in zip(['statement', 'gender', 'emotion'], [statement_label, gender_label, emotion_label]):
        data = {'X': All_tsne.T[0],
                'Y': All_tsne.T[1],
                'label': emotion_label
                }
        sample_source = ColumnDataSource(data=data)
        current_label = sample_labels['emotion']
        plot_all = create_sample_scatter(x_data="X", y_data="Y", source=sample_source,
                                         label=current_label,
                                         title="ALL: %s" % feature_name,
                                         unmute=emo)
        grid_list = list()
        for t in range(N):
            data = {'X': tsne_list[t].T[0],
                    'Y': tsne_list[t].T[1],
                    'label': emotion_label
                    }
            sample_source = ColumnDataSource(data=data)
            plot = create_sample_scatter(x_data="X", y_data="Y", source=sample_source,
                                         label=current_label,
                                         title="%s" % t,
                                         unmute=emo)
            grid_list.append(plot)

        grid = gridplot(grid_list, ncols=cols, width=400, height=350)
        tab_list.append(Panel(child=column(plot_all, grid), title=f'emotion-{emo}'))

    tabs = Tabs(tabs=tab_list)
    show(tabs)
