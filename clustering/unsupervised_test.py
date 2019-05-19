from sklearn import cluster
import csv
from scipy import io
from sklearn.model_selection import train_test_split
import dataset_split
import numpy as np

cluster_number = 2


def k_means_clustering(data, path='', log=True):
    print("Compute K-means clustering...")
    k_means = cluster.KMeans(n_clusters=cluster_number)
    k_means.fit(data)
    if path == '':
        write_csv(train_label, k_means.labels_ + 1, path='csv/kmeans_{0}.csv'.format(cluster_number))
    else:
        write_csv(train_label, k_means.labels_ + 1, path=path)


def hierarchical_clustering(data, path='', log=True):
    print("Compute unstructured hierarchical clustering...")
    # connectivity = kneighbors_graph(data, n_neighbors=5, include_self=False)
    ward = cluster.AgglomerativeClustering(n_clusters=cluster_number,  # connectivity=connectivity,
                                           linkage='ward').fit(data)
    if path == '':
        write_csv(train_label, ward.labels_ + 1, path='csv/hierarchical_{0}.csv'.format(cluster_number))
    else:
        write_csv(train_label, ward.labels_ + 1, path=path)


def write_csv(img_name_list, cat_list, path):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow((["NAME", "KMEANS_CAT{0}".format(cluster_number)]))
        lines = []
        for i in range(len(img_name_list)):
            lines.append([img_name_list[i], cat_list[i]])
        writer.writerows(lines)


root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
mat_file_name = 'mfcc.mat'
mat_path = root_path + mat_file_name
split_method = 'rep'
digits = io.loadmat(mat_path)
train_data, train_label = digits.get('feature_matrix'), digits.get('actor_label')[0]
k_means_clustering(train_data.reshape(len(train_data), -1))
