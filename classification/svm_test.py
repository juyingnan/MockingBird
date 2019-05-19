from scipy import io
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import dataset_split
import mat_reconstruct_svd

root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
mat_file_name = 'mfcc_logf.mat'
mat_path = root_path + mat_file_name
split_method = 'rep'
digits = io.loadmat(mat_path)
train_data, train_label, test_data, test_label, normal_test_sets, strong_test_sets = \
    dataset_split.train_test_rep_split4(digits, 2, split_method)

# try filter some mfcc
# wanted_columns = [x for x in range(26) if x not in ([] + [13, 16])]
# train_data = train_data[:, :, wanted_columns]
# test_data = test_data[:, :, wanted_columns]

train_data = train_data.reshape(train_data.shape[0], -1)
test_data = test_data.reshape(test_data.shape[0], -1)

discard_eigne_list = [1]
if len(discard_eigne_list) > 0:
    train_data = mat_reconstruct_svd.eigen_discard(train_data, discard_eigne_list, need_reshape=False)
    test_data = mat_reconstruct_svd.eigen_discard(test_data, discard_eigne_list, need_reshape=False)
X_train, X_test, Y_train, Y_test = train_data, test_data, train_label, test_label

tuned_parameters = [  # {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    {'kernel': ['poly'], 'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
    {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000]}]
# tuned_parameters = [{'kernel': ['linear'], 'C': [1, ]}]

scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score, n_jobs=6)
    clf.fit(X_train, Y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    cm = confusion_matrix(Y_test, y_pred)
    print(cm)
