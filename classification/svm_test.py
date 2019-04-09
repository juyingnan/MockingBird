from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy import io
from sklearn.model_selection import train_test_split
import dataset_split
from sklearn.model_selection import GridSearchCV

# X = [[0], [1], [2], [3]]
# Y = [0, 1, 2, 3]
root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
mat_file_name = 'mfcc.mat'
mat_path = root_path + mat_file_name
split_method = 'rep'
digits = io.loadmat(mat_path)
train_data, train_label, test_data, test_label, normal_test_sets, strong_test_sets = \
    dataset_split.train_test_rep_split4(digits, 1, split_method)

# try filter some mfcc
wanted_columns = [x for x in range(26) if x not in ([] + [13, 16])]
train_data = train_data[:, :, wanted_columns]
test_data = test_data[:, :, wanted_columns]

train_data = train_data.reshape(train_data.shape[0], -1)
test_data = test_data.reshape(test_data.shape[0], -1)
X_train, X_test, Y_train, Y_test = train_data, test_data, train_label, test_label

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
svc = svm.SVC()
clf = GridSearchCV(svc, tuned_parameters)
clf.fit(X_train, Y_train)
print(clf.best_params_)

# clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
# clf.fit(X_train, Y_train)

svm_prediction = clf.predict(X_test)
accuracy = clf.score(X_test, Y_test)
print(accuracy)
cm = confusion_matrix(Y_test, svm_prediction)
print(cm)
