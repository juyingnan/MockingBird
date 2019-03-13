import numpy as np
from scipy import io
# import keras
# from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.backend import set_session


def get_data(raw_data):
    x, y, z, sr, file_ids, slice_ids, rep = raw_data.get('feature_matrix'), raw_data.get('emotion_label')[0], \
                                            raw_data.get('intensity_label')[0], raw_data.get('sample_rate')[0], \
                                            raw_data.get('file_id')[0], raw_data.get('slice_id')[0], \
                                            raw_data.get('repetition_label')[0]
    y = y - 1
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], c))
    return x, y, z, sr, file_ids, slice_ids, rep


def train_test_rep_split2(raw_data, rate=0.2):
    x, y, z, sr, file_ids, slice_ids, rep = get_data(raw_data)

    assert len(x) == len(y)
    index = int(len(x) * rate)
    _train_x = x[index:]
    _train_y = y[index:]
    _test_x = x[:index]
    _test_y = y[:index]
    assert len(_train_x) == len(_train_y)
    assert len(_test_x) == len(_test_y)
    return np.array(_train_x), np.array(_train_y), np.array(_test_x), np.array(_test_y)


def train_test_rep_split3(raw_data, rate=1.0):
    x, y, z, sr, file_ids, slice_ids, rep = get_data(raw_data)
    _test_x = []
    _test_y = []
    _test_id = []
    index = int(len(x) * rate)
    assert len(x) == len(y) == len(rep)
    for i in range(len(x)):
        if i <= index and rep[i] == 2:
            _test_x.append(x[i])
            _test_y.append(y[i])
            _test_id.append((file_ids[i], slice_ids[i], z[i]))
    assert len(_test_x) == len(_test_y)
    return np.array(_test_x), np.array(_test_y), _test_id


def get_max_and_confidence(pred_results):
    result_as_list = [v for v in pred_results]
    max_confidence = max(result_as_list)
    index = result_as_list.index(max_confidence)
    return index, max_confidence


def draw_confusion_matrix(_x_test, _test_label, _test_ids):
    emotion_list = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    count_list_strong = [0, 0, 0, 0, 0, 0, 0, 0]
    count_list_normal = [0, 0, 0, 0, 0, 0, 0, 0]
    correct_list_strong = [0, 0, 0, 0, 0, 0, 0, 0]
    correct_list_normal = [0, 0, 0, 0, 0, 0, 0, 0]
    confusion_list_strong = []
    confusion_list_normal = []
    for i in range(len(emotion_list)):
        confusion_list_strong.append([])
        confusion_list_normal.append([])
        for j in range(len(emotion_list)):
            confusion_list_strong[-1].append(0)
            confusion_list_normal[-1].append(0)
    current_file = _test_ids[0][0]
    current_y = _test_label[0]
    results = model.predict(np.array(_x_test))
    prob_list = results[0]
    for i in range(len(_x_test)):
        if _test_ids[i][0] == current_file:
            prob_list = prob_list + results[i]
        else:
            # finish last
            cat = get_max_and_confidence(prob_list)[0]
            if cat == current_y:
                if _test_ids[i][2] == 1:
                    correct_list_normal[current_y] += 1
                else:
                    correct_list_strong[current_y] += 1
            # confusion matrix
            if _test_ids[i][2] == 1:
                confusion_list_normal[current_y][cat] += 1
            else:
                confusion_list_strong[current_y][cat] += 1
            # start new
            current_file = _test_ids[i][0]
            current_y = _test_label[i]
            prob_list = results[i]
            if _test_ids[i][2] == 1:
                count_list_normal[current_y] += 1
            else:
                count_list_strong[current_y] += 1
    print('Test accuracy:\t{}'.format(
        (sum(correct_list_strong) + sum(correct_list_normal)) / (sum(count_list_strong) + sum(count_list_normal))))
    print('\t'.join(emotion_list))
    print('\t'.join([str(correct_list_normal[i] / count_list_normal[i]) for i in range(len(count_list_normal))]))
    count_list_strong[0] = 1  # prevent 0/0
    print('\t'.join([str(correct_list_strong[i] / count_list_strong[i]) for i in range(len(count_list_strong))]))
    print('strong cm')
    print('\t', '\t'.join(emotion_list))
    for i in range(len(emotion_list)):
        print(emotion_list[i], '\t', '\t'.join([str(item) for item in confusion_list_strong[i]]))
    print('normal cm')
    print('\t', '\t'.join(emotion_list))
    for i in range(len(emotion_list)):
        print(emotion_list[i], '\t', '\t'.join([str(item) for item in confusion_list_normal[i]]))


h = 129
w = 186
c = 1
train_image_count = 100000
input_shape = (h, w, c)
learning_rate = 0.001
regularization_rate = 0.0001
category_count = 7 + 1
n_epoch = 100
mini_batch_size = 64
root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
model = Sequential()

# Layer 1
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

# Layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten
model.add(Flatten(input_shape=input_shape))

# fc layers
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularization_rate)))
model.add(Dense(category_count, activation='softmax', kernel_regularizer=regularizers.l2(regularization_rate)))
model.load_weights(root_path + '/stft_slice_model_weight.h5')

# read image
mat_path = root_path + 'stft_slice_256.mat'
digits = io.loadmat(mat_path)
test_data, test_label, test_ids = train_test_rep_split3(digits, 0.4)
x_test = test_data
y_test = test_label

print(x_test.shape[0], 'test samples')
y_test = keras.utils.to_categorical(y_test, category_count)

draw_confusion_matrix(x_test, test_label, test_ids)
