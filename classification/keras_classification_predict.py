import numpy as np
from scipy import io
# import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import dataset_split


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
    if _test_ids[0][2] == 1:
        count_list_normal[current_y] += 1
    else:
        count_list_strong[current_y] += 1
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

    # cm test for slices
    confusion_list_strong = []
    confusion_list_normal = []
    for i in range(len(emotion_list)):
        confusion_list_strong.append([])
        confusion_list_normal.append([])
        for j in range(len(emotion_list)):
            confusion_list_strong[-1].append(0)
            confusion_list_normal[-1].append(0)
    for i in range(len(x_test)):
        current_y = test_label[i]
        cat = get_max_and_confidence(results[i])[0]
        if test_ids[i][2] == 1:
            confusion_list_normal[current_y][cat] += 1
        else:
            confusion_list_strong[current_y][cat] += 1
    print('strong cm')
    print('\t', '\t'.join(emotion_list))
    for i in range(len(emotion_list)):
        print(emotion_list[i], '\t', '\t'.join([str(item) for item in confusion_list_strong[i]]))

    print('normal cm')
    print('\t', '\t'.join(emotion_list))
    for i in range(len(emotion_list)):
        print(emotion_list[i], '\t', '\t'.join([str(item) for item in confusion_list_normal[i]]))


root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
file_name = 'mfcc_logf_slice_150_025'
split_method = 'rep'
category_count = 7 + 1
c = 2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
model = load_model(root_path + '/model_' + file_name + '_' + split_method + '.h5')

# read image
mat_path = root_path + file_name
digits = io.loadmat(mat_path)
test_data, test_label, test_ids = dataset_split.train_test_rep_split4(digits, c, split_method, is_test_only=True)
x_test = test_data
y_test = test_label

print(x_test.shape[0], 'test samples')
y_test = keras.utils.to_categorical(y_test, category_count)

draw_confusion_matrix(x_test, test_label, test_ids)
