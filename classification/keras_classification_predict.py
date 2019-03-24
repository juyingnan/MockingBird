import numpy as np
from scipy import io
# import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import dataset_split
import model_parameter
import logging
import sys


def parse_file_name(full_file_name):
    file_name_no_postfix = full_file_name.split('.')[0]
    if str(file_name_no_postfix).startswith('model_'):
        file_name_no_postfix = file_name_no_postfix[len('model_'):]
    _split_method = file_name_no_postfix[-3:]
    _meaningful_file_name = file_name_no_postfix[:-4]
    return _meaningful_file_name, _split_method


def get_max_and_confidence(pred_results):
    result_as_list = [v for v in pred_results]
    max_confidence = max(result_as_list)
    index = result_as_list.index(max_confidence)
    return index, max_confidence


def get_early_predict(_x_test, _test_label, _test_ids, length, step):
    # emotion_list = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    results = model.predict(np.array(_x_test))

    # find longest possibility
    longest = 0
    for i in range(len(_x_test)):
        if _test_ids[i][1] > longest:
            longest = _test_ids[i][1]
    longest += 1

    normal_accuracy_list = []
    strong_accuracy_list = []
    accuracy_list = []
    uncomplete_predict_list = []

    for l in range(longest):
        count_list_strong = [0, 0, 0, 0, 0, 0, 0, 0]
        count_list_normal = [0, 0, 0, 0, 0, 0, 0, 0]
        correct_list_strong = [0, 0, 0, 0, 0, 0, 0, 0]
        correct_list_normal = [0, 0, 0, 0, 0, 0, 0, 0]
        uncomplete_file_list = []
        current_file = 0
        current_y = 0
        prob_list = []

        for i in range(len(_x_test)):
            if _test_ids[i][1] == 0:
                # start new
                current_file = _test_ids[i][0]
                current_y = _test_label[i]
                prob_list = results[i]
                if _test_ids[i][2] == 1:
                    count_list_normal[current_y] += 1
                else:
                    count_list_strong[current_y] += 1
            else:
                if _test_ids[i][1] <= l:
                    prob_list = prob_list + results[i]
                else:
                    uncomplete_file_list.append(_test_ids[i][0])
            if i + 1 == len(_x_test) or _test_ids[i + 1][0] != current_file:  # last one or file end
                # finish last
                cat = get_max_and_confidence(prob_list)[0]
                if cat == current_y:
                    if _test_ids[i][2] == 1:
                        correct_list_normal[current_y] += 1
                    else:
                        correct_list_strong[current_y] += 1

        normal_correct_count = sum(correct_list_normal)
        strong_correct_count = sum(correct_list_strong)
        correct_count = normal_correct_count + strong_correct_count
        normal_count = sum(count_list_normal)
        strong_count = sum(count_list_strong)
        all_count = normal_count + strong_count
        normal_accuracy_list.append(normal_correct_count / normal_count)
        strong_accuracy_list.append(strong_correct_count / strong_count)
        accuracy_list.append(correct_count / all_count)
        uncomplete_predict_list.append(len(set(uncomplete_file_list)))

    print('Early predict:')
    print('\t'.join([str(i + 1) for i in range(longest)]))
    print('\t'.join([str(length + step * i) for i in range(longest)]))
    print('\t'.join([str(normal_accuracy_list[i]) for i in range(longest)]))
    print('\t'.join([str(strong_accuracy_list[i]) for i in range(longest)]))
    print('\t'.join([str(accuracy_list[i]) for i in range(longest)]))
    print('\t'.join([str(uncomplete_predict_list[i]) for i in range(longest)]))


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
    current_file = 0
    current_y = 0
    prob_list = []
    results = model.predict(np.array(_x_test))

    for i in range(len(_x_test)):
        if _test_ids[i][1] == 0:
            # start new
            current_file = _test_ids[i][0]
            current_y = _test_label[i]
            prob_list = results[i]
            if _test_ids[i][2] == 1:
                count_list_normal[current_y] += 1
            else:
                count_list_strong[current_y] += 1
        else:
            prob_list = prob_list + results[i]
        if i + 1 == len(_x_test) or _test_ids[i + 1][0] != current_file:  # last one or file end
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

    print('Test accuracy:\t{}'.format(
        (sum(correct_list_strong) + sum(correct_list_normal)) / (sum(count_list_strong) + sum(count_list_normal))))
    print('\t'.join(emotion_list))
    print('\t'.join([str(correct_list_normal[i] / count_list_normal[i]) for i in range(len(count_list_normal))]))
    count_list_strong[0] = 1  # prevent 0/0
    print('\t'.join([str(correct_list_strong[i] / count_list_strong[i]) for i in range(len(count_list_strong))]))
    count_list_strong[0] = 0
    # all
    print('\t'.join(
        [str((correct_list_strong[i] + correct_list_normal[i]) / (count_list_strong[i] + count_list_normal[i]))
         for i in range(len(count_list_strong))]))
    print('normal cm')
    print('\t', '\t'.join(emotion_list))
    for i in range(len(emotion_list)):
        print(emotion_list[i], '\t', '\t'.join([str(item) for item in confusion_list_normal[i]]))
    print('strong cm')
    print('\t', '\t'.join(emotion_list))
    for i in range(len(emotion_list)):
        print(emotion_list[i], '\t', '\t'.join([str(item) for item in confusion_list_strong[i]]))
    # all cm
    print('all cm')
    print('\t', '\t'.join(emotion_list))
    for i in range(len(emotion_list)):
        print(emotion_list[i], '\t', '\t'.join([str(confusion_list_normal[i][j] + confusion_list_strong[i][j]) for j in
                                                range(len(confusion_list_normal[i]))]))

    # cm test for slices
    print('\ncm for slices')
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

    print('normal cm (slice)')
    print('\t', '\t'.join(emotion_list))
    for i in range(len(emotion_list)):
        print(emotion_list[i], '\t', '\t'.join([str(item) for item in confusion_list_normal[i]]))
    print('strong cm (slice)')
    print('\t', '\t'.join(emotion_list))
    for i in range(len(emotion_list)):
        print(emotion_list[i], '\t', '\t'.join([str(item) for item in confusion_list_strong[i]]))
    print('all cm (slice)')
    print('\t', '\t'.join(emotion_list))
    for i in range(len(emotion_list)):
        print(emotion_list[i], '\t', '\t'.join([str(confusion_list_normal[i][j] + confusion_list_strong[i][j]) for j in
                                                range(len(confusion_list_normal[i]))]))


# Disable Tensorflow debugging information
logging.getLogger("tensorflow").setLevel(logging.ERROR)

root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
meaningful_file_name = 'mfcc_logf_slice_150_025'
split_method = 'rep'
category_count = 7 + 1

# override filepath via args
if len(sys.argv) >= 2:
    file_name = sys.argv[1]
    # c = int(sys.argv[3])
    meaningful_file_name, split_method = parse_file_name(file_name)
c = 2 if 'mfcc_logf' in meaningful_file_name else 1

sys.stdout = model_parameter.Logger(root_path + '/log_predict_' + meaningful_file_name + '_' + split_method + '.log')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
model = load_model(root_path + '/model_' + meaningful_file_name + '_' + split_method + '.h5')

# read image
mat_path = root_path + meaningful_file_name
digits = io.loadmat(mat_path)
test_data, test_label, test_ids = dataset_split.train_test_rep_split4(digits, c, split_method, is_test_only=True)
x_test = test_data
y_test = test_label

print(x_test.shape[0], 'test samples')
y_test = keras.utils.to_categorical(y_test, category_count)

draw_confusion_matrix(x_test, test_label, test_ids)
if sum([test_ids[s][1] for s in range(len(test_ids))]) > 0:  # slices
    slice_len = int(meaningful_file_name.split('_')[-2]) / 100
    slice_step = int(meaningful_file_name.split('_')[-1]) / 100 * slice_len
    get_early_predict(x_test, test_label, test_ids, slice_len, slice_step)
