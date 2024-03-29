import numpy as np
from scipy import io
from tensorflow import keras
from tensorflow.keras.models import load_model
import dataset_split
import model_parameter
import logging
import sys
import time
from scipy import signal
import os

# Solve the "CUDNN_STATUS_ALLOC_FAILED" problem
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def parse_file_name(full_file_name):
    file_name_no_postfix = full_file_name.split('.')[0]
    if str(file_name_no_postfix).startswith('model_'):
        file_name_no_postfix = file_name_no_postfix[len('model_'):]
    _split_method = file_name_no_postfix.split('_')[-1]
    _meaningful_file_name = file_name_no_postfix.replace(f'_{_split_method}', '')
    print(f"Split method: {split_method}")
    print(f"meaningful_file_name: {_meaningful_file_name}")
    # _split_method = file_name_no_postfix[-3:]
    # _meaningful_file_name = file_name_no_postfix[:-4]
    return _meaningful_file_name, _split_method


def get_max_and_confidence(pred_results):
    result_as_list = [v for v in pred_results]
    max_confidence = max(result_as_list)
    index = result_as_list.index(max_confidence)
    return index, max_confidence


def calculate_weighted_prob_list(pred_result_list, window_type=''):
    weight_list = [1] * len(pred_result_list)
    if window_type == 'hann':
        weight_list = list(np.hanning(len(pred_result_list)))
    if window_type == 'tukey':
        weight_list = list(signal.windows.tukey(len(pred_result_list), alpha=0.25))
    return [sum([(pred_result_list[j][i] * weight_list[j]) for j in range(len(pred_result_list))]) for i in
            range(len(pred_result_list[0]))]


def get_early_predict(_x_test, _test_label, _test_ids, length, step):
    # emotion_list = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    results = model.predict(np.array(_x_test))

    # find longest possibility and set slice+id to start at 0
    longest = 0
    current_file = -1
    current_slice_id = -1
    for i in range(len(_x_test)):
        if _test_ids[i][0] != current_file:
            # start new
            current_file = _test_ids[i][0]
            current_slice_id = 0
        _test_ids[i][1] = current_slice_id
        current_slice_id += 1
        if _test_ids[i][1] > longest:
            longest = _test_ids[i][1]
    longest += 1

    normal_accuracy_list = []
    strong_accuracy_list = []
    accuracy_list = []
    uncomplete_predict_list = []

    for m in range(longest):
        count_list_strong = [0, 0, 0, 0, 0, 0, 0, 0]
        count_list_normal = [0, 0, 0, 0, 0, 0, 0, 0]
        correct_list_strong = [0, 0, 0, 0, 0, 0, 0, 0]
        correct_list_normal = [0, 0, 0, 0, 0, 0, 0, 0]
        uncomplete_file_list = []
        current_file = -1
        current_y = -1
        prob_list = []

        for i in range(len(_x_test)):
            if _test_ids[i][0] != current_file:
                # start new
                current_file = _test_ids[i][0]
                current_y = _test_label[i]
                prob_list = list()
                prob_list.append(results[i])
                if _test_ids[i][2] == 1:
                    count_list_normal[current_y] += 1
                else:
                    count_list_strong[current_y] += 1
            else:
                if _test_ids[i][1] <= m:
                    # prob_list = prob_list + results[i]
                    prob_list.append(results[i])
                else:
                    uncomplete_file_list.append(_test_ids[i][0])
            assert current_file != -1
            assert current_y != -1

            if i + 1 == len(_x_test) or _test_ids[i + 1][0] != current_file:  # last one or file end
                # finish last
                final_prob_list = calculate_weighted_prob_list(prob_list, window_type='')
                cat = get_max_and_confidence(final_prob_list)[0]
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
    current_file = -1
    current_y = -1
    prob_list = list()
    results = model.predict(np.array(_x_test))

    for i in range(len(_x_test)):
        if _test_ids[i][0] != current_file:
            # start new
            current_file = _test_ids[i][0]
            current_y = _test_label[i]
            prob_list = list()
            prob_list.append(results[i])
            if _test_ids[i][2] == 1:
                count_list_normal[current_y] += 1
            else:
                count_list_strong[current_y] += 1
        else:
            prob_list.append(results[i])
        assert current_file != -1
        assert current_y != -1

        if i + 1 == len(_x_test) or _test_ids[i + 1][0] != current_file:  # last one or file end
            # finish last
            final_prob_list = calculate_weighted_prob_list(prob_list, window_type='')
            cat = get_max_and_confidence(final_prob_list)[0]
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

root_path = r'D:\Projects\emotion_in_speech\vis_mat/'
meaningful_file_name = 'mfcc_logf'
split_method = 'rep'
category_count = 7 + 1

# override filepath via args
if len(sys.argv) >= 2:
    file_name = sys.argv[1]
    # c = int(sys.argv[3])
    meaningful_file_name, split_method = parse_file_name(file_name)
if len(sys.argv) >= 3:
    root_path = sys.argv[2]

c = 2 if 'mfcc_logf' in meaningful_file_name else 1

model_path = root_path + '/model_' + meaningful_file_name + '_' + split_method + '.h5'
print(model_path)
model = load_model(model_path)

# read image
mat_path = root_path + meaningful_file_name
digits = io.loadmat(mat_path)

real_split_method = split_method

if len(sys.argv) >= 4:
    real_split_method = sys.argv[3]

# sys.stdout = model_parameter.Logger(root_path + '/log_predict_' + meaningful_file_name + '_' + split_method + '.log')
sys.stdout = model_parameter.Logger(
    f'{root_path}/log_predict_{meaningful_file_name}_{split_method}_'
    f'{real_split_method}_{time.strftime("%Y%m%d-%H%M%S")}.log')

test_data, test_label, test_ids = dataset_split.train_test_rep_split4(digits, c, real_split_method, is_test_only=True)
x_test = test_data
y_test = test_label

print(x_test.shape[0], 'test samples')
y_test = keras.utils.to_categorical(y_test, category_count)

draw_confusion_matrix(x_test, test_label, test_ids)
if sum([test_ids[s][1] for s in range(len(test_ids))]) > 0:  # slices
    slice_len = int(meaningful_file_name.split('_')[-2]) / 100
    slice_step = int(meaningful_file_name.split('_')[-1]) / 100 * slice_len
    get_early_predict(x_test, test_label, test_ids, slice_len, slice_step)
