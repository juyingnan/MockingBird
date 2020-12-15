import numpy as np


def get_data(raw_data, channel):
    x, y, z, sr, rep, sen, act = raw_data.get('feature_matrix'), raw_data.get('emotion_label')[0], \
                                 raw_data.get('intensity_label')[0], raw_data.get('sample_rate')[0], \
                                 raw_data.get('repetition_label')[0], raw_data.get('statement_label')[0], \
                                 raw_data.get('actor_label')[0]
    file_ids = None
    slice_ids = None
    if 'file_id' in raw_data:
        file_ids = raw_data.get('file_id')[0]
    if 'slice_id' in raw_data:
        slice_ids = raw_data.get('slice_id')[0]
    y = y - 1
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], channel))
    return x, y, z, sr, file_ids, slice_ids, rep, sen, act


def train_test_rep_split(raw_data, channel, rate=1.0):
    x, y, z, sr, file_ids, slice_ids, rep, sen, act = get_data(raw_data, channel)

    _train_x = []
    _train_y = []
    _test_x = []
    _test_y = []
    index = int(len(x) * rate)
    assert len(x) == len(y) == len(rep)
    for i in range(len(x)):
        if i <= index and rep[i] == 2:
            _test_x.append(x[i])
            _test_y.append(y[i])
        else:
            _train_x.append(x[i])
            _train_y.append(y[i])
    assert len(_train_x) == len(_train_y)
    assert len(_test_x) == len(_test_y)
    return np.array(_train_x), np.array(_train_y), np.array(_test_x), np.array(_test_y)


def train_test_rep_split2(raw_data, channel, rate=0.2):
    x, y, z, sr, file_ids, slice_ids, rep, sen, act = get_data(raw_data, channel)

    assert len(x) == len(y)
    index = int(len(x) * rate)
    _train_x = x[index:]
    _train_y = y[index:]
    _test_x = x[:index]
    _test_y = y[:index]
    assert len(_train_x) == len(_train_y)
    assert len(_test_x) == len(_test_y)
    return np.array(_train_x), np.array(_train_y), np.array(_test_x), np.array(_test_y)


def train_test_rep_split3(raw_data, channel, rate_start=0.0, rate_end=1.0):
    x, y, z, sr, file_ids, slice_ids, rep, sen, act = get_data(raw_data, channel)
    _train_x = []
    _train_y = []
    _test_x = []
    _test_y = []
    _normal_test_sets = [[], [], [], [], [], [], [], []]
    _strong_test_sets = [[], [], [], [], [], [], [], []]
    start_index = int(len(x) * rate_start)
    end_index = int(len(x) * rate_end)
    assert len(x) == len(y) == len(rep)
    for i in range(len(x)):
        if start_index <= i <= end_index and rep[i] == 2:
            _test_x.append(x[i])
            _test_y.append(y[i])
            # intensity
            if z[i] == 1:
                _normal_test_sets[y[i]].append(x[i])
            else:
                _strong_test_sets[y[i]].append(x[i])
        else:
            _train_x.append(x[i])
            _train_y.append(y[i])
    assert len(_train_x) == len(_train_y)
    assert len(_test_x) == len(_test_y)
    return np.array(_train_x), np.array(_train_y), np.array(_test_x), np.array(
        _test_y), _normal_test_sets, _strong_test_sets


def train_test_rep_split4(raw_data, channel, sep_criteria, is_test_only=False):
    x, y, z, sr, file_ids, slice_ids, rep, sen, act = get_data(raw_data, channel)
    _train_x = []
    _train_y = []
    _test_x = []
    _test_y = []
    _test_id = []
    _normal_test_sets = [[], [], [], [], [], [], [], []]
    _strong_test_sets = [[], [], [], [], [], [], [], []]
    assert len(x) == len(y) == len(rep)
    for i in range(len(x)):
        if (rep[i] == 2 and sep_criteria == 'rep') \
                or (sen[i] == 2 and sep_criteria == 'sen') \
                or (act[i] % 2 == 0 and sep_criteria == 'm2f') \
                or (act[i] % 2 == 1 and sep_criteria == 'f2m') \
                or (i >= len(x) / 2 and sep_criteria == 'act') \
                or (sep_criteria.startswith("male") and act[i] == int(sep_criteria[-2:])) \
                or (sep_criteria.startswith("female") and act[i] == int(sep_criteria[-2:])) \
                or (sep_criteria.startswith("single") and act[i] == int(sep_criteria[-2:])):
            _test_x.append(x[i])
            _test_y.append(y[i])
            if file_ids is not None and slice_ids is not None:
                _test_id.append([file_ids[i], slice_ids[i], z[i]])
            else:
                _test_id.append([i, 0, z[i]])
            # intensity
            if z[i] == 1:
                _normal_test_sets[y[i]].append(x[i])
            else:
                _strong_test_sets[y[i]].append(x[i])
        else:
            if (sep_criteria.startswith("male") and act[i] % 2 == 0) \
                    or (sep_criteria.startswith("female") and act[i] % 2 == 1):
                continue
            _train_x.append(x[i])
            _train_y.append(y[i])
    assert len(_train_x) == len(_train_y)
    assert len(_test_x) == len(_test_y)
    if not is_test_only:
        return np.array(_train_x), np.array(_train_y), np.array(_test_x), np.array(
            _test_y), _normal_test_sets, _strong_test_sets
    else:
        return np.array(_test_x), np.array(_test_y), _test_id
