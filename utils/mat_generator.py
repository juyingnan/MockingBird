import numpy as np
import os
import librosa
from scipy import io as sio

'''
modality_index = 0:         01 = full-AV, 02 = video-only, 03 = audio-only
vocal_channel_index = 1:    01 = speech, 02 = song
emotion_index = 2:          01 = neutral, 02 = calm, 03 = happy, 04 = sad, 
                            05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
intensity_index = 3:        01 = normal, 02 = strong
statement_index = 4:        01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"
repetition_index = 5:       01 = 1st repetition, 02 = 2nd repetition
actor_index = 6:            01 to 24. Odd numbered actors are male, even numbered actors are female
'''


def read_wav_files(path, is_normalized=False):
    cate = [path + folder for folder in os.listdir(path) if os.path.isdir(path + folder)]
    audio_list = []
    sr_list = []
    audio_length_list = []
    meta_info_lists = [[], [], [], [], [], [], []]
    separator = '-'

    max_length = 0
    for idx, folder in enumerate(cate):
        print('reading the audios:%s' % folder)
        count = 0
        for file_name in os.listdir(folder):
            if not os.path.isfile(os.path.join(folder, file_name)):
                continue
            file_path = os.path.join(folder, file_name)
            count += 1
            audio, sr_audio = librosa.load(file_path, sr=None)
            # audio = np.trim_zeros(audio)
            if is_normalized:
                # audio_list.append(audio / np.linalg.norm(audio))
                audio_list.append(2 * (audio - np.min(audio)) / np.ptp(audio) - 1)
            else:
                audio_list.append(audio)
            sr_list.append(sr_audio)
            audio_length_list.append(len(audio))
            if len(audio) > max_length:
                max_length = len(audio)

            for i in range(len(meta_info_lists)):
                file_name_prefix = file_name.split('.')[0]
                meta_info = str(file_name_prefix).split(separator)[i]
                meta_info_lists[i].append(meta_info)
            if count % 10 == 0:
                print("\rreading {0}/{1}".format(count, len(os.listdir(folder))), end='')
        print('\r', end='')
    for meta_info_list in meta_info_lists:
        assert len(meta_info_list) == len(audio_list)

    # make same length in matrix
    for i in range(len(audio_list)):
        audio_list[i] = np.append(audio_list[i], [[0] * (max_length - audio_length_list[i])])

    audio_array = np.asarray(audio_list, np.float32)
    return audio_array, np.asarray(sr_list, int), np.asarray(audio_length_list, int), np.asarray(
        meta_info_lists, int)


def normalize_features(data, v_max=1.0, v_min=0.0):
    data_array = np.asarray(data, np.float32)
    mins = np.min(data_array, axis=0)
    maxs = np.max(data_array, axis=0)
    rng = maxs - mins
    result = v_max - ((v_max - v_min) * (maxs - data_array) / rng)
    return result


if __name__ == '__main__':
    raw_file_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'
    np.seterr(all='ignore')
    raw_mat, sample_rates, lengths, meta_info_labels = read_wav_files(raw_file_path, is_normalized=True)

    sio.savemat(raw_file_path + 'raw_norm2.mat', mdict={'feature_matrix': raw_mat,
                                                       'sample_rate': sample_rates,
                                                       'actual_length': lengths,
                                                       'modality_label': meta_info_labels[0],
                                                       'vocal_channel_label': meta_info_labels[1],
                                                       'emotion_label': meta_info_labels[2],
                                                       'intensity_label': meta_info_labels[3],
                                                       'statement_label': meta_info_labels[4],
                                                       'repetition_label': meta_info_labels[5],
                                                       'actor_label': meta_info_labels[6],
                                                       })
