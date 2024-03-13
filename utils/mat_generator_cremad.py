import numpy as np
import csv
import os
import librosa
from scipy import io as sio
from tqdm import tqdm

'''
modality_index = 0:         01 = full-AV, 02 = video-only, 03 = audio-only
vocal_channel_index = 1:    01 = speech, 02 = song
emotion_index = 2:          01 = neutral, 02 = calm, 03 = happy, 04 = sad, 
                            05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
intensity_index = 3:        01 = normal, 02 = strong
statement_index = 4:        01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"
repetition_index = 5:       01 = 1st repetition, 02 = 2nd repetition
actor_index = 6:            01 to 24. Odd numbered actors are male, even numbered actors are female

Actors spoke from a selection of 12 sentences (in parentheses is the three letter acronym used in the second part of the filename):

0 It's eleven o'clock (IEO).
1 That is exactly what happened (TIE).
2 I'm on my way to the meeting (IOM).
3 I wonder what this is about (IWW).
4 The airplane is almost full (TAI).
5 Maybe tomorrow it will be cold (MTI).
6 I would like a new alarm clock (IWL)
7 I think I have a doctor's appointment (ITH).
8 Don't forget a jacket (DFA).
9 I think I've seen this before (ITS).
10 The surface is slick (TSI).
11 We'll stop in a couple of minutes (WSI).

The sentences were presented using different emotion (in parentheses is the three letter code used in the third part of the filename):

0 Anger (ANG)
1 Disgust (DIS)
2 Fear (FEA)
3 Happy/Joy (HAP)
4 Neutral (NEU)
5 Sad (SAD)

and emotion level (in parentheses is the two letter code used in the fourth part of the filename):

0 Low (LO)
1 Medium (MD)
2 High (HI)
3 Unspecified (XX)
'''

# dict of sentence
sentence_dict = {
    "IEO": 0,
    "TIE": 1,
    "IOM": 2,
    "IWW": 3,
    "TAI": 4,
    "MTI": 5,
    "IWL": 6,
    "ITH": 7,
    "DFA": 8,
    "ITS": 9,
    "TSI": 10,
    "WSI": 11,
}

# dict of emotion
emotion_dict = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5,
}

# dict of intensity
intensity_dict = {
    "LO": 0,
    "MD": 1,
    "HI": 2,
    "XX": 3,
}

gender_dict = {
    "Male": 0,
    "Female": 1,
}


# read VideoDemographics.csv, get actorID and gender relation
def read_gender_info(file_path):
    info = csv.reader(open(file_path, 'r'))
    # column actorID, Sex
    id_gender_dict = {}
    # skip the first line
    next(info)
    for line in info:
        id_gender_dict[line[0]] = gender_dict[line[2]]
    return id_gender_dict


def read_wav_files(path, is_normalized=False):
    audio_list = []
    sr_list = []  # sample rate
    audio_length_list = []
    meta_info_lists = [[], [], [], [], []]
    separator = '_'

    max_length = 0
    actor_gender_dict = read_gender_info(
        os.path.join(os.path.abspath(os.path.join(path, os.pardir)), 'VideoDemographics.csv'))
    print('reading the audios:')
    filenames = os.listdir(path)
    for i in tqdm(range(len(filenames)), desc="Reading"):
        file_name = filenames[i]
        if not os.path.isfile(os.path.join(path, file_name)):
            continue
        file_path = os.path.join(path, file_name)
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

        file_name_prefix = file_name.split('.')[0]
        meta_info = str(file_name_prefix).split(separator)
        # info list
        # 'actor_label': meta_info_labels[0],
        # 'statement_label': meta_info_labels[1],
        # 'emotion_label': meta_info_labels[2],
        # 'intensity_label': meta_info_labels[3],
        meta_info_lists[0].append(int(meta_info[0]))
        meta_info_lists[1].append(sentence_dict[meta_info[1]])
        meta_info_lists[2].append(emotion_dict[meta_info[2]])
        meta_info_lists[3].append(intensity_dict[meta_info[3]])
        meta_info_lists[4].append(actor_gender_dict[meta_info[0]])

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
    root_path = r'D:\Projects\emotion_in_speech\CREMA-D/'
    raw_file_path = os.path.join(root_path, 'AudioWAV/')
    np.seterr(all='ignore')
    raw_mat, sample_rates, lengths, meta_info_labels = read_wav_files(raw_file_path, is_normalized=True)

    mat_path = os.path.join(root_path, 'raw_norm2.mat')
    sio.savemat(mat_path, mdict={'feature_matrix': raw_mat,
                                 'sample_rate': sample_rates,
                                 'actual_length': lengths,
                                 'actor_label': meta_info_labels[0],
                                 'statement_label': meta_info_labels[1],
                                 'emotion_label': meta_info_labels[2],
                                 'intensity_label': meta_info_labels[3],
                                 })
