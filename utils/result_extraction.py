import os


def extract_accuracy(path):
    start_word = 'Test accuracy:'
    with open(path) as f:
        for line in f:
            if line.startswith(start_word):
                accuracy = line.split(start_word)[1].strip()[:6]
                return accuracy
    return 'N/A'


root_path = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24/'

for phase in ['predict', 'train']:
    for split_method in ['rep', 'sen']:
        for method in ['stft', 'mfcc', 'logf', 'mfcc_logf']:
            print(phase, split_method, method)
            for length in ['050', '100', '150']:
                result = []
                for step in ['025', '050', '075']:
                    file_name = 'log_' + phase + '_' + method + '_slice_' + \
                                length + '_' + step + '_' + split_method + '.log'
                    file_path = root_path + file_name
                    result.append(extract_accuracy(file_path))
                print('\t'.join(result))
            raw_file_path = root_path + 'log_' + phase + '_' + method + '_' + split_method + '.log'
            print(extract_accuracy(raw_file_path))
            print()
