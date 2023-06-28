import os
import numpy as np
import librosa


# RMS (Root Mean Square): RMS is the square root of the mean of the squared signal values. It provides a measure
# of the overall signal strength, typically used to gauge the 'loudness' of an audio signal.
def calculate_volume(filename):
    # Load the audio file
    audio, sample_rate = librosa.load(filename)

    # Trim the silence from the audio
    audio, _ = librosa.effects.trim(audio)

    # Calculate the RMS value
    rms = np.sqrt(np.mean(audio ** 2))
    return rms


# Peak Amplitude: This is the maximum absolute value of the audio signal. While simple to calculate, it only
# represents the loudest moment in the audio and may not fully represent the overall volume.
def calculate_peak(filename):
    audio, _ = librosa.load(filename)
    audio, _ = librosa.effects.trim(audio)
    return np.max(np.abs(audio))


# Perceptual Loudness (LUFS - Loudness Units relative to Full Scale): This measure takes into account the human ear's
# sensitivity to different frequencies and the effects of temporal masking. It is designed to match human perception.
def calculate_loudness(filename):
    audio, _ = librosa.load(filename)
    audio, _ = librosa.effects.trim(audio)
    # Using amplitude_to_db to approximate LUFS
    return librosa.amplitude_to_db(np.abs(audio)).mean()


# Spectral Centroid: The spectral centroid is a measure of the "center of gravity" of the power spectrum of an audio
# signal. Higher values correspond to "brighter" or "lighter" sounds, whereas lower values correspond to "darker" or
# "heavier" sounds. It's often used to distinguish between different types of voices or instruments.
def calculate_spectral_centroid(filename):
    audio, sr = librosa.load(filename)
    audio, _ = librosa.effects.trim(audio)
    return librosa.feature.spectral_centroid(y=audio, sr=sr).mean()


# Pitch/Fundamental Frequency: Although not a measure of volume, the pitch or fundamental frequency of a voice is often
# used to differentiate between male and female voices. Male voices generally have a lower pitch on average.
def calculate_pitch(filename):
    audio, sr = librosa.load(filename)
    audio, _ = librosa.effects.trim(audio)
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
    pitch = np.sum(pitches * magnitudes) / np.sum(magnitudes)
    return pitch


actors = ['Actor_{:02d}'.format(i + 1) for i in range(24)]  # create the actor names
base_dir = r'D:\Projects\emotion_in_speech\Audio_Speech_Actors_01-24'  # replace with your actual directory

actor_measures = {}

for actor in actors:
    actor_dir = os.path.join(base_dir, actor)
    measures = {'rms': [], 'peak': [], 'loudness': [], 'spectral_centroid': [], 'pitch': []}

    for file in os.listdir(actor_dir):
        if file.endswith('.wav'):
            filepath = os.path.join(actor_dir, file)
            measures['rms'].append(calculate_volume(filepath))
            measures['peak'].append(calculate_peak(filepath))
            measures['loudness'].append(calculate_loudness(filepath))
            measures['spectral_centroid'].append(calculate_spectral_centroid(filepath))
            measures['pitch'].append(calculate_pitch(filepath))

    # Compute the average for each measure
    actor_measures[actor] = {k: np.mean(v) for k, v in measures.items()}

# Print out the average measures
for actor, measures in actor_measures.items():
    print('Average measures for {}: {}'.format(actor, measures))
