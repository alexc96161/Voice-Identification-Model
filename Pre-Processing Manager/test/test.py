import numpy as np
import glob
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import pydub 
import librosa
import librosa.display
import matplotlib.pyplot as plt

def match_target_amplitude(aChunk, target_dBFS):
    # Normalize given audio chunk 
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

sound_file = pydub.AudioSegment.from_wav("test.wav")
normalized_chunk = match_target_amplitude(sound_file, -20.0)



print("Duration in miliseconds: " +str(int(sound_file.duration_seconds * 1000)))

audio_chunks = pydub.silence.split_on_silence(sound_file, 
    # must be silent for at least half a second
    min_silence_len=10,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-30
)

audio_ranges = pydub.silence.detect_nonsilent(sound_file, 
    # must be silent for at least half a second
    min_silence_len=10,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-25
)

print(audio_ranges)

data, sampling_rate = librosa.load("test.wav");
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data, sr=sampling_rate)

for i, chunk in enumerate(audio_chunks):
    if i >= len(audio_ranges):
        break
    plt.plot([audio_ranges[i][0] / 1000, audio_ranges[i][1] / 1000], [0.0, 0.0], 'r-', lw=3, scalex=False, scaley=False)
    out_file = ".//splitAudio//chunk{0}.wav".format(i)
    print ("exporting", out_file)
    chunk.export(out_file, format="wav")

plt.show()