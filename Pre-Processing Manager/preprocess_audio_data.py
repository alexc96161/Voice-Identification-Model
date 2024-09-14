### Summary ###
###
### This script converts a folder with wav files, into mfcc json files in another
### it sets it up to be added to the AI, and has a max length.
###
### New: Now has options to shuffle, rename, and split by words
###
### Summary ###


### Settings ###
###
max_length = 500
wav_directory =  "/input_waves/"
mfcc_directory = "/output_mfccs/"
shuffle = True # Only useful when renamed
rename = True
rename_to = "" # Index is appended, only used if rename is on
###
### Settings ###


### Imports ###
###
import json
import librosa
import numpy as np
import multiprocessing
import time
import glob
import os
from joblib import Parallel, delayed
###
### Imports ###

########################################################################################################################

# Add to setting paths after import
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

wav_directory = dir_path + wav_directory
mfcc_directory = dir_path + mfcc_directory

# turn a wav into mfcc format, then write to the folder
def preprocess_wav(wav, i):
    print(i)
    # Get mfcc
    wave, sr = librosa.load(wav, mono=True)
    wave = librosa.resample(wave, orig_sr=sr, target_sr=22000)
    sr = 22000
    mfcc = librosa.feature.mfcc(y=wave, sr=sr)
    mfcc_trimmed = []
    # Limit length
    if (max_length < len(mfcc[0])):
        print("Warning: Over set max length, trimming. Index - " + str(i))
        for mfcc_list in mfcc:
            mfcc_trimmed.append(mfcc_list[:max_length])
            mfcc = mfcc_trimmed
    # Pad to setup for neural network input
    mfcc = np.pad(mfcc, ((0, 0), (0, max_length - len(mfcc[0]))), mode='constant', constant_values=0)
    # Write it to a json with same name
    realpath, extension = os.path.splitext(wav) 
    FILEPATH = mfcc_directory + os.path.basename(realpath) +".json"
    # Rename files by string and number (Optional)
    if (rename == True):
        FILEPATH = mfcc_directory + str(rename_to) + str(i) + ".json" 
    with open(FILEPATH, "w") as jsonfile:
        jsonfile.write(json.dumps(mfcc.tolist()))

# Make it get done quicker by running it many Parallel threads
def preprocess_data_parallel(data):
    start = time.time()

    # Parallelize
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(preprocess_wav)(data[i], i) for i in range(len(data)))

    # Give how long processing took
    end = time.time()
    print("Preprocessing elapsed: " + str(end - start))

# Get wav data and feed it into the preprocessing function.
data = glob.glob(wav_directory + "*.wav")

# Shuffle (Optional)
if (shuffle == True):
    import random 
    random.shuffle(data)
preprocess_data_parallel(data)