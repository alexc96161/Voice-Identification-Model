# Imports
import csv  
import wave
import pyaudio
import time
import os
import random
from AppKit import NSWorkspace
from pynput import keyboard

WAVFILE = "recorded_waves"


# Set envourment path
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Open and read CSV for sentences
sentences = []
with open('unique_sentences.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        sentences = row

SEED = random.seed()
SEED = 42

random.Random(SEED).shuffle(sentences)

#Have a way to check if in focus
def InFocus():
    activeAppName = NSWorkspace.sharedWorkspace().activeApplication()['NSApplicationName']
    if (activeAppName == "Code" or activeAppName == "Terminal"):
        return True
    return False

# Make a listener for keypresses for the audio recording
pressed = ""
def on_press(key):
    if InFocus():
        global pressed
        try:
            pressed = key.char
        except AttributeError:
            pressed = key.name

listener = keyboard.Listener(on_press=on_press)
listener.start()

# Get the next open file
def GetOpenFileIndex(filename):
    index = 0
    if os.path.exists(WAVFILE) == True:
        while True:
            file_name = filename + str(index) + ".wav"
            if os.path.exists(WAVFILE + "/" + file_name) == False:
                break
            index = index + 1   
    return index


# Audio Setup
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 #48000
RECORD_SECONDS_MAX = 20
WAVE_OUTPUT_FILENAME = "output"

# Audio recording key press
def RecordAudioUntilKeyPress():
    #Open a stream
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []
    # Make pressed be accessing the global variable.
    global pressed

    # Get an open file location
    index = GetOpenFileIndex(WAVE_OUTPUT_FILENAME)
    full_filepath = WAVFILE + "/" + WAVE_OUTPUT_FILENAME + str(index) + ".wav"
    
    # Record into the stream, allow inputs during such as well
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS_MAX)):
        data = stream.read(CHUNK)
        frames.append(data)
        if pressed == "enter":
            break
        if pressed == "r": 
            pressed = ""
            RecordAudioUntilKeyPress()
            return 2
        if pressed == "s":
            return -1

    print("* done recording")
    pressed = ""


    # Close the recording
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Write the file
    wf = wave.open(full_filepath, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("Press enter to continue, press s to skip and press r to re-record")
    # Deal with what the user wants to do.
    while True:
        if pressed == "enter":
            break
        elif pressed == "r":
            os.remove(full_filepath)
            RecordAudioUntilKeyPress()
            return 2
        elif pressed == "s":
            return -1
    return 1
            


#Go through all the sentences, stopping for user to press a key to continue
print("Press any key to get the next line:\n")

# Get the current position
i = 0
if os.path.isfile("position_storage.txt"):
    with open('position_storage.txt', 'r') as file:
        i = int(file.read())
        
# Go through the phrases one by one
while i < len(sentences):

    # Write the new position (So you can come back later and contenue)
    with open("position_storage.txt","w") as file: #in write mode
        file.write(str(i))

    # Print out the phrase
    element = sentences[i]
    print('\033[92m' + element + '\033[0m')
    pressed = ""

    print("Press enter to continue, s to skip, and b to go back")

    # Deal with user input after they see the phrase
    while True:
        if (pressed == "enter" or pressed == "s" or pressed == "b"): break
        if (pressed == "b" and i != 0): break
    if (pressed == "s"): 
        i += 1
        continue
    if (pressed == "b" and i != 0):
        i -= 1
        continue
    
    # Countdown 
    # or 2 seconds before audio gets recorded
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("0")
    # Extra buffer
    time.sleep(0.5)
    print("start")

    # Record
    pressed = ""
    return_code = RecordAudioUntilKeyPress()
    i += 1    