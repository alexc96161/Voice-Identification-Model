import pyaudio
import os
import wave
import sounddevice
import _portaudio as port

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
RECORD_SECONDS_MAX = 20
WAVE_OUTPUT_FILENAME = "output"

pa = pyaudio.PyAudio()

print("-------------------------------------------------------------------------------")
print("DEFAULT INPUT: " + str(pa.get_default_input_device_info()))
print("DEFAULT OUTPUT: " + str(pa.get_default_output_device_info()))
print("HOST: " + str((pa.get_default_host_api_info())))
print("-------------------------------------------------------------------------------")

version, description = sounddevice.get_portaudio_version()
print("sounddevice: {} {}".format(version, description))
print("pyaudio    : {} {}".format(pyaudio.get_portaudio_version(), pyaudio.get_portaudio_version_text()))

duration = 10.5  # seconds
samplerate = 48000
myrecording = sounddevice.rec(int(duration * samplerate), samplerate=samplerate, channels=2)


print("-------------------------------------------------------------------------------")

wav_file = wave.open('test.wav')

isSupported = pa.is_format_supported(rate=wav_file.getframerate(),     # sampling rate
    input_channels=wav_file.getnchannels(), # number of output channels
    input_format=pa.get_format_from_width(wav_file.getsampwidth()),  # sample format and length
    input_device=0,   # output device index
    )
print("SUPPORTED? " + str(isSupported))

stream_out = pa.open(
    rate=wav_file.getframerate(),     # sampling rate
    channels=wav_file.getnchannels(), # number of output channels
    format=pa.get_format_from_width(wav_file.getsampwidth()),  # sample format and length
    output=True,             # output stream flag
    output_device_index=1,   # output device index
    frames_per_buffer=1024,  # buffer length
)
print("-------------------------------------------------------------------------------")