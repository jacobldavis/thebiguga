from scipy.io import wavfile

from utils import return_char

SEGMENTS = 20

# file_name = input("Enter the WAV file name (witho extension): ")
sample_rate, audio_data = wavfile.read('a.wav')

if audio_data.ndim == 2:
    audio_data = audio_data[:, 0]

N = len(audio_data)

segment_length = N // SEGMENTS

extracted_message = ""
for i in range(SEGMENTS):
    start = i * segment_length
    end = (i + 1) * segment_length if i < SEGMENTS - 1 else N
    segment = audio_data[start:end]
    extracted_message += return_char(segment)

print("Extracted message:", extracted_message)