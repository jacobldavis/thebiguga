from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import numpy as np

sample_rate, audio_data = wavfile.read('a.wav')
wavfile.write('labubu.wav', sample_rate, audio_data)

print(f"Sample rate: {sample_rate} Hz")
print(f"Audio data shape: {audio_data.shape}")

transform = fft(audio_data)
xf = fftfreq(len(audio_data), 1 / sample_rate)

print(transform.shape)
print(xf.shape)
