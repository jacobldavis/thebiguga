from scipy.io import wavfile
from scipy.fft import irfft, rfft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

sample_rate, audio_data = wavfile.read('ex.wav')

if audio_data.ndim == 2:
    audio_data = audio_data[:, 0]

print(f"Sample rate: {sample_rate} Hz")
print(f"Audio data shape: {audio_data.shape}")

N = len(audio_data)

fft_vals = rfft(audio_data)

# fft_vals[2000:] = 0

reconstructed = irfft(fft_vals, n=N)

wavfile.write('labubu.wav', sample_rate, reconstructed.astype(audio_data.dtype))
plt.plot( fftfreq(N, 1 / sample_rate)[:N // 2], np.abs(fft_vals)[:-1])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum')
plt.show()
