import math
from scipy.io import wavfile
from scipy.fft import irfft, rfft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:',.<>?/`~"

def return_char(data):
    fft_vals = rfft(data)
    value = int(math.sqrt(sum(np.abs(np.log10(f))*np.abs(np.log10(f)) for f in fft_vals)))
    return characters[value % len(characters)]

def plot_spectrum(data, sample_rate):
    fft_vals = rfft(data)
    N = len(data)
    plt.plot( fftfreq(N, 1 / sample_rate)[:N // 2], np.abs(fft_vals)[:-1])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum')
    plt.show()

def write_reconstructed(data, sample_rate, filename="labubu.wav"):
    fft_vals = rfft(data)
    N = len(data)
    reconstructed = irfft(fft_vals, n=N)
    wavfile.write(filename, sample_rate, reconstructed.astype(data.dtype))