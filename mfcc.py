import librosa #type: ignore
import numpy as np
import matplotlib.pyplot as plt
import librosa.display #type: ignore
from scipy.fft import rfft

# Configuration for spectral hash
CHARSET = (
    'abcdefghijklmnopqrstuvwxyz' +
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ' +
    '0123456789' +
    '!@#$%^&*()'
)

NUM_BUCKETS = 64
FFT_SIZE = 2048
SILENCE_CHAR = '-'
SILENCE_RMS = 0.01
FREQ_MIN = 50
FREQ_MAX = 8000


def compute_spectral_hash(y, sr):
    """
    Compute spectral hash from audio samples.
    
    Args:
        y: Audio time series (numpy array)
        sr: Sample rate
    
    Returns:
        dict: Contains 'hash', 'hashLength', 'sampleRate', 'fftSize', 'silentBuckets', 'activeBuckets'
    """
    samples = y.astype(np.float32)
    
    # Compute hash
    bucket_len = len(samples) // NUM_BUCKETS
    log_min = np.log(FREQ_MIN)
    log_max = np.log(FREQ_MAX)
    hash_result = ''
    
    for i in range(NUM_BUCKETS):
        start = i * bucket_len
        end = min(start + bucket_len, len(samples))
        chunk = samples[start:end]
        
        # Silence check (RMS)
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < SILENCE_RMS:
            hash_result += SILENCE_CHAR
            continue
        
        # Prepare FFT input: Apply Hann window
        win_len = min(len(chunk), FFT_SIZE)
        windowed = np.zeros(FFT_SIZE, dtype=np.float64)
        
        # Apply Hann window
        hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(win_len) / (win_len - 1)))
        windowed[:win_len] = chunk[:win_len] * hann_window
        
        # Compute FFT
        fft_result = rfft(windowed)
        
        # Compute magnitudes squared (power spectrum)
        magnitudes_sq = np.abs(fft_result) ** 2
        
        # Compute spectral centroid
        half_n = FFT_SIZE // 2
        weighted_sum = 0
        mag_sum = 0
        
        for j in range(1, half_n):
            mag = magnitudes_sq[j]
            freq_hz = j * (sr / FFT_SIZE)
            weighted_sum += freq_hz * mag
            mag_sum += mag
        
        if mag_sum == 0:
            hash_result += SILENCE_CHAR
            continue
        
        # Spectral centroid in Hz
        centroid = weighted_sum / mag_sum
        
        # Map centroid to character (log scale)
        log_centroid = np.log(np.clip(centroid, FREQ_MIN, FREQ_MAX))
        t = (log_centroid - log_min) / (log_max - log_min)
        idx = int(np.clip(t * len(CHARSET), 0, len(CHARSET) - 1))
        hash_result += CHARSET[idx]
    
    # Count silent and active buckets
    silent_count = hash_result.count(SILENCE_CHAR)
    active_count = NUM_BUCKETS - silent_count
    
    return {
        'hash': hash_result,
        'hashLength': len(hash_result),
        'sampleRate': int(sr),
        'fftSize': FFT_SIZE,
        'silentBuckets': silent_count,
        'activeBuckets': active_count
    }


file_path1 = 'charlie1.wav'
file_path2 = 'charlie2.wav' 
y1, sr = librosa.load(file_path1, sr=None) 
y2, sr = librosa.load(file_path2, sr=None) 

# MFCC Analysis
mfccs1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=20)
mfccs2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=20)

mean_mfccs1 = np.mean(mfccs1.T, axis=0)
mean_mfccs2 = np.mean(mfccs2.T, axis=0)
print("Mean MFCCs for Charlie 1:", mean_mfccs1)
print("Mean MFCCs for Charlie 2:", mean_mfccs2)

# Spectral Hash Analysis
print("\n" + "="*60)
spectral_hash1 = compute_spectral_hash(y1, sr)
spectral_hash2 = compute_spectral_hash(y2, sr)

print("Spectral Hash for Charlie 1:")
print(f"  Hash: {spectral_hash1['hash']}")
print(f"  Hash Length: {spectral_hash1['hashLength']}")
print(f"  Sample Rate: {spectral_hash1['sampleRate']} Hz")
print(f"  Silent Buckets: {spectral_hash1['silentBuckets']}")
print(f"  Active Buckets: {spectral_hash1['activeBuckets']}")

print("\nSpectral Hash for Charlie 2:")
print(f"  Hash: {spectral_hash2['hash']}")
print(f"  Hash Length: {spectral_hash2['hashLength']}")
print(f"  Sample Rate: {spectral_hash2['sampleRate']} Hz")
print(f"  Silent Buckets: {spectral_hash2['silentBuckets']}")
print(f"  Active Buckets: {spectral_hash2['activeBuckets']}")
