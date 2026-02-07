import librosa #type: ignore
import numpy as np
import matplotlib.pyplot as plt
import librosa.display #type: ignore
from scipy.fft import rfft

# Configuration for spectral hash
CHARSET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
FIXED_BUCKETS = 20
FFT_SIZE = 2048
SILENCE_CHAR = '-'
SILENCE_THRESHOLD = 0.02
FREQ_MIN = 50
FREQ_MAX = 8000


def trim_audio(y, sr, top_db=30):
    """
    Trim silence from the beginning and end of audio.
    
    Args:
        y: Audio time series (numpy array)
        sr: Sample rate
        top_db: Threshold in dB below reference for silence detection
    
    Returns:
        Trimmed audio array
    """
    # Use librosa's trim function
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def compute_spectral_hash(y, sr):
    """
    Compute robust spectral hash from trimmed audio.
    Uses fixed bucket count for consistency.
    
    Args:
        y: Audio time series (numpy array)
        sr: Sample rate
    
    Returns:
        dict: Contains 'hash', 'hashLength', 'sampleRate', 'fftSize', 'silentBuckets', 'activeBuckets'
    """
    # Trim silence from start and end
    y_trimmed = trim_audio(y, sr)
    samples = y_trimmed.astype(np.float32)
    
    # Normalize by RMS to handle volume differences
    rms_global = np.sqrt(np.mean(samples ** 2))
    if rms_global > 0:
        samples = samples / rms_global
    
    # Create fixed-size buckets
    bucket_len = len(samples) // FIXED_BUCKETS
    if bucket_len == 0:
        return {
            'hash': SILENCE_CHAR * FIXED_BUCKETS,
            'hashLength': FIXED_BUCKETS,
            'sampleRate': int(sr),
            'fftSize': FFT_SIZE,
            'silentBuckets': FIXED_BUCKETS,
            'activeBuckets': 0,
            'trimmedLength': len(y_trimmed)
        }
    
    log_min = np.log(FREQ_MIN)
    log_max = np.log(FREQ_MAX)
    hash_result = ''
    
    for i in range(FIXED_BUCKETS):
        start = i * bucket_len
        end = min(start + bucket_len, len(samples))
        chunk = samples[start:end]
        
        if len(chunk) == 0:
            hash_result += SILENCE_CHAR
            continue
        
        # Silence check (RMS)
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < SILENCE_THRESHOLD:
            hash_result += SILENCE_CHAR
            continue
        
        # Split chunk into frames and compute median centroid
        num_frames = max(1, len(chunk) // 512)
        centroids = []
        
        for frame_idx in range(num_frames):
            frame_start = frame_idx * len(chunk) // num_frames
            frame_end = (frame_idx + 1) * len(chunk) // num_frames
            frame = chunk[frame_start:frame_end]
            
            if len(frame) < 2:
                continue
            
            # Prepare FFT input with Hann window
            win_len = min(len(frame), FFT_SIZE)
            windowed = np.zeros(FFT_SIZE, dtype=np.float64)
            
            hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(win_len) / (win_len - 1)))
            windowed[:win_len] = frame[:win_len] * hann_window
            
            # Compute FFT
            fft_result = rfft(windowed)
            magnitudes_sq = np.abs(fft_result) ** 2
            
            # Compute spectral centroid
            half_n = FFT_SIZE // 2
            weighted_sum = 0.0
            mag_sum = 0.0
            
            for j in range(1, half_n):
                mag = magnitudes_sq[j]
                freq_hz = j * (sr / FFT_SIZE)
                weighted_sum += freq_hz * mag
                mag_sum += mag
            
            if mag_sum > 0:
                centroid = weighted_sum / mag_sum
                centroids.append(centroid)
        
        if not centroids:
            hash_result += SILENCE_CHAR
            continue
        
        # Use median centroid for robustness
        centroid = np.median(centroids)
        
        # Map centroid to character (log scale)
        log_centroid = np.log(np.clip(centroid, FREQ_MIN, FREQ_MAX))
        t = (log_centroid - log_min) / (log_max - log_min)
        idx = int(np.clip(t * len(CHARSET), 0, len(CHARSET) - 1))
        hash_result += CHARSET[idx]
    
    # Count silent and active buckets
    silent_count = hash_result.count(SILENCE_CHAR)
    active_count = FIXED_BUCKETS - silent_count
    
    return {
        'hash': hash_result,
        'hashLength': len(hash_result),
        'sampleRate': int(sr),
        'fftSize': FFT_SIZE,
        'silentBuckets': silent_count,
        'activeBuckets': active_count,
        'trimmedLength': len(y_trimmed)
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
print(f"  Audio after trimming: {spectral_hash1['trimmedLength']} samples")
print(f"  Sample Rate: {spectral_hash1['sampleRate']} Hz")
print(f"  Silent Buckets: {spectral_hash1['silentBuckets']}")
print(f"  Active Buckets: {spectral_hash1['activeBuckets']}")

print("\nSpectral Hash for Charlie 2:")
print(f"  Hash: {spectral_hash2['hash']}")
print(f"  Hash Length: {spectral_hash2['hashLength']}")
print(f"  Audio after trimming: {spectral_hash2['trimmedLength']} samples")
print(f"  Sample Rate: {spectral_hash2['sampleRate']} Hz")
print(f"  Silent Buckets: {spectral_hash2['silentBuckets']}")
print(f"  Active Buckets: {spectral_hash2['activeBuckets']}")

# Compare hashes
print("\n" + "="*60)
print("Hash Similarity Analysis:")
hash1 = spectral_hash1['hash']
hash2 = spectral_hash2['hash']
matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
similarity = (matches / len(hash1)) * 100 if len(hash1) > 0 else 0
print(f"  Matching positions: {matches}/{len(hash1)} ({similarity:.1f}%)")

# Hamming distance
hamming = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
print(f"  Hamming distance: {hamming} characters")
