import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft
from scipy.spatial.distance import cosine, euclidean

SR = None
N_FFT = 2048
HOP = 256
EPS = 1e-8

def load_and_normalize(path):
    y, sr = librosa.load(path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y.astype(np.float32)
    y /= np.sqrt(np.mean(y**2) + EPS)
    return y, sr


def cosine_sim(a, b):
    return 1.0 - cosine(a, b)


def cepstral_embedding(y, sr, q_low=20, q_high=200):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    logS = np.log(S + EPS)
    cepstra = irfft(logS, axis=0)

    cep = cepstra[q_low:q_high, :]
    return np.mean(cep, axis=1)


def harmonic_features(y, sr):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    avg_spec = np.mean(S, axis=1)

    valid = freqs > 50
    log_f = np.log(freqs[valid])
    log_s = np.log(avg_spec[valid] + EPS)

    slope, intercept = np.polyfit(log_f, log_s, 1)

    return np.array([slope, intercept])


def get_pitch_features(y, sr):
    # Use librosa's piptrack to find the fundamental frequency (F0)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    # Extract the average pitch where magnitude is high
    pitch_val = np.mean(pitches[pitches > 0]) 
    return pitch_val

def build_embeddings(path):
    y, sr = load_and_normalize(path)

    cep = cepstral_embedding(y, sr)
    harm = harmonic_features(y, sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=23)
    mfcc_mean = np.mean(mfccs, axis=1) 
    pitch = get_pitch_features(y, sr)

    return {
        "cepstrum": (cep, 0.00),
        "harmonics": (harm, 0.00),
        "voice_shape": (mfcc_mean, 0.0),
        "pitch": (pitch, 1.0)
    }


def compare_files(path1, path2):
    path1 = f"recording_{path1}.wav"
    path2 = f"recording_{path2}.wav"

    emb1 = build_embeddings(path1)
    emb2 = build_embeddings(path2)

    diff = 0.0
    for key in emb1.keys():
        v1, w1 = emb1[key]
        v2, w2 = emb2[key]

        # Handle scalar values (like pitch) differently from vectors
        if np.isscalar(v1) or (isinstance(v1, np.ndarray) and v1.ndim == 0):
            # For scalar values, use normalized absolute difference
            pitch_diff = abs(v1 - v2)
            max_pitch = max(abs(v1), abs(v2), 1.0)  # avoid division by zero
            similarity = 1.0 - (pitch_diff / max_pitch)
            similarity = max(0.0, similarity)  # ensure non-negative
            diff += w1 * similarity
        else:
            # For vector values, use cosine similarity
            cos = cosine_sim(v1, v2)
            diff += w1 * cos

    return diff

# compare j1, j2, r1, r2, t1, t2 via a matrix
def compare_all():
    names = ['j1', 'j2', 'r1', 'r2', 't1', 't2']
    matrix = np.zeros((len(names), len(names)))

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            matrix[i, j] = compare_files(name1, name2)

    return names, matrix


# Display the comparison matrix
if __name__ == "__main__":
    print("Computing speaker similarity matrix...")
    print("=" * 60)
    
    names, matrix = compare_all()
    
    # Print header
    print("\nSimilarity Matrix (higher = more similar):")
    print("\n     ", end="")
    for name in names:
        print(f"{name:>8}", end="")
    print()
    print("     " + "-" * (8 * len(names)))
    
    # Print matrix rows
    for i, name in enumerate(names):
        print(f"{name:>4} |", end="")
        for j in range(len(names)):
            print(f"{matrix[i, j]:>8.4f}", end="")
        print()
    
    print("\n" + "=" * 60)
    print("Analysis:")
    
    # Within-speaker comparisons
    print("\nWithin-speaker similarities:")
    print(f"  j1 vs j2: {matrix[0, 1]:.4f}")
    print(f"  r1 vs r2: {matrix[2, 3]:.4f}")
    print(f"  t1 vs t2: {matrix[4, 5]:.4f}")
    
    # Cross-speaker comparisons (sample)
    print("\nCross-speaker similarities (examples):")
    print(f"  j1 vs r1: {matrix[0, 2]:.4f}")
    print(f"  j1 vs t1: {matrix[0, 4]:.4f}")
    print(f"  r1 vs t1: {matrix[2, 4]:.4f}")
    
    # Visualize with matplotlib
    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, label='Similarity Score')
    
    # Set ticks and labels
    plt.xticks(range(len(names)), names, fontsize=12)
    plt.yticks(range(len(names)), names, fontsize=12)
    
    # Add values in cells
    for i in range(len(names)):
        for j in range(len(names)):
            text_color = 'white' if matrix[i, j] < 0.5 else 'black'
            plt.text(j, i, f'{matrix[i, j]:.3f}', 
                    ha='center', va='center', 
                    color=text_color, fontsize=10, fontweight='bold')
    
    plt.title('Speaker Similarity Matrix\n(Green = More Similar, Red = Less Similar)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Speaker', fontsize=12, fontweight='bold')
    plt.ylabel('Speaker', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Visualization displayed!")

