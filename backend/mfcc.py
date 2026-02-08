import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft
from scipy.spatial.distance import cosine, euclidean
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import os
import glob
import warnings

# Suppress librosa warnings about n_fft size
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


SR = None
N_FFT = 512
HOP = 256
EPS = 1e-8

def load_and_normalize(path):
    y, sr = librosa.load(path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=40)
    y = y.astype(np.float32)
    y /= np.sqrt(np.mean(y**2) + EPS)
    return y, sr


def get_safe_n_fft(y, preferred_n_fft=None):
    """Return an appropriate n_fft size for the signal length.
    Uses preferred_n_fft if provided and signal is long enough,
    otherwise uses largest power of 2 that fits in the signal."""
    n_fft = preferred_n_fft if preferred_n_fft else N_FFT
    if len(y) < n_fft:
        # Use largest power of 2 that fits
        n_fft = 2 ** int(np.log2(len(y)))
        n_fft = max(n_fft, 64)  # minimum 64
    return n_fft


def cosine_sim(a, b):
    return 1.0 - cosine(a, b)


def cepstral_embedding(y, sr, q_low=20, q_high=200):
    n_fft = get_safe_n_fft(y)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=HOP))
    logS = np.log(S + EPS)
    cepstra = irfft(logS, axis=0)

    cep = cepstra[q_low:q_high, :]
    return np.mean(cep, axis=1)


def harmonic_features(y, sr):
    n_fft = get_safe_n_fft(y)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    avg_spec = np.mean(S, axis=1)

    valid = freqs > 50
    log_f = np.log(freqs[valid])
    log_s = np.log(avg_spec[valid] + EPS)

    slope, intercept = np.polyfit(log_f, log_s, 1)

    return np.array([slope, intercept])


def get_pitch_features(y, sr):
    # Use librosa's piptrack to find the fundamental frequency (F0)
    n_fft = get_safe_n_fft(y)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft)
    # Extract the average pitch where magnitude is high
    pitch_val = np.mean(pitches[pitches > 0]) 
    return pitch_val

def get_anatomy_features(y, sr):
    n_fft = get_safe_n_fft(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
    
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft)
    
    # Safe LPC order based on signal length
    lpc_order = min(16, len(y) // 4)
    lpc_order = max(lpc_order, 2)  # at least 2
    try:
        lpc_coeffs = librosa.lpc(y, order=lpc_order)
    except:
        lpc_coeffs = np.zeros(lpc_order + 1)
    
    return {
        "brightness": np.mean(centroid),
        "richness": np.mean(bandwidth),
        "vocal_tract": lpc_coeffs
    }

def get_pitch_stats(y, sr):
    n_fft = get_safe_n_fft(y)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft)
    valid_pitches = pitches[pitches > 0]
    if len(valid_pitches) == 0: return 0, 0
    
    return np.mean(valid_pitches), np.std(valid_pitches)


def resample_to_fixed(feature_2d, n_frames=32):
    """Resample a 2D feature matrix (n_features x time) to fixed number of frames.
    This preserves the temporal shape/trajectory of features."""
    n_feat, n_time = feature_2d.shape
    if n_time < 2:
        return np.tile(feature_2d, (1, n_frames))[:, :n_frames]
    x_orig = np.linspace(0, 1, n_time)
    x_new  = np.linspace(0, 1, n_frames)
    resampled = np.zeros((n_feat, n_frames))
    for i in range(n_feat):
        f = interp1d(x_orig, feature_2d[i], kind='linear')
        resampled[i] = f(x_new)
    return resampled


def get_spectral_flux(y, sr):
    """Frame-to-frame spectral change — captures temporal dynamics of the FFT."""
    n_fft = get_safe_n_fft(y)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=HOP))
    # Diff between consecutive frames
    diff = np.diff(S, axis=1)
    # L2 norm per frame
    flux = np.sqrt(np.sum(diff**2, axis=0))
    return flux


def get_onset_envelope(y, sr, n_frames=32):
    """Onset strength envelope — captures rhythm and timing of speech energy.
    Resampled to fixed length for comparison."""
    # Skip for very short signals
    if len(y) < 512:
        return np.zeros(n_frames)
    
    try:
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
        # Normalize
        onset = onset / (np.max(onset) + EPS)
        # Resample to fixed length
        if len(onset) < 2:
            return np.zeros(n_frames)
        x_orig = np.linspace(0, 1, len(onset))
        x_new  = np.linspace(0, 1, n_frames)
        f = interp1d(x_orig, onset, kind='linear')
        return f(x_new)
    except:
        return np.zeros(n_frames)


def get_jitter_shimmer(y, sr):
    """Jitter (pitch period perturbation) and shimmer (amplitude perturbation).
    These are involuntary micro-variations unique to each person's vocal folds."""
    n_fft = get_safe_n_fft(y)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=HOP)
    # Extract strongest pitch per frame
    pitch_track = []
    mag_track = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        p = pitches[idx, t]
        m = magnitudes[idx, t]
        if p > 50:  # only voiced frames
            pitch_track.append(p)
            mag_track.append(m)

    if len(pitch_track) < 3:
        return 0.0, 0.0

    pitch_track = np.array(pitch_track)
    mag_track = np.array(mag_track)

    # Jitter: mean absolute difference of consecutive periods / mean period
    periods = 1.0 / (pitch_track + EPS)
    period_diffs = np.abs(np.diff(periods))
    jitter = np.mean(period_diffs) / (np.mean(periods) + EPS)

    # Shimmer: mean absolute difference of consecutive amplitudes / mean amplitude
    amp_diffs = np.abs(np.diff(mag_track))
    shimmer = np.mean(amp_diffs) / (np.mean(mag_track) + EPS)

    return jitter, shimmer


def get_hnr(y, sr):
    """Harmonic-to-noise ratio — how clean vs breathy the voice is.
    Structural property of vocal fold closure."""
    # Skip for very short signals
    if len(y) < 512:
        return 0.0
    
    try:
        harmonic = librosa.effects.harmonic(y, margin=8)
        noise = y - harmonic
        h_power = np.mean(harmonic ** 2)
        n_power = np.mean(noise ** 2)
        hnr = 10 * np.log10(h_power / (n_power + EPS) + EPS)
        return hnr
    except:
        return 0.0


def get_spectral_rolloff(y, sr):
    """Frequency below which 85% of spectral energy lives.
    Captures voice darkness/brightness differently than centroid."""
    n_fft = get_safe_n_fft(y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, roll_percent=0.85)
    return np.mean(rolloff)


def get_zcr_stats(y):
    """Zero-crossing rate — distinguishes voiced/unvoiced patterns
    and correlates with speaking style."""
    n_fft = get_safe_n_fft(y)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=HOP)[0]
    return np.array([np.mean(zcr), np.std(zcr)])


def get_spectral_contrast(y, sr):
    """Difference between peaks and valleys in frequency subbands.
    Captures vocal texture that MFCCs smooth over."""
    n_fft = get_safe_n_fft(y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=HOP)
    return np.mean(contrast, axis=1)  # mean across time, per subband


def get_chroma(y, sr):
    """Pitch class distribution — harmonic structure independent of octave.
    Robust to pitch drift between recordings of the same speaker."""
    # Skip chroma for very short signals to avoid hanging
    if len(y) < 512:
        return np.zeros(12)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
        return np.mean(chroma, axis=1)  # 12-dim vector
    except:
        return np.zeros(12)


def build_embeddings(path, debug=False):
    if debug:
        print(f"      [1/16] Loading audio...", flush=True)
    y, sr = load_and_normalize(path)
    
    # Use safe n_fft for short audio
    n_fft = get_safe_n_fft(y)
    
    if debug:
        print(f"      [2/16] Computing MFCCs...", flush=True)
    # ── Static features (time-averaged) ──
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft)
    
    if debug:
        print(f"      [3/16] Anatomy features...", flush=True)
    anatomy = get_anatomy_features(y, sr)
    
    if debug:
        print(f"      [4/16] Pitch stats...", flush=True)
    p_mean, p_std = get_pitch_stats(y, sr)

    if debug:
        print(f"      [5/16] Delta MFCCs...", flush=True)
    # ── Temporal features ──
    # Delta MFCCs: velocity of spectral shape change
    delta_mfcc = librosa.feature.delta(mfccs, order=1)
    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)

    if debug:
        print(f"      [6/16] Delta2 MFCCs...", flush=True)
    # Delta-delta MFCCs: acceleration of spectral change
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)

    if debug:
        print(f"      [7/16] Spectral flux...", flush=True)
    # Spectral flux: frame-to-frame FFT change
    flux = get_spectral_flux(y, sr)
    flux_stats = np.array([np.mean(flux), np.std(flux)])

    if debug:
        print(f"      [8/16] Onset envelope...", flush=True)
    # Onset/rhythm envelope (resampled to fixed 32 frames)
    onset_env = get_onset_envelope(y, sr, n_frames=32)

    if debug:
        print(f"      [9/16] MFCC trajectory...", flush=True)
    # MFCC trajectory: temporal evolution resampled to 32 frames
    mfcc_traj = resample_to_fixed(mfccs, n_frames=32).flatten()

    if debug:
        print(f"      [10/16] Jitter/shimmer...", flush=True)
    # ── New voice biometric features ──
    jitter, shimmer = get_jitter_shimmer(y, sr)
    
    if debug:
        print(f"      [11/16] HNR...", flush=True)
    hnr = get_hnr(y, sr)
    
    if debug:
        print(f"      [12/16] Spectral rolloff...", flush=True)
    rolloff = get_spectral_rolloff(y, sr)
    
    if debug:
        print(f"      [13/16] ZCR stats...", flush=True)
    zcr_stats = get_zcr_stats(y)
    
    if debug:
        print(f"      [14/16] Spectral contrast...", flush=True)
    spec_contrast = get_spectral_contrast(y, sr)
    
    if debug:
        print(f"      [15/16] Chroma...", flush=True)
    chroma = get_chroma(y, sr)
    
    if debug:
        print(f"      [16/16] Done!", flush=True)

    # ── Temporal features ──
    # Delta MFCCs: velocity of spectral shape change
    delta_mfcc = librosa.feature.delta(mfccs, order=1)
    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)

    # Delta-delta MFCCs: acceleration of spectral change
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)

    # Spectral flux: frame-to-frame FFT change
    flux = get_spectral_flux(y, sr)
    flux_stats = np.array([np.mean(flux), np.std(flux)])

    # Onset/rhythm envelope (resampled to fixed 32 frames)
    onset_env = get_onset_envelope(y, sr, n_frames=32)

    # MFCC trajectory: temporal evolution resampled to 32 frames
    mfcc_traj = resample_to_fixed(mfccs, n_frames=32).flatten()

    # ── New voice biometric features ──
    jitter, shimmer = get_jitter_shimmer(y, sr)
    hnr = get_hnr(y, sr)
    rolloff = get_spectral_rolloff(y, sr)
    zcr_stats = get_zcr_stats(y)
    spec_contrast = get_spectral_contrast(y, sr)
    chroma = get_chroma(y, sr)

    """"  "voice_shape":        (... , 0.0075),
        "pitch_avg":          (... , 0.0040),
        "pitch_std":          (... , 0.0036),
        "brightness":         (... , 0.1800),
        "delta_mfcc":         (... , 0.0224),
        "delta2_mfcc":        (... , 0.0266),
        "spectral_flux":      (... , 0.0046),
        "onset_rhythm":       (... , 0.0476),
        "mfcc_trajectory":    (... , 0.2414),
        "jitter":             (... , 0.0037),
        "shimmer":            (... , 0.0535),
        "hnr":                (... , 0.1315),
        "spectral_rolloff":   (... , 0.2540),
        "zcr":                (... , 0.0080),
        "spectral_contrast":  (... , 0.0071),
        "chroma":             (... , 0.0045),"""

    # if pitch_std is > 0.85 between these two 
    return {
        "voice_shape":       (np.mean(mfccs, axis=1), 0.0075),
        "pitch_avg":         (p_mean, 0.0040),
        "pitch_std":         (p_std, 0.0036),
        "brightness":        (anatomy['brightness'], 0.1800),
        "delta_mfcc":        (delta_mfcc_mean, 0.0224),
        "delta2_mfcc":       (delta2_mfcc_mean, 0.0266),
        "spectral_flux":     (flux_stats, 0.0046),
        "onset_rhythm":      (onset_env, 0.0476),
        "mfcc_trajectory":   (mfcc_traj, 0.2414),
        "jitter":            (jitter, 0.0037),
        "shimmer":           (shimmer, 0.0535),
        "hnr":               (hnr, 0.1315),
        "spectral_rolloff":  (rolloff, 0.2540),
        "zcr":               (zcr_stats, 0.0080),
        "spectral_contrast": (spec_contrast, 0.0071),
        "chroma":            (chroma, 0.0045),
    }


def compare_files(path1, path2, debug=False):
    if not os.path.isabs(path1):
        path1 = f"../{path1}.wav"
    else:
        path1 = f"{path1}.wav"
    if not os.path.isabs(path2):
        path2 = f"../{path2}.wav"
    else:
        path2 = f"{path2}.wav"

    if debug:
        print(f"    Loading {os.path.basename(path1)}...", flush=True)
    emb1 = build_embeddings(path1, debug=debug)
    if debug:
        print(f"    Loading {os.path.basename(path2)}...", flush=True)
    emb2 = build_embeddings(path2, debug=debug)

    if debug:
        print(f"    Computing similarity...", flush=True)
    diff = 0.0
    for idx, key in enumerate(emb1.keys()):
        if debug:
            print(f"      Feature {idx+1}/16: {key}", flush=True)
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
            # Check for valid vectors
            if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
                if debug:
                    print(f"        WARNING: NaN in {key}, skipping", flush=True)
                continue
            if np.any(np.isinf(v1)) or np.any(np.isinf(v2)):
                if debug:
                    print(f"        WARNING: Inf in {key}, skipping", flush=True)
                continue
            if len(v1) != len(v2):
                if debug:
                    print(f"        WARNING: Shape mismatch in {key} ({len(v1)} vs {len(v2)}), skipping", flush=True)
                continue
            
            try:
                cos = cosine_sim(v1, v2)
                if np.isnan(cos) or np.isinf(cos):
                    cos = 0.0
                diff += w1 * cos
            except Exception as e:
                if debug:
                    print(f"        ERROR in {key}: {e}", flush=True)
                continue

    if debug:
        print(f"    Similarity computed: {diff:.4f}", flush=True)
    return diff

# compare j1, j2, r1, r2, t1, t2 via a matrix
def compare_all():
    # Discover all WAV files in wavs/ directory
    wavs_dir = os.path.join(os.path.dirname(__file__), '..', 'wavs')
    wav_files = sorted(glob.glob(os.path.join(wavs_dir, '*.wav')))
    
    # Convert to relative paths without .wav extension (for compare_files)
    names = []
    file_paths = []
    for f in wav_files:
        # Get just the filename without extension
        basename = os.path.splitext(os.path.basename(f))[0]
        names.append(f'wavs/{basename}')
        file_paths.append(f"../{names[-1]}.wav")
    
    if not names:
        raise FileNotFoundError(f"No WAV files found in {wavs_dir}")
    
    print(f"Found {len(names)} files to compare")
    
    # PRE-COMPUTE ALL EMBEDDINGS (like train_weights.py does)
    print(f"\nPre-computing embeddings for all {len(names)} files...")
    embeddings_cache = {}
    for i, (name, path) in enumerate(zip(names, file_paths)):
        print(f"  [{i+1}/{len(names)}] {os.path.basename(name)}", flush=True)
        try:
            embeddings_cache[name] = build_embeddings(path, debug=False)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            embeddings_cache[name] = None
    
    print(f"\nComputing similarity matrix...")
    matrix = np.zeros((len(names), len(names)))

    total_comparisons = len(names) * len(names)
    current = 0
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            current += 1
            
            if current % 100 == 0 or current == 1:
                print(f"Progress: {current}/{total_comparisons} ({100*current/total_comparisons:.1f}%)", flush=True)
            
            try:
                # Use cached embeddings instead of recomputing
                emb1 = embeddings_cache.get(name1)
                emb2 = embeddings_cache.get(name2)
                
                if emb1 is None or emb2 is None:
                    matrix[i, j] = 0.0
                    continue
                
                # Compute similarity directly
                diff = 0.0
                for key in emb1.keys():
                    v1, w1 = emb1[key]
                    v2, w2 = emb2[key]
                    
                    if np.isscalar(v1) or (isinstance(v1, np.ndarray) and v1.ndim == 0):
                        pitch_diff = abs(v1 - v2)
                        max_pitch = max(abs(v1), abs(v2), 1.0)
                        similarity = max(0.0, 1.0 - (pitch_diff / max_pitch))
                        diff += w1 * similarity
                    else:
                        if np.any(np.isnan(v1)) or np.any(np.isnan(v2)) or np.any(np.isinf(v1)) or np.any(np.isinf(v2)):
                            continue
                        if len(v1) != len(v2):
                            continue
                        try:
                            cos = cosine_sim(v1, v2)
                            if not (np.isnan(cos) or np.isinf(cos)):
                                diff += w1 * cos
                        except:
                            continue
                
                matrix[i, j] = diff
                
            except Exception as e:
                print(f"\n  ERROR in comparison {current}: {e}", flush=True)
                matrix[i, j] = 0.0

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
    
    # Find pairs of similar recordings (high similarity, not diagonal)
    print("\nTop similar pairs (excluding self-comparison):")
    similarities = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            similarities.append((names[i], names[j], matrix[i, j]))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Show top 5
    for idx, (name1, name2, sim) in enumerate(similarities[:5], 1):
        print(f"  {idx}. {name1} vs {name2}: {sim:.4f}")
    
    print("\nLeast similar pairs:")
    for idx, (name1, name2, sim) in enumerate(similarities[-5:], 1):
        print(f"  {idx}. {name1} vs {name2}: {sim:.4f}")
    
    # Visualize with matplotlib
    plt.figure(figsize=(12, 10))
    im = plt.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, label='Similarity Score')
    
    # Set ticks and labels
    plt.xticks(range(len(names)), names, fontsize=9, rotation=45, ha='right')
    plt.yticks(range(len(names)), names, fontsize=9)
    
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

