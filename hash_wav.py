"""
hash_wav.py – Spectral-fingerprint a .wav file and visualise the result.

Usage:
    python hash_wav.py recording.wav
    python hash_wav.py recording.wav -o waveform.png
    python hash_wav.py recording.wav -o waveform.png --spectrogram spectrogram.png
    python hash_wav.py recording.wav --json hash.json --spectrogram spec.png

Replicates the *exact* algorithm from frontend/app.js so hashes match.
Uses spectral-centroid quantization (locality-sensitive).
"""

import sys
import json
import argparse
import wave
import struct
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── Constants (must match app.js) ────────────────────────
CHARSET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "!@#$%^&*()"
)
NUM_BUCKETS  = 64
FFT_SIZE     = 2048
SILENCE_CHAR = "-"
SILENCE_RMS  = 0.001
FREQ_MIN     = 50      # Hz – lower bound for centroid mapping
FREQ_MAX     = 8000    # Hz – upper bound


# ─── WAV reader (pure-stdlib, returns float32 in [-1, 1]) ─
def read_wav(path: str):
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        framerate  = wf.getframerate()
        n_frames   = wf.getnframes()
        raw        = wf.readframes(n_frames)

    if sampwidth == 2:
        fmt = f"<{n_frames * n_channels}h"
        int_samples = struct.unpack(fmt, raw)
        samples = np.array(int_samples, dtype=np.float64) / 32768.0
    elif sampwidth == 1:
        int_samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        samples = (int_samples - 128.0) / 128.0
    elif sampwidth == 4:
        fmt = f"<{n_frames * n_channels}i"
        int_samples = struct.unpack(fmt, raw)
        samples = np.array(int_samples, dtype=np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Downmix to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples.astype(np.float32), framerate


# ─── Per-bucket: spectral centroid → character ───────────
def _hash_bucket(chunk: np.ndarray, sample_rate: int):
    """Compute a locality-sensitive hash character for one time bucket.

    Uses the spectral centroid (power-weighted mean frequency)
    mapped logarithmically onto the CHARSET.  Similar sounds produce
    the same or neighbouring characters.
    """
    n = len(chunk)

    # Silence check
    rms = math.sqrt(float(np.mean(chunk ** 2)))
    if rms < SILENCE_RMS:
        return SILENCE_CHAR, None   # None = no spectrum data

    # Hann window + zero-pad / truncate to FFT_SIZE
    win_len = min(n, FFT_SIZE)
    windowed = np.zeros(FFT_SIZE, dtype=np.float64)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(win_len) / (win_len - 1)))
    windowed[:win_len] = chunk[:win_len] * hann

    # FFT via numpy (matches JS Cooley-Tukey for centroid purposes
    # since centroid is a *smooth* statistic — immune to bin-level
    # rounding differences that plagued djb2)
    spectrum = np.fft.fft(windowed)
    half_n   = FFT_SIZE // 2
    power    = np.abs(spectrum[1:half_n]) ** 2   # skip DC bin 0

    mag_sum = power.sum()
    if mag_sum == 0:
        return SILENCE_CHAR, None

    # Frequency axis in Hz for bins 1 .. halfN-1
    freqs = np.arange(1, half_n) * (sample_rate / FFT_SIZE)

    # Spectral centroid
    centroid = float(np.sum(freqs * power) / mag_sum)

    # Log-scale mapping onto CHARSET
    log_min = math.log(FREQ_MIN)
    log_max = math.log(FREQ_MAX)
    log_c   = math.log(max(FREQ_MIN, min(FREQ_MAX, centroid)))
    t = (log_c - log_min) / (log_max - log_min)          # 0 .. 1
    idx = min(len(CHARSET) - 1, max(0, int(math.floor(t * len(CHARSET)))))

    # Return the char plus the power spectrum for the spectrogram
    return CHARSET[idx], power


# ─── Full spectral hash ──────────────────────────────────
def compute_spectral_hash(samples: np.ndarray, sample_rate: int):
    bucket_len = len(samples) // NUM_BUCKETS
    chars      = []
    boundaries = []
    spectra    = []      # list of power arrays (or None) per bucket

    for i in range(NUM_BUCKETS):
        start = i * bucket_len
        end   = min(start + bucket_len, len(samples))
        chunk = samples[start:end].astype(np.float64)
        boundaries.append((start, end))

        ch, pwr = _hash_bucket(chunk, sample_rate)
        chars.append(ch)
        spectra.append(pwr)
        print(f"\r  Hashing bucket {i+1}/{NUM_BUCKETS}…", end="", flush=True)

    print()
    return "".join(chars), boundaries, spectra


# ─── Visualisation: waveform ──────────────────────────────
def plot_waveform(samples, sample_rate, hash_str, boundaries, out_path=None):
    duration = len(samples) / sample_rate
    t = np.linspace(0, duration, len(samples), endpoint=False)

    fig, ax = plt.subplots(figsize=(18, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0d0d1a")

    ax.plot(t, samples, color="#8be9fd", linewidth=0.4, alpha=0.85)

    colors = ["#50fa7b33", "#ff79c633"]

    for i, ((s, e), ch) in enumerate(zip(boundaries, hash_str)):
        t_start = s / sample_rate
        t_end   = e / sample_rate
        ax.axvspan(t_start, t_end, color=colors[i % 2])
        ax.axvline(t_start, color="#555", linewidth=0.4, linestyle=":")
        t_mid = (t_start + t_end) / 2
        ax.text(
            t_mid, 0.95, ch,
            ha="center", va="top",
            fontsize=7, fontfamily="monospace", fontweight="bold",
            color="#50fa7b" if ch != SILENCE_CHAR else "#666",
        )

    y_top = max(abs(samples.min()), abs(samples.max())) * 1.15 or 0.05
    ax.set_ylim(-y_top, y_top)
    for txt in ax.texts:
        txt.set_y(y_top * 0.92)

    ax.set_xlabel("Time (s)", color="#aaa")
    ax.set_ylabel("Amplitude", color="#aaa")
    title_hash = "".join(c if c != SILENCE_CHAR else "·" for c in hash_str)
    ax.set_title(
        f"Spectral Hash  ({NUM_BUCKETS} buckets)    {title_hash}",
        color="#eee", fontfamily="monospace", fontsize=10,
        usetex=False, math_fontfamily="dejavusans",
    )
    ax.title.set_math_fontfamily("dejavusans")
    ax.title.set_parse_math(False)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Waveform saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ─── Visualisation: spectrogram ───────────────────────────
def plot_spectrogram(samples, sample_rate, hash_str, boundaries, spectra,
                     out_path=None):
    """Draw a frequency-vs-time spectrogram with bucket boundaries and
    hash-character labels.

    Each column of the spectrogram corresponds to one hash bucket.
    The power spectrum is displayed with a log-frequency y-axis and
    dB-scaled colour so quiet harmonics are visible.
    """
    duration = len(samples) / sample_rate
    half_n   = FFT_SIZE // 2
    freq_axis = np.arange(1, half_n) * (sample_rate / FFT_SIZE)   # Hz

    # Build 2-D spectrogram matrix  (freq_bins × NUM_BUCKETS)
    n_freq = half_n - 1   # bins 1..halfN-1
    spec_matrix = np.zeros((n_freq, NUM_BUCKETS))

    for i, pwr in enumerate(spectra):
        if pwr is not None:
            spec_matrix[:, i] = pwr

    # Convert to dB (floor at -80 dB)
    with np.errstate(divide="ignore", invalid="ignore"):
        spec_db = 10.0 * np.log10(spec_matrix + 1e-20)
    vmax = spec_db.max()
    vmin = max(vmax - 80, spec_db.min())

    # Time centres for each bucket (shading='nearest' needs centres, not edges)
    t_centers = np.array([(b[0] + b[1]) / 2.0 / sample_rate for b in boundaries])

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0d0d1a")

    im = ax.pcolormesh(
        t_centers, freq_axis, spec_db,
        shading="nearest",
        cmap="inferno",
        vmin=vmin, vmax=vmax,
    )
    ax.set_yscale("log")
    ax.set_ylim(FREQ_MIN, min(FREQ_MAX, sample_rate / 2))

    # Bucket boundary lines + hash labels
    for i, ((s, e), ch) in enumerate(zip(boundaries, hash_str)):
        t_start = s / sample_rate
        t_end   = e / sample_rate
        ax.axvline(t_start, color="#ffffff44", linewidth=0.5, linestyle=":")
        t_mid = (t_start + t_end) / 2
        txt = ax.text(
            t_mid, FREQ_MAX * 0.85, ch,
            ha="center", va="top",
            fontsize=7, fontfamily="monospace", fontweight="bold",
            color="#50fa7b" if ch != SILENCE_CHAR else "#666",
        )
        txt.set_parse_math(False)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Power (dB)", color="#aaa")
    cbar.ax.yaxis.set_tick_params(color="#888")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#888")

    ax.set_xlabel("Time (s)", color="#aaa")
    ax.set_ylabel("Frequency (Hz)", color="#aaa")
    title_hash = "".join(c if c != SILENCE_CHAR else "·" for c in hash_str)
    ax.set_title(
        f"Spectrogram  ({NUM_BUCKETS} buckets)    {title_hash}",
        color="#eee", fontfamily="monospace", fontsize=10,
        usetex=False, math_fontfamily="dejavusans",
    )
    ax.title.set_math_fontfamily("dejavusans")
    ax.title.set_parse_math(False)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"Spectrogram saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ─── CLI ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Spectral-hash a .wav file and visualise it."
    )
    parser.add_argument("wav", help="Path to the .wav file")
    parser.add_argument("-o", "--output",
                        help="Save waveform figure to this path")
    parser.add_argument("--spectrogram",
                        help="Save spectrogram figure to this path")
    parser.add_argument("--json",
                        help="Also write the hash JSON to this path")
    args = parser.parse_args()

    samples, sr = read_wav(args.wav)
    print(f"Loaded {args.wav}: {len(samples)} samples, {sr} Hz, "
          f"{len(samples)/sr:.2f} s")

    hash_str, boundaries, spectra = compute_spectral_hash(samples, sr)
    print(f"\nSpectral hash ({len(hash_str)} chars):\n{hash_str}\n")

    if args.json:
        payload = {
            "hash":          hash_str,
            "hashLength":    NUM_BUCKETS,
            "sampleRate":    sr,
            "fftSize":       FFT_SIZE,
            "freqMin":       FREQ_MIN,
            "freqMax":       FREQ_MAX,
            "silentBuckets": hash_str.count(SILENCE_CHAR),
            "activeBuckets": NUM_BUCKETS - hash_str.count(SILENCE_CHAR),
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Hash JSON written to {args.json}")

    plot_waveform(samples, sr, hash_str, boundaries, args.output)
    plot_spectrogram(samples, sr, hash_str, boundaries, spectra,
                     args.spectrogram)


if __name__ == "__main__":
    main()
