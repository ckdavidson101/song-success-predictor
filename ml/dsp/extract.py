"""
DSP feature extraction for audio analysis.

Provides baseline librosa-based feature extraction including tempo, onset rate,
spectral features, MFCCs, chroma, and harmonic-percussive separation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np


def load_mono_30s(audio_path: str | Path, sr: int = 44100) -> tuple[np.ndarray, int]:
    """
    Load audio file, convert to mono, resample to target SR, and trim/pad to exactly 30.0s.

    Args:
        audio_path: Path to audio file
        sr: Target sample rate (default 44100 Hz)

    Returns:
        Tuple of (audio_data, sample_rate)

    Raises:
        Exception: If audio loading fails with audio_path in error message
    """
    try:
        # Load audio with librosa (handles MP3 via audioread/ffmpeg)
        y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)

        # Calculate target length (30 seconds)
        target_length = int(30.0 * sr)

        # Trim or pad to exactly 30 seconds
        if len(y) > target_length:
            # Trim from center
            start = (len(y) - target_length) // 2
            y = y[start : start + target_length]
        elif len(y) < target_length:
            # Pad with zeros (centered)
            pad_length = target_length - len(y)
            pad_before = pad_length // 2
            pad_after = pad_length - pad_before
            y = np.pad(y, (pad_before, pad_after), mode="constant", constant_values=0)

        return y, sr

    except Exception as e:
        raise Exception(f"Failed to load audio from {audio_path}: {str(e)}") from e


def extract_features(audio_path: str | Path) -> dict[str, Any]:
    """
    Extract comprehensive audio features using librosa.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with flattened feature names and values

    Raises:
        Exception: If feature extraction fails with audio_path in error message
    """
    try:
        # Load audio
        y, sr = load_mono_30s(audio_path, sr=44100)
        duration_sec = len(y) / sr

        features = {"duration_sec": duration_sec, "sr": sr}

        # Tempo estimation
        tempo, _ = librosa.beat.tempo(y=y, sr=sr)
        features["tempo_librosa"] = float(tempo[0]) if len(tempo) > 0 else 0.0

        # Onset detection for onset rate
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        features["onset_rate"] = len(onset_times) / duration_sec

        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))
        features["rms_p10"] = float(np.percentile(rms, 10))
        features["rms_p90"] = float(np.percentile(rms, 90))

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spec_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spec_centroid_std"] = float(np.std(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spec_rolloff_mean"] = float(np.mean(spectral_rolloff))
        features["spec_rolloff_std"] = float(np.std(spectral_rolloff))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features["spec_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        features["spec_bandwidth_std"] = float(np.std(spectral_bandwidth))

        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features["spec_flatness_mean"] = float(np.mean(spectral_flatness))
        features["spec_flatness_std"] = float(np.std(spectral_flatness))

        # Spectral flux (difference between consecutive frames)
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        spectral_flux = np.mean(np.diff(magnitude, axis=1) ** 2, axis=0)
        features["spec_flux_mean"] = float(np.mean(spectral_flux))
        features["spec_flux_std"] = float(np.std(spectral_flux))

        # Harmonic-Percussive Separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Calculate energy ratio
        energy_harmonic = np.sum(y_harmonic**2)
        energy_percussive = np.sum(y_percussive**2)
        features["hpss_ratio"] = float(energy_harmonic / (energy_percussive + 1e-9))

        # MFCCs with deltas
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        # MFCC features (mean/std for each coefficient)
        for i in range(20):
            features[f"mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc{i+1}_std"] = float(np.std(mfccs[i]))
            features[f"mfcc{i+1}_delta_mean"] = float(np.mean(mfcc_delta[i]))
            features[f"mfcc{i+1}_delta_std"] = float(np.std(mfcc_delta[i]))
            features[f"mfcc{i+1}_delta2_mean"] = float(np.mean(mfcc_delta2[i]))
            features[f"mfcc{i+1}_delta2_std"] = float(np.std(mfcc_delta2[i]))

        # Chroma CQT features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_names = [
            "c",
            "c_sharp",
            "d",
            "d_sharp",
            "e",
            "f",
            "f_sharp",
            "g",
            "g_sharp",
            "a",
            "a_sharp",
            "b",
        ]

        for i, note in enumerate(chroma_names):
            features[f"chroma_{note}_mean"] = float(np.mean(chroma[i]))
            features[f"chroma_{note}_std"] = float(np.std(chroma[i]))

        return features

    except Exception as e:
        raise Exception(
            f"Failed to extract features from {audio_path}: {str(e)}"
        ) from e
