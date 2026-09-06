"""
audio_features.py
Genera la representacion de entrada MS-PCEN (Mel-Spectrogram + Per-Channel
Energy Normalization) descrita en la seccion "Input representation" del
paper de Olcay et al. (2026).

Tambien incluye una funcion para generar un Mel-spectrogram simple (sin
PCEN), usada como variante para la ablacion (Seccion 6 del laboratorio).
"""

import numpy as np
import librosa

import config as cfg


def load_audio(path: str, sr: int = cfg.SAMPLE_RATE, duration: float = cfg.CLIP_DURATION_SEC) -> np.ndarray:
    """
    Carga un archivo de audio, lo resamplea a `sr` si hace falta, y lo
    ajusta (pad o truncado) a una duracion fija `duration` en segundos.
    """
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def compute_mel_spectrogram(y: np.ndarray, sr: int = cfg.SAMPLE_RATE) -> np.ndarray:
    """
    Mel-spectrogram (potencia) crudo, sin PCEN. Se usa como input alterno
    en la ablacion.
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        win_length=cfg.N_FFT,
        window="hann",
        n_mels=cfg.N_MELS,
        power=2.0,
    )
    return S.astype(np.float32)


def compute_ms_pcen(y: np.ndarray, sr: int = cfg.SAMPLE_RATE) -> np.ndarray:
    """
    MS-PCEN: Mel-spectrogram (64 bandas) + PCEN con los hiperparametros del
    paper (delta=0.05, alpha=0.98, r=0.5, eps=1.4). librosa.pcen() espera
    magnitud en escala tipo int16, por eso se escala S antes de aplicarlo.
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        win_length=cfg.N_FFT,
        window="hann",
        n_mels=cfg.N_MELS,
        power=1.0,   # PCEN espera magnitud, no potencia
    )

    pcen = librosa.pcen(
        S * (2 ** 31),          # librosa.pcen espera magnitudes tipo int16-scale; escalamos para evitar underflow
        sr=sr,
        hop_length=cfg.HOP_LENGTH,
        gain=cfg.PCEN_ALPHA,
        bias=cfg.PCEN_DELTA,
        power=cfg.PCEN_R,
        time_constant=cfg.PCEN_GAIN_TIME_CONSTANT,
        eps=cfg.PCEN_EPS,
    )
    return pcen.astype(np.float32)


def to_3channel(spec: np.ndarray) -> np.ndarray:
    """
    Replica el espectrograma de 1 canal a 3 canales, tal como el paper
    (para mantener compatibilidad con backbones tipo imagen y con el
    formato usado en el estudio original).
    Retorna un array (3, n_mels, n_frames).
    """
    return np.repeat(spec[np.newaxis, :, :], cfg.N_CHANNELS, axis=0)


def extract_features(path: str, use_pcen: bool = True) -> np.ndarray:
    """
    Pipeline completo: audio -> MS-PCEN (o Mel simple) -> 3 canales.
    Devuelve un array float32 de forma (3, n_mels, n_frames).
    """
    y = load_audio(path)
    spec = compute_ms_pcen(y) if use_pcen else compute_mel_spectrogram(y)
    return to_3channel(spec)
