import librosa
import numpy as np
import soundfile as sf

from src.config import SAMPLE_RATE


def load_audio(path):
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    return audio


def fit_to_one_second(audio):
    if len(audio) >= SAMPLE_RATE:
        start = (len(audio) - SAMPLE_RATE) // 2
        return audio[start:start + SAMPLE_RATE]
    missing = SAMPLE_RATE - len(audio)
    return np.pad(audio, (missing // 2, missing - missing // 2))


def to_int16(audio):
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


def load_clip(path):
    return to_int16(fit_to_one_second(load_audio(path)))
