import numpy as np

from src.config import PROCESSED_DIR, SAMPLE_RATE


def load_split(name):
    data = np.load(PROCESSED_DIR / f"{name}.npz")
    return data["x"].astype(np.float32) / 32768, data["y"]


def load_noise():
    data = np.load(PROCESSED_DIR / "noise.npz")
    return {split: data[split].astype(np.float32) / 32768 for split in ["train", "dev", "test"]}


def random_noise(noise, n, rng):
    clip = rng.integers(len(noise), size=n)
    start = rng.integers(noise.shape[1] - SAMPLE_RATE + 1, size=n)
    return noise[clip[:, None], start[:, None] + np.arange(SAMPLE_RATE)]


def mix(speech, noise, snr_db, silence_level_db):
    speech_power = (speech**2).mean(axis=1, keepdims=True)
    noise_power = (noise**2).mean(axis=1, keepdims=True) + 1e-10
    snr_gain = np.sqrt(speech_power / (noise_power * 10 ** (snr_db[:, None] / 10)))
    silence_gain = 10 ** (silence_level_db[:, None] / 20) / np.sqrt(noise_power)
    gain = np.where(speech_power > 1e-8, snr_gain, silence_gain)
    return speech + gain * noise


def time_shift(audio, rng, max_shift=1600):
    shift = rng.integers(-max_shift, max_shift + 1, len(audio))
    index = np.arange(SAMPLE_RATE)[None] - shift[:, None]
    inside = (index >= 0) & (index < SAMPLE_RATE)
    moved = np.take_along_axis(audio, index.clip(0, SAMPLE_RATE - 1), axis=1)
    return np.where(inside, moved, 0.0)


def add_fixed_noise(audio, noise, snr, seed=1234):
    rng = np.random.default_rng(seed)
    n = len(audio)
    silence_level = rng.uniform(-50, -15, n)
    snr_db = rng.uniform(-5, 20, n) if snr == "random" else np.full(n, 0.0 if snr is None else float(snr))
    mixed = mix(audio, random_noise(noise, n, rng), snr_db, silence_level)
    if snr is None:
        no_speech = (audio**2).mean(axis=1) < 1e-8
        mixed = np.where(no_speech[:, None], mixed, audio)
    return mixed.astype("float32")
