import numpy as np

from src.config import PROCESSED_DIR, SAMPLE_RATE


def load_split(name):
    data = np.load(PROCESSED_DIR / f"{name}.npz")
    return data["x"].astype(np.float32) / 32768, data["y"]


def load_noise():
    data = np.load(PROCESSED_DIR / "noise.npz")
    return {split: data[split].astype(np.float32) / 32768 for split in ["train", "dev", "test"]}


def loud_windows(audio, threshold_db=-60, hop=SAMPLE_RATE // 10):
    energy = np.cumsum(audio**2, axis=1)
    energy = np.concatenate([np.zeros((len(audio), 1), np.float32), energy], axis=1)
    starts = np.arange(0, audio.shape[1] - SAMPLE_RATE + 1, hop)
    level_db = 10 * np.log10((energy[:, starts + SAMPLE_RATE] - energy[:, starts]) / SAMPLE_RATE + 1e-12)
    clip, k = np.nonzero(level_db > threshold_db)
    return np.stack([clip, starts[k]], axis=1)


class NoiseBank:
    def __init__(self, audio):
        self.audio = audio
        self.windows = loud_windows(audio)

    def sample(self, n, rng):
        clip, start = self.windows[rng.integers(len(self.windows), size=n)].T
        return self.audio[clip[:, None], start[:, None] + np.arange(SAMPLE_RATE)]


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


def add_fixed_noise(audio, bank, snr, seed=1234):
    rng = np.random.default_rng(seed)
    n = len(audio)
    noise = bank.sample(n, rng)
    silence_level = rng.uniform(-50, -15, n)
    if snr == "random":
        snr_db = rng.uniform(-5, 20, n)
    else:
        snr_db = np.full(n, 0.0 if snr is None else float(snr))
    mixed = mix(audio, noise, snr_db, silence_level)
    if snr is None:
        no_speech = (audio**2).mean(axis=1) < 1e-8
        mixed = np.where(no_speech[:, None], mixed, audio)
    return mixed.astype("float32")
