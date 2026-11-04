import queue
import sys
import time
from collections import deque

import numpy as np
import tensorflow as tf

from src.config import CLASSES, SAMPLE_RATE, SILENCE, UNKNOWN

TARGET_LEVEL_DB = -20.0
MAX_GAIN_DB = 30.0
NO_PERMISSION = ("The microphone only returns zeros. Allow microphone access for your terminal in "
                 "System Settings > Privacy & Security > Microphone and restart the terminal.")


def active_level_db(audio, frame=320):
    n = len(audio) // frame * frame
    energy = np.sort((audio[:n].reshape(-1, frame) ** 2).mean(axis=1))
    return float(10 * np.log10(energy[len(energy) // 2:].mean() + 1e-12))


def normalize(audio):
    gain_db = min(TARGET_LEVEL_DB - active_level_db(audio), MAX_GAIN_DB)
    audio = audio * 10 ** (gain_db / 20)
    peak = np.abs(audio).max()
    return audio / peak * 0.99 if peak > 0.99 else audio


class Detector:
    def __init__(self, model, threshold=0.6, smoothing=3, gate_db=-50.0, cooldown=1.0):
        self.predict = tf.function(lambda audio: model(audio, training=False)["label"])
        self.threshold, self.gate_db, self.cooldown = threshold, gate_db, cooldown
        self.window = np.zeros(SAMPLE_RATE, np.float32)
        self.history = deque(maxlen=smoothing)
        self.last, self.last_time = None, -1e9
        self.level, self.state = -120.0, ("silence", 0.0)

    def probabilities(self, audio):
        return self.predict(normalize(audio)[None].astype("float32"))[0].numpy()

    def process(self, block, now):
        self.window = np.concatenate([self.window[len(block):], block])
        self.level = active_level_db(self.window)
        if self.level < self.gate_db:
            self.history.clear()
            self.state = ("silence", 0.0)
            return None

        self.history.append(self.probabilities(self.window))
        p = np.mean(self.history, axis=0)
        p[SILENCE] = 0
        p /= p.sum()
        k = int(p.argmax())
        self.state = (CLASSES[k], float(p[k]))
        repeated = k == self.last and now - self.last_time < self.cooldown
        if k != UNKNOWN and p[k] >= self.threshold and not repeated:
            self.last, self.last_time = k, now
            self.history.clear()
            return CLASSES[k], float(p[k]), p
        return None


def listen(model, device=None, threshold=0.6, hop=0.1, gate_db=-50.0, smoothing=3):
    import sounddevice as sd

    blocks = queue.Queue()
    detector = Detector(model, threshold, smoothing, gate_db)
    heard_signal, start = False, time.time()

    def callback(indata, frames, t, status):
        blocks.put(indata[:, 0].copy())

    words = [c for c in CLASSES if c not in ("unknown", "silence")]
    print(f"Listening (threshold {threshold:.2f}). Say: {', '.join(words)}")
    print("Ctrl+C to stop.\n")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device,
                            blocksize=int(hop * SAMPLE_RATE), callback=callback)
    with stream:
        try:
            while True:
                block = blocks.get()
                if np.abs(block).max() > 0:
                    heard_signal = True
                elif not heard_signal and time.time() - start > 3:
                    sys.exit("\n" + NO_PERMISSION)

                result = detector.process(block, time.time())
                bar = "#" * int(np.clip((detector.level + 70) / 3, 0, 20))
                word, confidence = detector.state
                sys.stdout.write(f"\r  level {detector.level:6.1f} dB {bar:<20} {word:<12} {confidence:.2f}   ")
                sys.stdout.flush()
                if result:
                    word, confidence, p = result
                    others = "  ".join(f"{CLASSES[j]} {p[j]:.2f}" for j in p.argsort()[::-1][1:3])
                    print(f"\r> {word.upper():<12} {confidence:.2f}   [{others}]" + " " * 20)
        except KeyboardInterrupt:
            print("\nDone.")
