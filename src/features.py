import numpy as np
import tensorflow as tf
from keras import layers

from src.config import SAMPLE_RATE

N_FFT = 480
HOP = 160
N_MELS = 40


class LogMel(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mel_filters = tf.signal.linear_to_mel_weight_matrix(N_MELS, N_FFT // 2 + 1, SAMPLE_RATE, 20.0, 7600.0)
        self.mean = self.add_weight(shape=(N_MELS,), initializer="zeros", trainable=False, name="mean")
        self.std = self.add_weight(shape=(N_MELS,), initializer="ones", trainable=False, name="std")

    def unnormalized(self, audio):
        audio = tf.pad(audio, [[0, 0], [N_FFT // 2, N_FFT // 2]], mode="REFLECT")
        stft = tf.signal.stft(audio, frame_length=N_FFT, frame_step=HOP, fft_length=N_FFT)
        power = tf.square(tf.abs(stft))
        return tf.math.log(tf.tensordot(power, self.mel_filters, 1) + 1e-6)

    def call(self, audio):
        logmel = (self.unnormalized(audio) - self.mean) / self.std
        return tf.transpose(logmel, [0, 2, 1])[..., None]

    def adapt(self, speech, batch_size=1000):
        values = np.concatenate([
            self.unnormalized(speech[i:i + batch_size]).numpy() for i in range(0, len(speech), batch_size)
        ])
        self.mean.assign(values.mean(axis=(0, 1)))
        self.std.assign(values.std(axis=(0, 1)))
