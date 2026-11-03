import keras
import tensorflow as tf
from keras import layers

from src.config import N_CLASSES, SAMPLE_RATE
from src.features import LogMel


def build_model(with_decoder, logmel=None):
    logmel = logmel or LogMel(name="logmel")
    audio = keras.Input((SAMPLE_RATE,), name="audio")
    x = logmel(audio)

    for block, channels in enumerate([32, 64, 128], start=1):
        for conv in (1, 2):
            x = layers.Conv2D(channels, 3, padding="same", use_bias=False, name=f"block{block}_conv{conv}")(x)
            x = layers.BatchNormalization(name=f"block{block}_bn{conv}")(x)
            x = layers.ReLU(name=f"block{block}_relu{conv}")(x)
        x = layers.MaxPooling2D(2, name=f"block{block}_pool")(x)
    latent = x

    c = layers.GlobalAveragePooling2D(name="avg_pool")(latent)
    c = layers.Dropout(0.3, name="dropout")(c)
    outputs = {"label": layers.Dense(N_CLASSES, activation="softmax", name="label")(c)}

    if with_decoder:
        d = latent
        for step, channels in enumerate([64, 32, 16], start=1):
            d = layers.Conv2DTranspose(channels, 3, strides=2, padding="same", name=f"decoder_convt{step}")(d)
            d = layers.BatchNormalization(name=f"decoder_bn{step}")(d)
            d = layers.ReLU(name=f"decoder_relu{step}")(d)
        d = layers.Resizing(40, 101, name="decoder_resize")(d)
        outputs["reconstruction"] = layers.Conv2D(1, 3, padding="same", name="reconstruction")(d)

    return keras.Model(audio, outputs)


class ReconstructionLoss(keras.losses.Loss):
    def __init__(self, logmel, **kwargs):
        super().__init__(**kwargs)
        self.logmel = logmel

    def call(self, clean_audio, reconstruction):
        return tf.reduce_mean(tf.square(self.logmel(clean_audio) - reconstruction), axis=[1, 2, 3])
