import keras
import tensorflow as tf
from keras import layers

from src.config import N_CLASSES, SAMPLE_RATE
from src.features import LogMel

ENCODER_LAYERS = ("logmel", "block")


def encoder(logmel=None):
    audio = keras.Input((SAMPLE_RATE,), name="audio")
    x = (logmel or LogMel(name="logmel"))(audio)
    for block, channels in enumerate([32, 64, 128], start=1):
        for conv in (1, 2):
            x = layers.Conv2D(channels, 3, padding="same", use_bias=False, name=f"block{block}_conv{conv}")(x)
            x = layers.BatchNormalization(name=f"block{block}_bn{conv}")(x)
            x = layers.ReLU(name=f"block{block}_relu{conv}")(x)
        x = layers.MaxPooling2D(2, name=f"block{block}_pool")(x)
    return audio, x


def classifier(latent):
    c = layers.GlobalAveragePooling2D(name="avg_pool")(latent)
    c = layers.Dropout(0.3, name="dropout")(c)
    return layers.Dense(N_CLASSES, activation="softmax", name="label")(c)


def decoder(latent):
    d = latent
    for step, channels in enumerate([64, 32, 16], start=1):
        d = layers.Conv2DTranspose(channels, 3, strides=2, padding="same", name=f"decoder_convt{step}")(d)
        d = layers.BatchNormalization(name=f"decoder_bn{step}")(d)
        d = layers.ReLU(name=f"decoder_relu{step}")(d)
    d = layers.Resizing(40, 101, name="decoder_resize")(d)
    return layers.Conv2D(1, 3, padding="same", name="reconstruction")(d)


def build_joint_model(logmel=None):
    audio, latent = encoder(logmel)
    return keras.Model(audio, {"label": classifier(latent), "reconstruction": decoder(latent)}, name="joint")


def build_autoencoder(logmel=None):
    audio, latent = encoder(logmel)
    return keras.Model(audio, {"reconstruction": decoder(latent)}, name="autoencoder")


def classifier_on_encoder(autoencoder, trainable, name):
    latent = autoencoder.get_layer("block3_pool").output
    model = keras.Model(autoencoder.input, {"label": classifier(latent)}, name=name)
    for layer in model.layers:
        if layer.name.startswith(ENCODER_LAYERS):
            layer.trainable = trainable and not isinstance(layer, layers.BatchNormalization)
    return model


def build_frozen_model(autoencoder):
    return classifier_on_encoder(autoencoder, trainable=False, name="frozen")


def build_finetune_model(autoencoder):
    return classifier_on_encoder(autoencoder, trainable=True, name="finetune")


class ReconstructionLoss(keras.losses.Loss):
    def __init__(self, logmel, **kwargs):
        super().__init__(**kwargs)
        self.logmel = logmel

    def call(self, clean_audio, reconstruction):
        return tf.reduce_mean(tf.square(self.logmel(clean_audio) - reconstruction), axis=[1, 2, 3])
