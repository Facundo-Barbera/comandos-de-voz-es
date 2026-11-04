import json
import time

import keras
import numpy as np

from src.config import MODELS_DIR, SEED, SILENCE
from src.data import add_fixed_noise, mix, random_noise, time_shift
from src.features import LogMel
from src.model import (ReconstructionLoss, build_autoencoder, build_finetune_model, build_frozen_model,
                       build_joint_model)

MODELS = ["joint", "frozen", "finetune"]
LAMBDA = 0.3
EPOCHS = {"autoencoder": 30, "joint": 30, "frozen": 15, "finetune": 15}
LEARNING_RATE = {"autoencoder": 1e-3, "joint": 1e-3, "frozen": 1e-3, "finetune": 1e-4}
BATCH_SIZE = 256


def noisy_batches(x, y, noise, outputs, seed=SEED):
    rng = np.random.default_rng(seed)
    weight = np.bincount(y)[y] ** -0.5
    probability = weight / weight.sum()
    n = BATCH_SIZE
    while True:
        index = rng.choice(len(y), n, p=probability)
        clean = time_shift(x[index], rng)
        noisy = mix(clean, random_noise(noise, n, rng),
                    snr_db=rng.uniform(-5, 20, n), silence_level_db=rng.uniform(-50, -15, n))
        no_speech = (clean**2).mean(axis=1) < 1e-8
        gets_noise = (rng.random(n) < 0.8) | no_speech
        noisy = np.where(gets_noise[:, None], noisy, clean)

        volume = 10 ** (rng.uniform(-6, 6, (n, 1)) / 20)
        targets = {"label": y[index], "reconstruction": (clean * volume).astype("float32")}
        yield (noisy * volume).astype("float32"), {key: targets[key] for key in outputs}


def validation_set(x_dev, y_dev, noise):
    x_val = np.concatenate([add_fixed_noise(x_dev, noise, None, seed=7),
                            add_fixed_noise(x_dev, noise, "random", seed=99)])
    return x_val, np.concatenate([y_dev, y_dev]), np.concatenate([x_dev, x_dev])


def adapted_logmel(x_train, y_train):
    logmel = LogMel(name="logmel")
    logmel.adapt(x_train[y_train != SILENCE][:6000])
    return logmel


def fit(model, name, losses, loss_weights, monitor, data, epochs, out_dir):
    x_train, y_train, x_dev, y_dev, noise = data
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE[name]), loss=losses, loss_weights=loss_weights,
                  metrics={"label": ["accuracy"]} if "label" in losses else None)

    x_val, y_val, x_val_clean = validation_set(x_dev, y_dev, noise["dev"])
    val_targets = {key: {"label": y_val, "reconstruction": x_val_clean}[key] for key in losses}
    sample_labels = y_train if "label" in losses else np.zeros_like(y_train)

    mode = "min" if monitor == "val_loss" else "max"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{name}.weights.h5"
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint(weights_path, monitor=monitor, mode=mode,
                                        save_best_only=True, save_weights_only=True),
    ]

    start = time.time()
    history = model.fit(noisy_batches(x_train, sample_labels, noise["train"], list(losses)),
                        steps_per_epoch=len(y_train) // BATCH_SIZE, epochs=epochs,
                        validation_data=(x_val, val_targets), validation_batch_size=512,
                        callbacks=callbacks, verbose=2)
    print(f"{name}: {(time.time() - start) / 60:.1f} minutes")

    history = {key: [float(v) for v in values] for key, values in history.history.items()}
    (out_dir / f"{name}_history.json").write_text(json.dumps(history))
    model.load_weights(weights_path)
    return model


def train_joint(data, epochs, out_dir):
    model = build_joint_model(adapted_logmel(*data[:2]))
    losses = {"label": keras.losses.SparseCategoricalCrossentropy(),
              "reconstruction": ReconstructionLoss(model.get_layer("logmel"))}
    return fit(model, "joint", losses, {"label": 1.0, "reconstruction": LAMBDA}, "val_label_accuracy",
               data, epochs, out_dir)


def pretrained_autoencoder(data, out_dir):
    if (out_dir / "autoencoder.weights.h5").exists():
        autoencoder = build_autoencoder()
        autoencoder.load_weights(out_dir / "autoencoder.weights.h5")
        return autoencoder
    autoencoder = build_autoencoder(adapted_logmel(*data[:2]))
    losses = {"reconstruction": ReconstructionLoss(autoencoder.get_layer("logmel"))}
    return fit(autoencoder, "autoencoder", losses, None, "val_loss", data, EPOCHS["autoencoder"], out_dir)


def train_frozen(data, epochs, out_dir):
    model = build_frozen_model(pretrained_autoencoder(data, out_dir))
    losses = {"label": keras.losses.SparseCategoricalCrossentropy()}
    return fit(model, "frozen", losses, None, "val_accuracy", data, epochs, out_dir)


def train_finetune(data, epochs, out_dir):
    if not (out_dir / "frozen.weights.h5").exists():
        train_frozen(data, EPOCHS["frozen"], out_dir)
    model = build_finetune_model(build_autoencoder())
    model.load_weights(out_dir / "frozen.weights.h5")
    losses = {"label": keras.losses.SparseCategoricalCrossentropy()}
    return fit(model, "finetune", losses, None, "val_accuracy", data, epochs, out_dir)


def train_model(name, x_train, y_train, x_dev, y_dev, noise, epochs=None, out_dir=MODELS_DIR):
    keras.utils.set_random_seed(SEED)
    train = {"joint": train_joint, "frozen": train_frozen, "finetune": train_finetune}[name]
    return train((x_train, y_train, x_dev, y_dev, noise), epochs or EPOCHS[name], out_dir)


def load_model(name, models_dir=MODELS_DIR):
    if name == "joint":
        model = build_joint_model()
    else:
        model = {"frozen": build_frozen_model, "finetune": build_finetune_model}[name](build_autoencoder())
    model.load_weights(models_dir / f"{name}.weights.h5")
    return model
