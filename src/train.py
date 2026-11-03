import json
import time

import keras
import numpy as np

from src.config import MODELS_DIR, N_CLASSES, SEED, SILENCE
from src.data import NoiseBank, add_fixed_noise, mix, time_shift
from src.features import LogMel
from src.model import ReconstructionLoss, build_model

LAMBDAS = {"base": 0.0, "multitask": 0.3}
EPOCHS = 30
BATCH_SIZE = 256


class NoisyBatches(keras.utils.PyDataset):
    def __init__(self, x, y, noise_bank, with_reconstruction, steps, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y, self.noise_bank = x, y, noise_bank
        self.with_reconstruction = with_reconstruction
        self.steps = steps
        weight = np.bincount(y)[y] ** -0.5
        self.probability = weight / weight.sum()
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return self.steps

    def __getitem__(self, _):
        rng, n = self.rng, BATCH_SIZE
        index = rng.choice(len(self.y), n, p=self.probability)
        clean = time_shift(self.x[index], rng)

        noisy = mix(clean, self.noise_bank.sample(n, rng),
                    snr_db=rng.uniform(-5, 20, n), silence_level_db=rng.uniform(-50, -15, n))
        no_speech = (clean**2).mean(axis=1) < 1e-8
        gets_noise = (rng.random(n) < 0.8) | no_speech
        noisy = np.where(gets_noise[:, None], noisy, clean)

        volume = 10 ** (rng.uniform(-6, 6, (n, 1)) / 20)
        targets = {"label": np.eye(N_CLASSES, dtype="float32")[self.y[index]]}
        if self.with_reconstruction:
            targets["reconstruction"] = (clean * volume).astype("float32")
        return (noisy * volume).astype("float32"), targets


def validation_set(x_dev, y_dev, noise_bank):
    x_val = np.concatenate([add_fixed_noise(x_dev, noise_bank, None, seed=7),
                            add_fixed_noise(x_dev, noise_bank, "random", seed=99)])
    return x_val, np.concatenate([y_dev, y_dev]), np.concatenate([x_dev, x_dev])


def compile_model(model, lam, total_steps):
    warmup = int(0.15 * total_steps)
    learning_rate = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-3 / 25, decay_steps=total_steps - warmup, alpha=1e-3,
        warmup_target=3e-3, warmup_steps=warmup)

    losses = {"label": keras.losses.CategoricalCrossentropy(label_smoothing=0.05)}
    weights = {"label": 1.0}
    if lam > 0:
        losses["reconstruction"] = ReconstructionLoss(model.get_layer("logmel"))
        weights["reconstruction"] = lam
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate, weight_decay=1e-3),
                  loss=losses, loss_weights=weights, metrics={"label": ["accuracy"]})


def train_model(name, x_train, y_train, x_dev, y_dev, noise, epochs=EPOCHS, out_dir=MODELS_DIR):
    lam = LAMBDAS[name]
    keras.utils.set_random_seed(SEED)
    logmel = LogMel(name="logmel")
    logmel.adapt(x_train[y_train != SILENCE][:6000])
    model = build_model(with_decoder=lam > 0, logmel=logmel)

    steps = len(y_train) // BATCH_SIZE
    compile_model(model, lam, epochs * steps)
    batches = NoisyBatches(x_train, y_train, NoiseBank(noise["train"]), with_reconstruction=lam > 0, steps=steps)

    x_val, y_val, x_val_clean = validation_set(x_dev, y_dev, NoiseBank(noise["dev"]))
    val_targets = {"label": np.eye(N_CLASSES, dtype="float32")[y_val]}
    if lam > 0:
        val_targets["reconstruction"] = x_val_clean

    metric = "val_label_accuracy" if lam > 0 else "val_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{name}.weights.h5"
    keep_best = keras.callbacks.ModelCheckpoint(
        weights_path, monitor=metric, mode="max", save_best_only=True, save_weights_only=True)

    start = time.time()
    history = model.fit(batches, epochs=epochs, validation_data=(x_val, val_targets),
                        validation_batch_size=512, callbacks=[keep_best], verbose=2)
    print(f"{name}: {(time.time() - start) / 60:.1f} minutes")

    history = {key: [float(v) for v in values] for key, values in history.history.items()}
    (out_dir / f"{name}_history.json").write_text(json.dumps(history))
    model.load_weights(weights_path)
    return model


def load_model(name, models_dir=MODELS_DIR):
    model = build_model(with_decoder=LAMBDAS[name] > 0)
    model.load_weights(models_dir / f"{name}.weights.h5")
    return model
