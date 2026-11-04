import pandas as pd

from src.data import add_fixed_noise

SNRS = [20, 10, 5, 0, -5]
CONDITIONS = {"clean": None, **{f"{snr} dB": snr for snr in SNRS}}


def test_sets(x_test, noise):
    return {condition: add_fixed_noise(x_test, noise, snr) for condition, snr in CONDITIONS.items()}


def predict_labels(model, audio):
    return model.predict(audio, batch_size=512, verbose=0)["label"].argmax(axis=1)


def accuracy_table(predictions, y_test):
    table = pd.DataFrame({
        model: {condition: (labels == y_test).mean() for condition, labels in by_condition.items()}
        for model, by_condition in predictions.items()
    }).T
    table["noisy average"] = table[[f"{snr} dB" for snr in SNRS]].mean(axis=1)
    return 100 * table
