import argparse
from pathlib import Path

import numpy as np

from src.audio import fit_to_one_second, load_audio
from src.config import CLASSES, MODELS_DIR
from src.data import NoiseBank, load_noise, load_split
from src.evaluate import accuracy_table, predict_labels, test_sets
from src.train import EPOCHS, LAMBDAS, load_model, train_model


def train(args):
    x_train, y_train = load_split("train")
    x_dev, y_dev = load_split("dev")
    noise = load_noise()
    names = LAMBDAS if args.model == "all" else [args.model]
    for name in names:
        train_model(name, x_train, y_train, x_dev, y_dev, noise, epochs=args.epochs, out_dir=args.models_dir)


def evaluate(args):
    x_test, y_test = load_split("test")
    sets = test_sets(x_test, NoiseBank(load_noise()["test"]))
    predictions = {}
    for name in LAMBDAS:
        model = load_model(name, args.models_dir)
        predictions[name] = {condition: predict_labels(model, audio) for condition, audio in sets.items()}
    print("Test accuracy (%)")
    print(accuracy_table(predictions, y_test).round(1).to_string())


def predict(args):
    audio = fit_to_one_second(load_audio(args.file)).astype("float32")
    model = load_model(args.model, args.models_dir)
    probability = model.predict(audio[None], verbose=0)["label"][0]
    for k in np.argsort(probability)[::-1][:3]:
        print(f"{CLASSES[k]:<12} {probability[k]:.2f}")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="where the weights are read or saved")

    parser = argparse.ArgumentParser(description="Spanish voice commands with a denoising autoencoder.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("train", parents=[common], help="train the models")
    p.add_argument("--model", choices=["all", *LAMBDAS], default="all")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.set_defaults(run=train)

    p = commands.add_parser("evaluate", parents=[common], help="accuracy on the test set, clean and with noise")
    p.set_defaults(run=evaluate)

    p = commands.add_parser("predict", parents=[common], help="classify an audio file")
    p.add_argument("file", type=Path)
    p.add_argument("--model", choices=list(LAMBDAS), default="multitask")
    p.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
