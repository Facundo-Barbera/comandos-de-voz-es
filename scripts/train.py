import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR
from src.data import load_noise, load_split
from src.evaluate import accuracy_table, predict_labels, test_sets
from src.train import MODELS, load_model, train_model


def train(args):
    x_train, y_train = load_split("train")
    x_dev, y_dev = load_split("dev")
    noise = load_noise()
    for name in MODELS if args.modelo == "all" else [args.modelo]:
        train_model(name, x_train, y_train, x_dev, y_dev, noise, epochs=args.epochs, out_dir=args.models_dir)


def evaluate(args):
    x_test, y_test = load_split("test")
    sets = test_sets(x_test, load_noise()["test"])
    predictions = {}
    for name in MODELS:
        if (args.models_dir / f"{name}.weights.h5").exists():
            model = load_model(name, args.models_dir)
            predictions[name] = {condition: predict_labels(model, audio) for condition, audio in sets.items()}
    print("Test accuracy (%)")
    print(accuracy_table(predictions, y_test).round(1).to_string())


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the voice command models.")
    parser.add_argument("--modelo", choices=["all", *MODELS], default="all", help="model to train")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="where the weights are read or saved")
    parser.add_argument("--epochs", type=int, default=None, help="default: the value in src/train.py")
    parser.add_argument("--skip-training", action="store_true", help="only evaluate the saved weights")
    args = parser.parse_args()

    if not args.skip_training:
        train(args)
    evaluate(args)


if __name__ == "__main__":
    main()
