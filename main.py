import argparse
from pathlib import Path

from src.config import MODELS_DIR
from src.microphone import listen
from src.train import MODELS, load_model


def main():
    parser = argparse.ArgumentParser(description="Live Spanish voice commands from the microphone.")
    parser.add_argument("--model", choices=MODELS, default="joint")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="where the weights are read")
    parser.add_argument("--device", default=None, help="input device index or name (see scripts/mic_check.py)")
    parser.add_argument("--threshold", type=float, default=0.6, help="minimum probability to report a word")
    parser.add_argument("--gate-db", type=float, default=-50.0, help="windows quieter than this count as silence")
    args = parser.parse_args()

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    model = load_model(args.model, args.models_dir)
    listen(model, device=device, threshold=args.threshold, gate_db=args.gate_db)


if __name__ == "__main__":
    main()
