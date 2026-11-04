import sys

import numpy as np
import sounddevice as sd


def describe(audio):
    rms = np.sqrt(np.mean(audio**2))
    return f"peak {np.abs(audio).max():.4f}  rms {20 * np.log10(rms + 1e-12):6.1f} dBFS"


def main():
    print(f"Python: {sys.executable}")
    print(sd.query_devices())
    print(f"default input: {sd.default.device[0]}\n")
    inputs = [i for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    for i in inputs:
        device = sd.query_devices(i)
        for rate in sorted({16000, int(device["default_samplerate"])}):
            print(f"{i} {device['name']} at {rate} Hz, speak for 2 s...", end=" ", flush=True)
            try:
                audio = sd.rec(2 * rate, samplerate=rate, channels=1, dtype="float32", device=i)
                sd.wait()
                print(describe(audio))
            except Exception as error:
                print(f"ERROR {error}")
    print("\nIf every device shows 'peak 0.0000', allow microphone access for your terminal in "
          "System Settings > Privacy & Security > Microphone, quit it with Cmd+Q and open it again.")


if __name__ == "__main__":
    main()
