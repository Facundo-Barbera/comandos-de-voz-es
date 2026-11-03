from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLIPS_DIR = DATA_DIR / "clips"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

SAMPLE_RATE = 16000
SEED = 0

WORDS = [
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "siguiente", "anterior", "música", "alto", "sigue", "adelante",
    "arriba", "abajo", "izquierda", "derecha",
]
CLASSES = WORDS + ["unknown", "silence"]
N_CLASSES = len(CLASSES)
UNKNOWN = CLASSES.index("unknown")
SILENCE = CLASSES.index("silence")
