from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
VAL_FILE = RAW_DATA_DIR / "val.csv"
LAWS_FILE = RAW_DATA_DIR / "laws_de.csv"
SUBMISSION_FILE = PROJECT_ROOT / "submission.csv"

DEFAULT_BM25_TOP_K = 10
DEFAULT_VECTOR_TOP_K = 10
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MODEL_NAME = "local-hf-model"
