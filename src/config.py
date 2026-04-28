from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "raw" / "nepal_dataset_v2_train.csv"
TEST_FILE = BASE_DIR / "data" / "raw" / "nepal_dataset_v2_test.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

for folder in [OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "Irrigation_Need"
ID_COL = "id"
RANDOM_STATE = 42
TEST_SIZE = 0.2