from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
RAW_ZIP = RAW_DIR / "dataset_to_release.zip"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_CACHE = PROCESSED_DIR / "raw.npz"
CLEAN_DATA = PROCESSED_DIR / "clean.npz"
CLEANING_RULES = ROOT / "p1" / "cleaning.py"  # CLEAN_DATA is stale if this is newer

CONFIGS_DIR = ROOT / "configs"
OUTPUTS_DIR = ROOT / "outputs"
CV_RESULTS = OUTPUTS_DIR / "cv_results.csv"
OOF_DIR = OUTPUTS_DIR / "oof"  # one folder of fold files per cross-validated run
MODELS_DIR = OUTPUTS_DIR / "models"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
FIGURES_DIR = ROOT / "report" / "figures"
