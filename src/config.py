from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Añadiendo paths para el proceso del train - rama ml-pipeline
RAW_TRAIN = "data/raw/train.csv"
RAW_TEST  = "data/raw/test.csv"
PROC_PATH = "data/processed/train_proc.parquet"
FEAST_TRAINSET = "data/feast/training_set.parquet"

ARTIFACTS_DIR = PROJ_ROOT / "artifacts"

TARGET   = "Premium Amount"
DATE_COL = "Policy Start Date"

ORDINAL_COLS = {
   "Policy Type": ["Basic","Comprehensive","Premium"],
   "Education Level": ["High School","Bachelor's","Master's","PhD"],
   "Customer Feedback": ["Poor","Average","Good"],  # en tu caso es ordinal
}
BINARY_COLS = ["Gender","Smoking Status"]
NOMINAL_COLS = ["Marital Status","Occupation","Location","Exercise Frequency","Property Type"]

NUM_COLS = ["Age","Annual Income","Number of Dependents","Health Score","Previous Claims",
            "Vehicle Age","Credit Score","Insurance Duration"]


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
