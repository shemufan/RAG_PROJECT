# core/config.py
import os
from dotenv import load_dotenv

load_dotenv("api_key.env", override=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(BASE_DIR, "db"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))
TESTDATA_DIR = os.getenv("TESTDATA_DIR", os.path.join(BASE_DIR, "testdata"))

MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH", os.path.join(BASE_DIR, "models", "sentence-transformer")
)

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "")
