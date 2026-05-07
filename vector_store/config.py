import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 4
FETCH_K = 10
LAMBDA_MULT = 0.5

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mistral")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")