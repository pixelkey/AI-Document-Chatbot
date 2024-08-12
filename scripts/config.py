# scripts/config.py

import os
from dotenv import load_dotenv
import logging

# Delete the OPENAI_API_KEY from the OS environment in case it is set to avoid using the wrong key.
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# Load environment variables from .env file if it exists
load_dotenv()

# Configurable variables with defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "../embeddings/faiss_index.bin")
METADATA_PATH = os.getenv("METADATA_PATH", "../embeddings/metadata.pkl")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "../embeddings/docstore.pkl")
INGEST_PATH = os.getenv("INGEST_PATH", "../ingest")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Please provide responses based only on the context documents provided if they are relevant to the user's prompt. If the context documents are not relevant, or if the information is not available, please let me know. Do not provide information beyond what is available in the context documents.",
)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.25))
TOP_SIMILARITY_RESULTS = int(os.getenv("TOP_SIMILARITY_RESULTS", 3))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 128000))
CHUNK_SIZE_MAX = int(os.getenv("CHUNK_SIZE_MAX", 512))

# Configure logging
logging.basicConfig(level=logging.INFO)

# Print environment variables for debugging
logging.info(f"SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}")
logging.info(f"TOP_SIMILARITY_RESULTS: {TOP_SIMILARITY_RESULTS}")
