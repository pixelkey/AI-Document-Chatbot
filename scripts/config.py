# scripts/config.py

import os
from dotenv import load_dotenv
import logging

# Environment configuration setup
env_config = {
    "MODEL_SOURCE": {"default": "local", "type": str},  # Set to "local" or "openai"
    "OPENAI_API_KEY": {"default": "your-api-key-here", "type": str},
    "EMBEDDING_MODEL": {"default": "nomic-embed-text:latest", "type": str},
    "EMBEDDING_DIM": {"default": "768", "type": int},
    "FAISS_INDEX_PATH": {"default": "../embeddings/faiss_index.bin", "type": str},
    "METADATA_PATH": {"default": "../embeddings/metadata.pkl", "type": str},
    "DOCSTORE_PATH": {"default": "../embeddings/docstore.pkl", "type": str},
    "INGEST_PATH": {"default": "../ingest", "type": str},
    "SYSTEM_PROMPT": {
        "default": (
            "Please provide responses based only on the context document chunks provided if they are relevant to the user's prompt. "
            "If the context document chunks are not relevant, or if the information is not available, please let me know. "
            "Do not provide information beyond what is available in the context documents. "
            "Chunks are sorted by relevancy, where the first chunk listed is the most relevant. "
            "Note: Chunks may overlap and so may contain duplicate information."
        ),
        "type": str
    },
    "SIMILARITY_THRESHOLD": {"default": "0.25", "type": float},
    "TOP_SIMILARITY_RESULTS": {"default": "10", "type": int},
    "LLM_MODEL": {"default": "mistral:7b", "type": str},
    "LLM_MAX_TOKENS": {"default": "128000", "type": int},
    "CHUNK_SIZE_MAX": {"default": "512", "type": int},
    "CHUNK_OVERLAP_PERCENTAGE": {"default": "20", "type": int},
    "TOKEN_ENCODING": {"default": "cl100k_base", "type": str},
}

# Reset environment variables before loading .env to ensure they are not reused
for key in env_config:
    if key in os.environ:
        del os.environ[key]

# Load environment variables from .env file
load_dotenv()

# Apply environment settings
for key, settings in env_config.items():
    value = os.getenv(key, settings["default"])
    converted_value = settings["type"](value)
    os.environ[key] = str(converted_value)  # Store as string in OS environment
    globals()[key] = converted_value  # Set globally in the script

# Configure logging
logging.basicConfig(level=logging.INFO)

# Log the settings for verification
for key in env_config.keys():
    logging.info(f"{key}: {globals()[key]}")
