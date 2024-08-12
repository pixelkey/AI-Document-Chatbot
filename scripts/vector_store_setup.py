# scripts/vector_store_setup.py

import os
from vector_store_utils import (
    calculate_file_paths,
    load_or_initialize_vector_store
)
import config  # Import the config file

def setup_vector_store(embeddings):
    """
    Initialize or load the vector store.
    Returns:
        tuple: Initialized vector store, and mapping of index to document store IDs
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_index_path, metadata_path, docstore_path = calculate_file_paths(
        script_dir, config.FAISS_INDEX_PATH, config.METADATA_PATH, config.DOCSTORE_PATH
    )
    ingest_path = os.path.join(script_dir, config.INGEST_PATH)
    
    return load_or_initialize_vector_store(
        embeddings, ingest_path, config.CHUNK_SIZE_MAX, config.EMBEDDING_DIM, 
        faiss_index_path, metadata_path, docstore_path
    )
