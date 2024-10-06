# scripts/vector_store_setup.py

import os
import faiss  # Ensure FAISS is imported
from vector_store_utils import (
    calculate_file_paths,
    load_or_initialize_vector_store
)
import config  # Import the config file

def setup_vector_store(embeddings):
    """
    Initialize or load the vector store.
    Returns:
        object: Initialized vector store.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_index_path, metadata_path, docstore_path = calculate_file_paths(
        script_dir, config.FAISS_INDEX_PATH, config.METADATA_PATH, config.DOCSTORE_PATH
    )
    ingest_path = os.path.join(script_dir, config.INGEST_PATH)

    # Initialize or load the FAISS index and associated components
    vector_store, _ = load_or_initialize_vector_store(
        embeddings, ingest_path, config.CHUNK_SIZE_MAX, config.EMBEDDING_DIM,
        faiss_index_path, metadata_path, docstore_path
    )

    # Ensure the index has a search method and is of the correct type
    index = vector_store.index
    assert isinstance(index, faiss.Index), "Index is not a valid FAISS index"

    return vector_store
