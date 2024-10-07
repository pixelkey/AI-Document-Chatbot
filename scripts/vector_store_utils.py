# scripts/vector_store_utils.py

import os
import faiss
import logging
import numpy as np
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from faiss_utils import (
    save_faiss_index_metadata_and_docstore,
    load_faiss_index_metadata_and_docstore,
    train_faiss_index,
    add_vectors_to_faiss_index
)
from document_processing import load_documents_from_folder, normalize_text
import config

def calculate_file_paths(script_dir, faiss_index_path, metadata_path, docstore_path):
    faiss_index_path = os.path.join(script_dir, faiss_index_path)
    metadata_path = os.path.join(script_dir, metadata_path)
    docstore_path = os.path.join(script_dir, docstore_path)
    return faiss_index_path, metadata_path, docstore_path

def load_or_initialize_vector_store(
    embeddings, ingest_path, CHUNK_SIZE_MAX, EMBEDDING_DIM, faiss_index_path, metadata_path, docstore_path
):
    # Load existing index, metadata, and docstore
    loaded_faiss_index, loaded_metadata, loaded_docstore = (
        load_faiss_index_metadata_and_docstore(
            faiss_index_path, metadata_path, docstore_path
        )
    )

    if loaded_faiss_index and loaded_metadata and loaded_docstore:
        vector_store = FAISS(
            embedding_function=embeddings,
            index=loaded_faiss_index,
            docstore=loaded_docstore,
            index_to_docstore_id=loaded_metadata,
        )
        logging.info("Loaded vector store from saved files.")
        return vector_store, loaded_metadata

    # If loading fails, initialize a new vector store
    documents = load_documents_from_folder(ingest_path, CHUNK_SIZE_MAX)

    if not documents:
        raise ValueError(f"No documents found in the folder: {ingest_path}")

    logging.info(f"Total chunks loaded: {len(documents)}")

    num_documents = len(documents)
    num_clusters = min(
        max(10, num_documents // 2), num_documents
    )

    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIVFFlat(
        quantizer, EMBEDDING_DIM, num_clusters, faiss.METRIC_INNER_PRODUCT
    )
    index.nprobe = 10

    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    training_vectors = []
    for doc in documents:
        normalized_doc = normalize_text(doc["content"])
        vectors = embeddings.embed_documents([normalized_doc])[0]
        doc_sample = doc["content"][:50]

        logging.info(
            f"Embedding dimension: {len(vectors)}, Config EMBEDDING_DIM: {EMBEDDING_DIM} for doc: {doc_sample}"
        )
        assert (
            len(vectors) == EMBEDDING_DIM
        ), f"Embedding dimension {len(vectors)} does not match expected {EMBEDDING_DIM} for doc: {doc_sample}"

        training_vectors.append(vectors)

    training_vectors = np.array(training_vectors, dtype="float32")
    # Normalize training vectors
    faiss.normalize_L2(training_vectors)

    # Train the FAISS index
    train_faiss_index(vector_store, training_vectors, num_clusters)

    # Add vectors to FAISS index
    add_vectors_to_faiss_index(documents, vector_store, embeddings, normalize_text)

    # Save the FAISS index, metadata, and docstore
    save_faiss_index_metadata_and_docstore(
        vector_store.index,
        index_to_docstore_id,
        docstore,
        faiss_index_path,
        metadata_path,
        docstore_path
    )

    logging.info(f"Document ID to Docstore Mapping: {index_to_docstore_id}")

    return vector_store, index_to_docstore_id
