# scripts/faiss_utils.py

import faiss
import pickle
import os
import logging
import numpy as np
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
import config


def save_faiss_index_metadata_and_docstore(
    faiss_index, metadata, docstore, faiss_index_path, metadata_path, docstore_path
):
    faiss.write_index(faiss_index, faiss_index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    with open(docstore_path, "wb") as f:
        pickle.dump(docstore._dict, f)
    logging.info("Saved FAISS index, metadata, and docstore to disk.")


def load_faiss_index_metadata_and_docstore(
    faiss_index_path, metadata_path, docstore_path
):
    if (
        os.path.exists(faiss_index_path)
        and os.path.exists(metadata_path)
        and os.path.exists(docstore_path)
    ):
        faiss_index = faiss.read_index(faiss_index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        with open(docstore_path, "rb") as f:
            docstore_data = pickle.load(f)
            docstore = InMemoryDocstore(docstore_data)
        logging.info("Loaded FAISS index, metadata, and docstore from disk.")
        return faiss_index, metadata, docstore
    return None, None, None


def train_faiss_index(vector_store, training_vectors, num_clusters):
    if not vector_store.index.is_trained:
        vector_store.index.train(training_vectors)
        logging.info(f"FAISS index trained with {num_clusters} clusters.")


def add_vectors_to_faiss_index(chunks, vector_store, embeddings, normalize_text):
    docstore = vector_store.docstore
    index_to_docstore_id = vector_store.index_to_docstore_id

    for idx, doc in enumerate(chunks):
        normalized_doc = normalize_text(doc["content"])
        # Use embed_documents for document embeddings
        vectors = embeddings.embed_documents([normalized_doc])[0]
        vectors = np.array(vectors, dtype="float32").reshape(1, -1)
        # Normalize the vector
        faiss.normalize_L2(vectors)
        # Add the vector to the index
        vector_store.index.add(vectors)

        chunk_id = str(idx)  # Use string IDs for consistency
        # Add chunk to docstore
        chunk = Document(
            page_content=normalized_doc,
            metadata={
                "id": chunk_id,
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "filepath": doc["filepath"],
                "chunk_size": doc["chunk_size"],
                "overlap_size": doc["overlap_size"],
            },
        )
        docstore._dict[chunk_id] = chunk  # Directly add Chunk to the in-memory dictionary
        index_to_docstore_id[len(index_to_docstore_id)] = chunk_id  # Map index position to chunk ID

        logging.info(f"Added chunk {chunk_id} to vector store with normalized content.")


def similarity_search_with_score(query, vector_store, embeddings, EMBEDDING_DIM, k=100):
    # Embed the query
    query_vector = embeddings.embed_query(query)
    query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)
    # Normalize query vector for cosine similarity
    faiss.normalize_L2(query_vector)

    logging.info(f"Query embedding shape: {query_vector.shape}")
    logging.info(f"Query embedding first 5 values: {query_vector[0][:5]}")

    # Ensure query vector dimensionality matches the FAISS index dimensionality
    assert query_vector.shape[1] == EMBEDDING_DIM, (
        f"Query vector dimension {query_vector.shape[1]} does not match index dimension {EMBEDDING_DIM}"
    )

    # Search the FAISS index
    D, I = vector_store.index.search(query_vector, k)

    results = []
    for i, score in zip(I[0], D[0]):
        if i != -1:  # Ensure valid index
            try:
                chunk_id = vector_store.index_to_docstore_id.get(i, None)
                if chunk_id is None:
                    raise KeyError(f"Chunk ID {i} not found in mapping.")
                doc = vector_store.docstore.search(chunk_id)
                results.append({
                    "id": chunk_id,
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata
                })
                logging.info(f"Matched chunk {chunk_id} with score {score} and content: {doc.page_content[:200]}...")
            except KeyError as e:
                logging.error(f"KeyError finding chunk id {i}: {e}")
    return results
