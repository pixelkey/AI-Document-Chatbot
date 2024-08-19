# scripts/faiss_utils.py

import faiss
import pickle
import os
import logging
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.docstore.document import Document  # Import the Document class
from chunk_merging import intelligent_chunk_merging  # Import the merging function

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

def add_vectors_to_faiss_index(documents, vector_store, embeddings, normalize_text):
    docstore = vector_store.docstore
    index_to_docstore_id = vector_store.index_to_docstore_id

    for idx, doc in enumerate(documents):
        normalized_doc = normalize_text(doc["content"])
        vectors = np.array(embeddings.embed_query(normalized_doc), dtype="float32")

        logging.info(
            f"Processed document chunk for embedding with vector: {vectors[:5]}..."
        )  # Show the first 5 elements of the vector for brevity
        doc_id = idx  # Use integer ID for chunked documents
        # Add document to vector store
        document = Document(
            page_content=normalized_doc,
            metadata={"id": doc_id, "filename": doc["filename"], "filepath": doc["filepath"], "chunk_size": doc["chunk_size"], "overlap_size": doc["overlap_size"]},
        )
        vector_store.add_texts(
            [normalized_doc], metadatas=[document.metadata], ids=[doc_id]
        )
        docstore._dict[doc_id] = (
            document  # Directly add Document to the in-memory dictionary
        )
        index_to_docstore_id[doc_id] = doc_id  # Use integer ID for the docstore ID

        logging.info(
            f"Added document chunk {doc_id} to vector store with normalized content."
        )

def similarity_search_with_score(query, vector_store, embeddings, EMBEDDING_DIM, k=100):
    # Embed the query
    query_vector = np.array(embeddings.embed_query(query), dtype="float32").reshape(
        1, -1
    )
    logging.info(
        f"Processed query for embedding with vector: {query_vector[:5]}..."
    )  # Show the first 5 elements of the vector for brevity

    # Ensure query vector dimensionality matches the FAISS index dimensionality
    assert (
        query_vector.shape[1] == EMBEDDING_DIM
    ), f"Query vector dimension {query_vector.shape[1]} does not match index dimension {EMBEDDING_DIM}"

    # Normalize query vector for cosine similarity
    faiss.normalize_L2(query_vector)

    # Search the FAISS index
    D, I = vector_store.index.search(query_vector, k)

    results = []
    for i, score in zip(I[0], D[0]):
        if i != -1:  # Ensure valid index
            try:
                doc_id = vector_store.index_to_docstore_id.get(
                    i, None
                )  # Use integer ID
                if doc_id is None:
                    raise KeyError(f"Document ID {i} not found in mapping.")
                doc = vector_store.docstore.search(doc_id)
                results.append({
                    "id": doc_id, 
                    "content": doc.page_content, 
                    "score": float(score),
                    "metadata": doc.metadata
                })
                logging.info(
                    f"Matched document {doc_id} with score {score} and content: {doc.page_content[:200]}..."
                )  # Show first 200 characters of content for brevity
                logging.info(
                    f"Metadata for document {doc_id}: filename={doc.metadata.get('filename', '')}, filepath={doc.metadata.get('filepath', '')}, chunk_size={doc.metadata.get('chunk_size', 0)}, overlap_size={doc.metadata.get('overlap_size', 0)}"
                )
            except KeyError as e:
                logging.error(f"KeyError finding document id {i}: {e}")
                if doc_id not in vector_store.docstore._dict:
                    logging.error(f"Document id {doc_id} not found in docstore._dict")

    # Log the results for debugging
    logging.info(f"Total documents considered: {len(results)}")
    for res in results:
        logging.info(f"Document ID: {res['id']}, Score: {res['score']}")

    return results

    # # Merge overlapping chunks intelligently
    # merged_chunks = intelligent_chunk_merging(results)

    # return merged_chunks
