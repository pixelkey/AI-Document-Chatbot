# scripts/chatbot.py

import os
import faiss  # Ensure faiss is imported
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings  # Replace with actual OpenAIEmbeddings
from openai import OpenAI  # Import the OpenAI client
from document_processing import load_documents_from_folder, normalize_text, encoding
from vector_store_utils import (
    calculate_file_paths,
    load_or_initialize_vector_store
)
from interface import setup_gradio_interface  # Import the interface setup
import config  # Import the config file


def initialize_llm_client():
    """
    Initialize the LLM client and embeddings.
    Returns:
        tuple: LLM client, embeddings instance
    """
    # Initialize the OpenAI client and embeddings
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
    return client, embeddings


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


def main():
    # Initialize LLM client and embeddings
    client, embeddings = initialize_llm_client()

    # Initialize memory for conversation
    memory = ConversationBufferMemory()

    # Setup vector store
    vector_store, index_to_docstore_id = setup_vector_store(embeddings)

    # Prepare context to pass to the functions
    context = {
        "client": client,
        "memory": memory,
        "encoding": encoding,
        "embeddings": embeddings,
        "vector_store": vector_store,
        "EMBEDDING_DIM": config.EMBEDDING_DIM,
        "SYSTEM_PROMPT": config.SYSTEM_PROMPT,
        "LLM_MODEL": config.LLM_MODEL,
        "MAX_TOKENS": config.MAX_TOKENS,
        "SIMILARITY_THRESHOLD": config.SIMILARITY_THRESHOLD,
        "TOP_SIMILARITY_RESULTS": config.TOP_SIMILARITY_RESULTS,
    }

    # Setup and launch Gradio interface
    app = setup_gradio_interface(context)
    app.launch()


if __name__ == "__main__":
    main()
