# scripts/initialize.py

import config  # Import the config file
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings  # Replace with actual OpenAIEmbeddings
from openai import OpenAI  # Import the OpenAI client
from vector_store_setup import setup_vector_store  # Import vector store setup
import tiktoken  # Import tiktoken for encoding

# Initialize tiktoken for token counting with cl100k_base encoding
encoding = tiktoken.get_encoding("cl100k_base")

def initialize_model_and_retrieval():
    """
    Initialize the LLM client, embeddings, and any retrieval or RAG components.
    Returns:
        dict: Context dictionary with initialized components.
    """
    # Initialize the OpenAI client and embeddings
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)

    # Initialize memory for conversation
    memory = ConversationBufferMemory()

    # Setup vector store
    vector_store = setup_vector_store(embeddings)

    # Prepare and return the context
    context = {
        "client": client,
        "memory": memory,
        "encoding": encoding,  # Ensure encoding is included
        "embeddings": embeddings,
        "vector_store": vector_store,
        "EMBEDDING_DIM": config.EMBEDDING_DIM,
        "SYSTEM_PROMPT": config.SYSTEM_PROMPT,
        "LLM_MODEL": config.LLM_MODEL,
        "MAX_TOKENS": config.MAX_TOKENS,
        "SIMILARITY_THRESHOLD": config.SIMILARITY_THRESHOLD,
        "TOP_SIMILARITY_RESULTS": config.TOP_SIMILARITY_RESULTS,
    }
    return context
