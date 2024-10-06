# scripts/initialize.py

import config  # Import the config file
from langchain.memory import ConversationBufferMemory
from vector_store_setup import setup_vector_store  # Import vector store setup
import tiktoken  # Import tiktoken for encoding

# Initialize tiktoken for token counting with cl100k_base encoding
token_encoding = tiktoken.get_encoding(config.TOKEN_ENCODING)

def initialize_model_and_retrieval():
    """
    Initialize the LLM client, embeddings, and any retrieval or RAG components.
    Returns:
        dict: Context dictionary with initialized components.
    """
    # Initialize memory for conversation
    memory = ConversationBufferMemory()

    if config.MODEL_SOURCE == "openai":
        # Initialize OpenAI client and embeddings
        from openai import OpenAI
        from langchain.embeddings import OpenAIEmbeddings
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
    elif config.MODEL_SOURCE == "local":
        # Initialize local model client and embeddings
        from langchain_community.llms import Ollama
        from langchain_community.embeddings import OllamaEmbeddings
        client = Ollama(model=config.LLM_MODEL)
        embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    else:
        raise ValueError("Invalid MODEL_SOURCE in config.")

    # Setup vector store
    vector_store = setup_vector_store(embeddings)

    # Prepare and return the context
    context = {
        "client": client,
        "memory": memory,
        "encoding": token_encoding,
        "embeddings": embeddings,
        "vector_store": vector_store,
        "EMBEDDING_DIM": config.EMBEDDING_DIM,
        "SYSTEM_PROMPT": config.SYSTEM_PROMPT,
        "LLM_MODEL": config.LLM_MODEL,
        "LLM_MAX_TOKENS": config.LLM_MAX_TOKENS,
        "SIMILARITY_THRESHOLD": config.SIMILARITY_THRESHOLD,
        "TOP_SIMILARITY_RESULTS": config.TOP_SIMILARITY_RESULTS,
        "MODEL_SOURCE": config.MODEL_SOURCE,
    }
    return context
