# scripts/chatbot.py

import os
import openai
import gradio as gr
from dotenv import load_dotenv
import logging
import nltk
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from document_processing import load_documents_from_folder, normalize_text, encoding
from vector_store_utils import (
    calculate_file_paths,
    load_or_initialize_vector_store
)
from chatbot_functions import chatbot_response, clear_history  # Import the functions here
import faiss  # Ensure faiss is imported

# Configure logging
logging.basicConfig(level=logging.INFO)

# Download NLTK data for sentence tokenization
nltk.download("punkt")

# Delete the OPENAI_API_KEY from the OS environment in case it is set to avoid using the wrong key.
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# Load environment variables from .env file if it exists
load_dotenv()

# Configurable variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "text-embedding-3-small"
)  # Updated embedding model
EMBEDDING_DIM = int(
    os.getenv("EMBEDDING_DIM", 1536)
)  # Ensure this matches the actual embedding dimension
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "../embeddings/faiss_index.bin")
METADATA_PATH = os.getenv("METADATA_PATH", "../embeddings/metadata.pkl")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "../embeddings/docstore.pkl")
INGEST_PATH = os.getenv("INGEST_PATH", "../ingest")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Please provide responses based only on the context documents provided if they are relevant to the user's prompt. If the context documents are not relevant, or if the information is not available, please let me know. Do not provide information beyond what is available in the context documents.",
)
SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", 0.25)
)  # Lowered threshold
TOP_SIMILARITY_RESULTS = int(
    os.getenv("TOP_SIMILARITY_RESULTS", 3)
)  # Ensure top results are correctly set
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 128000))
CHUNK_SIZE_MAX = int(os.getenv("CHUNK_SIZE_MAX", 512))  # Chunk size in tokens

# Print environment variables
logging.info(f"SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}")
logging.info(f"TOP_SIMILARITY_RESULTS: {TOP_SIMILARITY_RESULTS}")

# Initialize OpenAI client
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Initialize memory for conversation
memory = ConversationBufferMemory()

# Calculate absolute paths based on the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_index_path, metadata_path, docstore_path = calculate_file_paths(
    script_dir, FAISS_INDEX_PATH, METADATA_PATH, DOCSTORE_PATH
)

# Correct the ingest path relative to the script's directory
ingest_path = os.path.join(script_dir, INGEST_PATH)

# Initialize or load vector store
vector_store, index_to_docstore_id = load_or_initialize_vector_store(
    embeddings, ingest_path, CHUNK_SIZE_MAX, EMBEDDING_DIM, faiss_index_path, metadata_path, docstore_path
)

# Prepare context to pass to the functions
context = {
    "client": client,
    "memory": memory,
    "encoding": encoding,
    "embeddings": embeddings,
    "vector_store": vector_store,
    "EMBEDDING_DIM": EMBEDDING_DIM,
    "SYSTEM_PROMPT": SYSTEM_PROMPT,
    "LLM_MODEL": LLM_MODEL,
    "MAX_TOKENS": MAX_TOKENS,
    "SIMILARITY_THRESHOLD": SIMILARITY_THRESHOLD,
    "TOP_SIMILARITY_RESULTS": TOP_SIMILARITY_RESULTS,
}

# Create Gradio Blocks interface
with gr.Blocks() as demo:
    # Output fields
    chat_history = gr.Textbox(label="Chat History", lines=10, interactive=False)
    references = gr.Textbox(label="References", lines=2, interactive=False)

    # Input fields
    input_text = gr.Textbox(label="Input Text")
    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Clear")

    # Setup event handlers
    submit_button.click(
        lambda input_text: chatbot_response(input_text, context),
        inputs=input_text,
        outputs=[chat_history, references, input_text]
    )
    clear_button.click(lambda: clear_history(context), outputs=[chat_history, references, input_text])

    input_text.submit(
        lambda input_text: chatbot_response(input_text, context),
        inputs=input_text,
        outputs=[chat_history, references, input_text]
    )

    # Layout
    gr.Column([chat_history, references, input_text, submit_button, clear_button])

# Launch the interface
demo.launch()
