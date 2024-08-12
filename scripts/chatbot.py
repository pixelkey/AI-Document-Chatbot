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
from faiss_utils import similarity_search_with_score  # Import the function here
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

# Chatbot response function with RAG
def chatbot_response(input_text):
    # Normalize input text
    normalized_input = normalize_text(input_text)
    logging.info(f"Normalized input: {normalized_input}")

    # Retrieve relevant documents with similarity scores
    try:
        search_results = similarity_search_with_score(
            normalized_input, vector_store, embeddings, EMBEDDING_DIM
        )
        logging.info(f"Retrieved documents with scores")
    except KeyError as e:
        logging.error(f"Error while retrieving documents: {e}")
        return memory.buffer, "Error in retrieving documents.", ""

    # Filter documents based on similarity score and limit to top results with the token limit consideration
    filtered_results = [
        result for result in search_results if result[1] >= SIMILARITY_THRESHOLD
    ]
    logging.info(
        f"Filtered results by similarity threshold: {[result[1] for result in filtered_results]}"
    )

    # Remove duplicates by content
    seen_contents = set()
    unique_filtered_results = []
    for result in filtered_results:
        content_hash = hash(result[0].page_content)
        if content_hash not in seen_contents:
            unique_filtered_results.append(result)
            seen_contents.add(content_hash)

    # Sort filtered results by similarity score in descending order
    unique_filtered_results.sort(key=lambda x: x[1], reverse=True)
    filtered_docs = [
        result[0] for result in unique_filtered_results[:TOP_SIMILARITY_RESULTS]
    ]
    logging.info(
        f"Top similarity results: {[(doc.metadata['id'], score) for doc, score in unique_filtered_results[:TOP_SIMILARITY_RESULTS]]}"
    )

    # Create the final combined input
    combined_input = f"{SYSTEM_PROMPT}\n\n"
    combined_input += "\n\n".join(
        [
            f"Context Document {idx+1} ({doc.metadata['filename']}):\n{doc.page_content}"
            for idx, doc in enumerate(filtered_docs)
        ]
    )
    combined_input += f"\n\nUser Prompt:\n{input_text}"

    logging.info(f"Prepared combined input for LLM")

    # log the combined input for debugging
    logging.info(f"Combined input: {combined_input}")

    # Create the list of messages for the Chat API
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for idx, doc in enumerate(filtered_docs):
        messages.append(
            {
                "role": "user",
                "content": f"Context Document {idx+1} ({doc.metadata['filename']}): {doc.page_content}",
            }
        )
    messages.append({"role": "user", "content": f"User Prompt: {input_text}"})

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=min(
                MAX_TOKENS - len(encoding.encode(str(messages))), 8000
            ),  # Adjust the max tokens for completion
        )
        logging.info(f"Generated LLM response successfully")
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return memory.buffer, "Error generating response.", ""

    # Update conversation memory
    memory.save_context(
        {"input": input_text}, {"output": response.choices[0].message.content}
    )

    # Construct reference list
    references = "References:\n" + "\n".join(
        [
            f"Chunk {doc.metadata['id']}: {doc.metadata['filename']}"
            for doc in filtered_docs
        ]
    )

    # Return chat history, references, and clear input
    return memory.buffer, references, ""


# Clear the chat history and references
def clear_history():
    memory.clear()  # Clear the conversation memory
    return "", "", ""


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
        chatbot_response,
        inputs=input_text,
        outputs=[chat_history, references, input_text],
    )
    clear_button.click(clear_history, outputs=[chat_history, references, input_text])
    input_text.submit(
        chatbot_response,
        inputs=input_text,
        outputs=[chat_history, references, input_text],
    )

    # Layout
    gr.Column([chat_history, references, input_text, submit_button, clear_button])

# Launch the interface
demo.launch()
