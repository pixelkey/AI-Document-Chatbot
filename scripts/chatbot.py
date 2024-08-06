import os
import openai
import gradio as gr
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
import tiktoken
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Explicitly delete the OPENAI_API_KEY from the OS environment in case it is set to avoid using the wrong key.
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# Load environment variables from .env file if it exists
load_dotenv()

# Configurable variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")  # Updated embedding model
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))  # Ensure this is appropriate for your model
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "../embeddings/faiss_index.bin")
METADATA_PATH = os.getenv("METADATA_PATH", "../embeddings/metadata.pkl")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "../embeddings/docstore.pkl")
INGEST_PATH = os.getenv("INGEST_PATH", "../ingest")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Please only provide responses based on the information provided. If it is not available, please let me know.")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.25))  # Lowered threshold
TOP_SIMILARITY_RESULTS = int(os.getenv("TOP_SIMILARITY_RESULTS", 3))  # Ensure top results are correctly set
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 128000))

# Print environment variables
logging.info(f"SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}")
logging.info(f"TOP_SIMILARITY_RESULTS: {TOP_SIMILARITY_RESULTS}")

# Initialize tiktoken for token counting with cl100k_base encoding
encoding = tiktoken.get_encoding("cl100k_base")

# Normalize text
def normalize_text(text):
    return text.strip().lower()

# Load documents from a folder
def load_documents_from_folder(folder_path):
    documents = []
    try:
        for idx, filename in enumerate(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r") as file:
                    content = file.read()
                    documents.append({"id": idx, "content": content, "filename": filename})
                    logging.info(f"Loaded document {filename}")
    except FileNotFoundError:
        logging.error(f"Folder not found: {folder_path}")
    return documents

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Initialize memory for conversation
memory = ConversationBufferMemory()

# Calculate absolute paths based on the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_index_path = os.path.join(script_dir, FAISS_INDEX_PATH)
metadata_path = os.path.join(script_dir, METADATA_PATH)
docstore_path = os.path.join(script_dir, DOCSTORE_PATH)
ingest_path = os.path.join(script_dir, INGEST_PATH)

# Save FAISS index, metadata, and docstore to disk
def save_faiss_index_metadata_and_docstore(faiss_index, metadata, docstore, faiss_index_path, metadata_path, docstore_path):
    faiss.write_index(faiss_index, faiss_index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    with open(docstore_path, "wb") as f:
        pickle.dump(docstore._dict, f)
    logging.info("Saved FAISS index, metadata, and docstore to disk.")

# Load FAISS index, metadata, and docstore from disk
def load_faiss_index_metadata_and_docstore(faiss_index_path, metadata_path, docstore_path):
    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path) and os.path.exists(docstore_path):
        faiss_index = faiss.read_index(faiss_index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        with open(docstore_path, "rb") as f:
            docstore_data = pickle.load(f)
            docstore = InMemoryDocstore(docstore_data)
        logging.info("Loaded FAISS index, metadata, and docstore from disk.")
        return faiss_index, metadata, docstore
    return None, None, None

# Load previous FAISS index, metadata, and docstore if they exist
loaded_faiss_index, loaded_metadata, loaded_docstore = load_faiss_index_metadata_and_docstore(faiss_index_path, metadata_path, docstore_path)

# Initialize the vector store
if loaded_faiss_index and loaded_metadata and loaded_docstore:
    vector_store = FAISS(
        embedding_function=embeddings,
        index=loaded_faiss_index,
        docstore=loaded_docstore,
        index_to_docstore_id=loaded_metadata
    )
    index_to_docstore_id = loaded_metadata
    logging.info("Loaded vector store from saved files.")
else:
    # Create a new FAISS index and docstore if loading failed
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}  # Simple index-to-docstore mapping
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    # Load and process documents from 'ingest' folder
    documents = load_documents_from_folder(ingest_path)

    if not documents:
        raise ValueError(f"No documents found in the folder: {ingest_path}")

    # Log the number of documents loaded
    logging.info(f"Total documents loaded: {len(documents)}")

    # Normalize and convert documents to vectors, then add to vector store
    for doc in documents:
        normalized_doc = normalize_text(doc["content"])
        vectors = embeddings.embed_query(normalized_doc)
        logging.info(f"Processed document {doc['filename']} for embedding with vector: {vectors[:5]}...")  # Show the first 5 elements of the vector for brevity
        doc_id = int(doc["id"])  # Ensure the ID is an integer
        # Add document to vector store
        document = Document(page_content=normalized_doc, metadata={"id": doc_id, "filename": doc["filename"]})
        vector_store.add_texts([normalized_doc], metadatas=[document.metadata], ids=[doc_id])
        docstore._dict[doc_id] = document  # Directly add Document to the in-memory dictionary
        index_to_docstore_id[doc_id] = doc_id  # Map index to docstore ID
        logging.info(f"Added document {doc_id} to vector store with normalized content.")

    # Save FAISS index, metadata, and docstore to disk
    save_faiss_index_metadata_and_docstore(vector_store.index, index_to_docstore_id, docstore, faiss_index_path, metadata_path, docstore_path)

# Search with scores
def similarity_search_with_score(query, k=100):
    # Embed the query
    query_vector = np.array(embeddings.embed_query(query)).reshape(1, -1)
    logging.info(f"Processed query for embedding with vector: {query_vector[:5]}...")  # Show the first 5 elements of the vector for brevity

    # Search the FAISS index
    D, I = vector_store.index.search(query_vector, k)

    results = []
    for i, score in zip(I[0], D[0]):
        if i != -1:  # Ensure valid index
            try:
                doc = vector_store.docstore.search(vector_store.index_to_docstore_id[i])
                results.append((doc, score))
                logging.info(f"Matched document {doc.metadata['id']} with score {score} and content: {doc.page_content[:200]}...")  # Show first 200 characters of content for brevity
            except KeyError as e:
                logging.error(f"KeyError finding document: {e}")

    # Log the results for debugging
    logging.info(f"Total documents considered: {len(results)}")
    for res in results:
        logging.info(f"Document ID: {res[0].metadata['id']}, Score: {res[1]}")

    return results

# Truncate text to fit within a token limit
def truncate_text_to_tokens(text, max_tokens):
    tokens = encoding.encode(text)
    if (len(tokens) > max_tokens):
        truncated_text = encoding.decode(tokens[:max_tokens])
        logging.warning(f"Truncated text to fit within {max_tokens} tokens.")
        return truncated_text
    return text

# Truncate documents to fit the token limit
def truncate_documents_to_fit_token_limit(documents, input_text):
    logging.info(f"Initial number of documents: {len(documents)}")

    # Calculate the token budget for each document
    max_tokens_per_doc = (MAX_TOKENS - len(encoding.encode(input_text)) - len(encoding.encode(SYSTEM_PROMPT))) // (len(documents) + 1)

    truncated_docs = []
    for doc in documents:
        truncated_content = truncate_text_to_tokens(doc.page_content, max_tokens_per_doc)
        truncated_docs.append(Document(page_content=truncated_content, metadata=doc.metadata))

    while True:
        combined_input = f"{SYSTEM_PROMPT}\n\n" + "\n\n".join([
            f"Document {doc.metadata['id']} ({doc.metadata['filename']}):\n{doc.page_content}"
            for doc in truncated_docs
        ] + [input_text])

        total_tokens = len(encoding.encode(combined_input))
        logging.info(f"Token count for combined input: {total_tokens}")

        if total_tokens <= MAX_TOKENS:
            break

        removed_doc = truncated_docs.pop()
        logging.info(f"Removed document {removed_doc.metadata['id']} with length: {len(encoding.encode(removed_doc.page_content))} tokens")

    logging.info(f"Final number of documents after truncation: {len(truncated_docs)}")
    return truncated_docs

# Chatbot response function with RAG
def chatbot_response(input_text):
    # Normalize input text
    normalized_input = normalize_text(input_text)
    logging.info(f"Normalized input: {normalized_input}")

    # Retrieve relevant documents with similarity scores
    try:
        search_results = similarity_search_with_score(normalized_input)
        logging.info(f"Retrieved documents with scores")
    except KeyError as e:
        logging.error(f"Error while retrieving documents: {e}")
        return memory.buffer, "Error in retrieving documents.", ""

    # Filter documents based on similarity score and limit to top results with the token limit consideration
    filtered_results = [
        result
        for result in search_results
        if result[1] >= SIMILARITY_THRESHOLD
    ]
    logging.info(f"Filtered results by similarity threshold: {[result[1] for result in filtered_results]}")

    # Remove duplicates
    seen_ids = set()
    unique_filtered_results = []
    for result in filtered_results:
        if result[0].metadata['id'] not in seen_ids:
            unique_filtered_results.append(result)
            seen_ids.add(result[0].metadata['id'])

    # Sort filtered results by similarity score in descending order
    unique_filtered_results.sort(key=lambda x: x[1], reverse=True)
    filtered_docs = unique_filtered_results[:TOP_SIMILARITY_RESULTS]
    logging.info(f"Top similarity results: {[(result[0].metadata['id'], result[1]) for result in filtered_docs]}")

    # Check if the combined input is within the token limit
    truncated_docs = truncate_documents_to_fit_token_limit(
        [result[0] for result in filtered_docs], input_text
    )

    combined_input = f"{SYSTEM_PROMPT}\n\n" + "\n\n".join([
        f"Document {doc.metadata['id']} ({doc.metadata['filename']}):\n{doc.page_content}"
        for doc in truncated_docs
    ] + [input_text])

    logging.info(f"Prepared combined input for LLM")

    # Create the list of messages for the Chat API
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text}
    ]
    for doc in truncated_docs:
        messages.append({"role": "user", "content": doc.page_content})

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=min(
                MAX_TOKENS - len(encoding.encode(str(messages))), 8000
            ) # Adjust the max tokens for completion
        )
        logging.info(f"Generated LLM response successfully")
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return memory.buffer, "Error generating response.", ""

    # Update conversation memory
    memory.save_context({"input": input_text}, {"output": response.choices[0].message.content})

    # Construct reference list
    references = "References:\n" + "\n".join([
        f"Document {doc.metadata['id']}: {doc.metadata['filename']}"
        for doc in truncated_docs
    ])

    # Return chat history, references, and clear input
    return memory.buffer, references, ""

# Clear the chat history and references
def clear_history():
    memory.clear()
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
        outputs=[chat_history, references, input_text]
    )
    clear_button.click(clear_history, outputs=[chat_history, references, input_text])
    input_text.submit(
        chatbot_response,
        inputs=input_text,
        outputs=[chat_history, references, input_text]
    )

    # Layout
    gr.Column(
        [chat_history, references, input_text, submit_button, clear_button]
    )

# Launch the interface
demo.launch()