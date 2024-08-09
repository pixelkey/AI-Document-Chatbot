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
import nltk
from nltk.tokenize import sent_tokenize, blankline_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO)

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Delete the OPENAI_API_KEY from the OS environment in case it is set to avoid using the wrong key.
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# Load environment variables from .env file if it exists
load_dotenv()

# Configurable variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # Updated embedding model
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))  # Ensure this matches the actual embedding dimension
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "../embeddings/faiss_index.bin")
METADATA_PATH = os.getenv("METADATA_PATH", "../embeddings/metadata.pkl")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "../embeddings/docstore.pkl")
INGEST_PATH = os.getenv("INGEST_PATH", "../ingest")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Please provide responses based only on the context documents provided if they are relevant to the user's prompt. If the context documents are not relevant, or if the information is not available, please let me know. Do not provide information beyond what is available in the context documents.")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.25))  # Lowered threshold
TOP_SIMILARITY_RESULTS = int(os.getenv("TOP_SIMILARITY_RESULTS", 3))  # Ensure top results are correctly set
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 128000))
CHUNK_SIZE_MAX = int(os.getenv("CHUNK_SIZE_MAX", 512))  # Chunk size in tokens

# Print environment variables
logging.info(f"SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}")
logging.info(f"TOP_SIMILARITY_RESULTS: {TOP_SIMILARITY_RESULTS}")

# Initialize tiktoken for token counting with cl100k_base encoding
encoding = tiktoken.get_encoding("cl100k_base")

# Normalize text
def normalize_text(text):
    return text.strip().lower()

# Split text into chunks based on paragraphs and sentences
def chunk_text_hybrid(text, chunk_size_max):
    paragraphs = blankline_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = encoding.encode(paragraph)
        paragraph_size = len(paragraph_tokens)
        
        if current_chunk_size + paragraph_size > chunk_size_max:
            # If adding the current paragraph exceeds the chunk size, finalize the current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_chunk_size = 0
            
            if paragraph_size > chunk_size_max:
                # If the paragraph itself is too large, split by sentences
                sentences = sent_tokenize(paragraph)
                for sentence in sentences:
                    sentence_tokens = encoding.encode(sentence)
                    sentence_size = len(sentence_tokens)
                    
                    if current_chunk_size + sentence_size > chunk_size_max:
                        # If adding the current sentence exceeds the chunk size, create a new chunk
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = []
                            current_chunk_size = 0
                        
                        current_chunk.append(sentence)
                        current_chunk_size = sentence_size
                    else:
                        current_chunk.append(sentence)
                        current_chunk_size += sentence_size
            else:
                current_chunk.append(paragraph)
                current_chunk_size = paragraph_size
        else:
            current_chunk.append(paragraph)
            current_chunk_size += paragraph_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Load documents from a folder and chunk them
def load_documents_from_folder(folder_path):
    documents = []
    try:
        for idx, filename in enumerate(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r") as file:
                    content = file.read()
                    chunks = chunk_text_hybrid(content, CHUNK_SIZE_MAX)
                    for chunk_idx, chunk in enumerate(chunks):
                        documents.append({
                            "id": f"{idx}-{chunk_idx}",
                            "content": chunk,
                            "filename": filename
                        })
                    logging.info(f"Loaded and chunked document {filename} into {len(chunks)} chunks")
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
    documents = load_documents_from_folder(ingest_path)

    if not documents:
        raise ValueError(f"No documents found in the folder: {ingest_path}")

    # Log the number of documents loaded
    logging.info(f"Total chunks loaded: {len(documents)}")

    # Determine the number of clusters dynamically based on the number of documents
    num_documents = len(documents)
    num_clusters = min(max(10, num_documents // 2), num_documents)  # At least 10 clusters or half the number of documents, but not more than the number of documents

    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product (cosine similarity)
    index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, num_clusters, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = 10  # Number of clusters to search. Adjust based on performance needs.
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}  # Simple index-to-docstore mapping
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    # Collect vectors for training
    training_vectors = []
    for doc in documents:
        normalized_doc = normalize_text(doc["content"])
        vectors = embeddings.embed_query(normalized_doc)
        
        # Debugging: Validate embedding dimensions
        logging.info(f"Embedding dimension: {len(vectors)}, Config EMBEDDING_DIM: {EMBEDDING_DIM}")
        assert len(vectors) == EMBEDDING_DIM, f"Embedding dimension {len(vectors)} does not match expected {EMBEDDING_DIM}"
        
        training_vectors.append(vectors)

    # Convert to numpy array
    training_vectors = np.array(training_vectors, dtype='float32')

    # Train the FAISS index with collected vectors if using IVFFlat
    if not vector_store.index.is_trained:
        vector_store.index.train(training_vectors)
        logging.info(f"FAISS index trained with {num_clusters} clusters.")

    # Add vectors to the FAISS index after training
    for idx, doc in enumerate(documents):
        normalized_doc = normalize_text(doc["content"])
        vectors = np.array(embeddings.embed_query(normalized_doc), dtype='float32')

        logging.info(f"Processed document chunk for embedding with vector: {vectors[:5]}...")  # Show the first 5 elements of the vector for brevity
        doc_id = idx  # Use integer ID for chunked documents
        # Add document to vector store
        document = Document(page_content=normalized_doc, metadata={"id": doc_id, "filename": doc["filename"]})
        vector_store.add_texts([normalized_doc], metadatas=[document.metadata], ids=[doc_id])
        docstore._dict[doc_id] = document  # Directly add Document to the in-memory dictionary
        index_to_docstore_id[doc_id] = doc_id  # Use integer ID for the docstore ID
        logging.info(f"Added document chunk {doc_id} to vector store with normalized content.")
    
    # Save FAISS index, metadata, and docstore to disk
    save_faiss_index_metadata_and_docstore(vector_store.index, index_to_docstore_id, docstore, faiss_index_path, metadata_path, docstore_path)

    # Verify mapping after saving
    logging.info(f"Document ID to Docstore Mapping: {index_to_docstore_id}")

# Search with scores
def similarity_search_with_score(query, k=100):
    # Embed the query
    query_vector = np.array(embeddings.embed_query(query), dtype='float32').reshape(1, -1)
    logging.info(f"Processed query for embedding with vector: {query_vector[:5]}...")  # Show the first 5 elements of the vector for brevity

    # Ensure query vector dimensionality matches the FAISS index dimensionality
    assert query_vector.shape[1] == EMBEDDING_DIM, f"Query vector dimension {query_vector.shape[1]} does not match index dimension {EMBEDDING_DIM}"

    # Normalize query vector for cosine similarity
    faiss.normalize_L2(query_vector)

    # Search the FAISS index
    D, I = vector_store.index.search(query_vector, k)

    results = []
    for i, score in zip(I[0], D[0]):
        if i != -1:  # Ensure valid index
            try:
                doc_id = vector_store.index_to_docstore_id.get(i, None)  # Use integer ID
                if doc_id is None:
                    raise KeyError(f"Document ID {i} not found in mapping.")
                doc = vector_store.docstore.search(doc_id)
                results.append((doc, score))
                logging.info(f"Matched document {doc_id} with score {score} and content: {doc.page_content[:200]}...")  # Show first 200 characters of content for brevity
            except KeyError as e:
                logging.error(f"KeyError finding document id {i}: {e}")
                if doc_id not in vector_store.docstore._dict:
                    logging.error(f"Document id {doc_id} not found in docstore._dict")

    # Log the results for debugging
    logging.info(f"Total documents considered: {len(results)}")
    for res in results:
        logging.info(f"Document ID: {res[0].metadata['id']}, Score: {res[1]}")

    return results

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
    filtered_docs = [result[0] for result in unique_filtered_results[:TOP_SIMILARITY_RESULTS]]
    logging.info(f"Top similarity results: {[(doc.metadata['id'], score) for doc, score in unique_filtered_results[:TOP_SIMILARITY_RESULTS]]}")

    # Create the final combined input
    combined_input = f"{SYSTEM_PROMPT}\n\n"
    combined_input += "\n\n".join([
        f"Context Document {idx+1} ({doc.metadata['filename']}):\n{doc.page_content}"
        for idx, doc in enumerate(filtered_docs)
    ])
    combined_input += f"\n\nUser Prompt:\n{input_text}"

    logging.info(f"Prepared combined input for LLM")

    # log the combined input for debugging
    logging.info(f"Combined input: {combined_input}")

    # Create the list of messages for the Chat API
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for idx, doc in enumerate(filtered_docs):
        messages.append({"role": "user", "content": f"Context Document {idx+1} ({doc.metadata['filename']}): {doc.page_content}"})
    messages.append({"role": "user", "content": f"User Prompt: {input_text}"})

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=min(
                MAX_TOKENS - len(encoding.encode(str(messages))), 8000
            )  # Adjust the max tokens for completion
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
        for doc in filtered_docs
    ])

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
