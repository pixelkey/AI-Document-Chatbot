# scripts/document_processing.py

import os
import logging
import tiktoken
from nltk.tokenize import sent_tokenize, blankline_tokenize

# Initialize tiktoken for token counting with cl100k_base encoding
# Note: This encoding is used for certain OpenAI models.
# It is only used for token counting and does not affect the tokenization process.
# However, it is important to use the same encoding for token counting as the model.
encoding = tiktoken.get_encoding("cl100k_base")

def normalize_text(text):
    """
    Normalize text by stripping leading/trailing whitespace and converting to lowercase.
    """
    return text.strip().lower()

def chunk_text_hybrid(text, chunk_size_max):
    """
    Split text into chunks based on paragraphs and sentences. If a paragraph
    exceeds the chunk size, it is split further by sentences.
    
    Args:
        text (str): The input text to be chunked.
        chunk_size_max (int): The maximum size of each chunk in tokens.
    
    Returns:
        list: List of text chunks.
    """
    paragraphs = blankline_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for paragraph in paragraphs:
        paragraph_tokens = encoding.encode(paragraph)
        paragraph_size = len(paragraph_tokens)

        if paragraph_size > chunk_size_max:
            process_large_paragraph(
                paragraph, chunk_size_max, chunks, current_chunk, current_chunk_size
            )
            current_chunk, current_chunk_size = finalize_chunk(current_chunk, chunks)
            continue

        if current_chunk_size + paragraph_size > chunk_size_max:
            current_chunk, current_chunk_size = finalize_chunk(current_chunk, chunks)

        current_chunk.append(paragraph)
        current_chunk_size += paragraph_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_large_paragraph(
    paragraph, chunk_size_max, chunks, current_chunk, current_chunk_size
):
    """
    Process a large paragraph by splitting it into smaller chunks based on sentences.
    
    Args:
        paragraph (str): The paragraph to be processed.
        chunk_size_max (int): The maximum size of each chunk in tokens.
        chunks (list): The list to store the resulting chunks.
        current_chunk (list): The current chunk being constructed.
        current_chunk_size (int): The current size of the chunk being constructed.
    """
    sentences = sent_tokenize(paragraph)
    for sentence in sentences:
        sentence_tokens = encoding.encode(sentence)
        sentence_size = len(sentence_tokens)

        if current_chunk_size + sentence_size > chunk_size_max:
            current_chunk, current_chunk_size = finalize_chunk(current_chunk, chunks)

        current_chunk.append(sentence)
        current_chunk_size += sentence_size

def finalize_chunk(current_chunk, chunks):
    """
    Finalize the current chunk by joining its content and adding it to the list of chunks.
    
    Args:
        current_chunk (list): The current chunk being finalized.
        chunks (list): The list to store the resulting chunks.
    
    Returns:
        tuple: An empty list and a size of 0 to reset the current chunk.
    """
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return [], 0

def load_documents_from_folder(folder_path, chunk_size_max):
    """
    Load all text files from the specified folder and its subdirectories.
    
    Args:
        folder_path (str): The root folder path to search for text files.
        chunk_size_max (int): The maximum size of each text chunk.
    
    Returns:
        list: List of document dictionaries containing the content and metadata.
    """
    documents = []
    
    # Traverse the directory tree and find all text files
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                content = load_file_content(file_path)
                chunks = chunk_text_hybrid(content, chunk_size_max)
                doc_id = len(documents)  # Use the current number of documents as an ID
                documents.extend(create_document_entries(doc_id, filename, chunks))
                logging.info(f"Loaded and chunked document {file_path} into {len(chunks)} chunks")
    
    if not documents:
        logging.error(f"No documents found in the folder: {folder_path}")
    
    return documents

def load_file_content(file_path):
    """
    Load the content of a text file.
    
    Args:
        file_path (str): The path to the text file.
    
    Returns:
        str: The content of the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def create_document_entries(doc_id, filename, chunks):
    """
    Create document entries with unique IDs for each chunk.
    
    Args:
        doc_id (int): The document ID.
        filename (str): The filename of the document.
        chunks (list): The chunks of text content.
    
    Returns:
        list: List of document dictionaries with ID, content, and filename.
    """
    return [
        {
            "id": f"{doc_id}-{chunk_idx}",
            "content": chunk,
            "filename": filename,
        }
        for chunk_idx, chunk in enumerate(chunks)
    ]
