import os
import logging
import tiktoken
from nltk.tokenize import sent_tokenize, blankline_tokenize
from nltk import download as nltk_download
from dotenv import load_dotenv
import re

# Ensure that the 'punkt' tokenizer is downloaded
nltk_download('punkt')

# Load environment variables from .env file
load_dotenv()
CHUNK_OVERLAP_PERCENTAGE = int(os.getenv("CHUNK_OVERLAP_PERCENTAGE", 20))  # Default to 20% if not set

# Initialize tiktoken for token counting with cl100k_base encoding
encoding = tiktoken.get_encoding("cl100k_base")

def normalize_text(text):
    """
    Normalize text by removing excessive whitespace (more than one line).
    """
    # Replace multiple line breaks with a single space
    text = re.sub(r'\n\s*\n+', ' ', text.strip())

    return text

def chunk_text_hybrid(text, chunk_size_max):
    """
    Split text into chunks based on paragraphs and sentences. 
    Retain paragraph breaks to preserve context.
    Chunks overlap by a percentage of the chunk size.
    
    Args:
        text (str): The input text to be chunked.
        chunk_size_max (int): The maximum size of each chunk in tokens.
    
    Returns:
        list: List of text chunks with their token counts and overlap sizes.
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
        chunks.append((" ".join(current_chunk), current_chunk_size))

    overlap_size_max = int((CHUNK_OVERLAP_PERCENTAGE / 100) * chunk_size_max)
    overlapped_chunks = add_chunk_overlap(chunks, chunk_size_max, overlap_size_max)

    return overlapped_chunks

def add_chunk_overlap(chunks, chunk_size_max, overlap_size_max):
    """
    Add overlap between chunks by a specified number of tokens, ensuring overlap
    occurs at the sentence level and includes meaningful content.
    
    Args:
        chunks (list): List of chunks with their token counts.
        chunk_size_max (int): The maximum size of each chunk in tokens.
        overlap_size_max (int): The number of tokens to overlap between chunks.
    
    Returns:
        list: List of overlapped chunks with updated token counts and overlap sizes.
    """
    overlapped_chunks = []
    previous_chunk_sentences = []

    for chunk, chunk_size in chunks:
        current_chunk_tokens = encoding.encode(chunk)

        if previous_chunk_sentences:
            # Create the overlap text by joining previous sentences
            overlap_text = " ".join(previous_chunk_sentences)
            overlap_tokens = encoding.encode(overlap_text)
            current_chunk_tokens = overlap_tokens + current_chunk_tokens

        # Ensure the chunk doesn't exceed the max size
        if len(current_chunk_tokens) > chunk_size_max:
            current_chunk_tokens = current_chunk_tokens[:chunk_size_max]

        # Decode the current chunk to text
        current_chunk_text = encoding.decode(current_chunk_tokens)

        # Tokenize the current chunk into sentences
        sentences = sent_tokenize(current_chunk_text)

        # Identify sentences for the next overlap
        overlap_text = ""
        overlap_tokens = []
        while sentences and len(overlap_tokens) < overlap_size_max:
            overlap_text = sentences.pop(-1) + " " + overlap_text
            overlap_tokens = encoding.encode(overlap_text.strip())

        overlap_size = len(overlap_tokens)  # Get the correct overlap size in tokens

        # If it is the first chunk, there is no overlap. So set it to 0
        if not overlapped_chunks:
            overlap_tokens = []
            overlap_size = 0

        # Save the current chunk and prepare the next overlap
        overlapped_chunks.append((current_chunk_text, chunk_size, overlap_size))
        previous_chunk_sentences = sent_tokenize(overlap_text.strip())

    return overlapped_chunks

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
        chunk_text = " ".join(current_chunk)
        chunk_size = len(encoding.encode(chunk_text))
        chunks.append((chunk_text, chunk_size))
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

                # Get Relative path to the file from the root folder and don't include the file name
                relative_path = os.path.relpath(root, folder_path)

                documents.extend(create_document_entries(doc_id, filename, relative_path, chunks))
                logging.info(f"Loaded and chunked document {relative_path}/{filename} into {len(chunks)} chunks")
    
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

def create_document_entries(doc_id, filename, filepath, chunks):
    """
    Create document entries with unique IDs for each chunk, including file path.
    
    Args:
        doc_id (int): The document ID.
        filename (str): The filename of the document.
        filepath (str): The relative path of the document.
        chunks (list): The chunks of text content and their token counts.
    
    Returns:
        list: List of document dictionaries with chunk ID, document ID, content, filename, filepath, token count, and overlap metadata.
    """
    return [
        {
            "id": chunk_idx,
            "doc_id": doc_id,
            "content": chunk,
            "filename": filename,
            "filepath": filepath,
            "chunk_size": chunk_size, # The token count of the chunk
            "overlap_size": overlap_size # The token count of the overlap
        }
        for chunk_idx, (chunk, chunk_size, overlap_size) in enumerate(chunks)
    ]
