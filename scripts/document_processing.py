# scripts/document_processing.py

import os
import logging
import tiktoken
from nltk.tokenize import sent_tokenize, blankline_tokenize

# Initialize tiktoken for token counting with cl100k_base encoding
# Note: This encoding is user for certain OpenAI models.
# It is only used for token counting and does not affect the tokenization process.
# However, it is important to use the same encoding for token counting as the model.
encoding = tiktoken.get_encoding("cl100k_base")

def normalize_text(text):
    return text.strip().lower()

def chunk_text_hybrid(text, chunk_size_max):
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
    sentences = sent_tokenize(paragraph)
    for sentence in sentences:
        sentence_tokens = encoding.encode(sentence)
        sentence_size = len(sentence_tokens)

        if current_chunk_size + sentence_size > chunk_size_max:
            current_chunk, current_chunk_size = finalize_chunk(current_chunk, chunks)

        current_chunk.append(sentence)
        current_chunk_size += sentence_size

def finalize_chunk(current_chunk, chunks):
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return [], 0

def load_documents_from_folder(folder_path, chunk_size_max):
    try:
        filenames = get_text_filenames(folder_path)
        documents = process_files_in_folder(folder_path, filenames, chunk_size_max)
    except FileNotFoundError:
        logging.error(f"Folder not found: {folder_path}")
        documents = []
    return documents

def get_text_filenames(folder_path):
    return [
        filename for filename in os.listdir(folder_path) if filename.endswith(".txt")
    ]

def process_files_in_folder(folder_path, filenames, chunk_size_max):
    documents = []
    for idx, filename in enumerate(filenames):
        file_path = os.path.join(folder_path, filename)
        content = load_file_content(file_path)
        chunks = chunk_text_hybrid(content, chunk_size_max)
        documents.extend(create_document_entries(idx, filename, chunks))
        logging.info(
            f"Loaded and chunked document {filename} into {len(chunks)} chunks"
        )
    return documents

def load_file_content(file_path):
    with open(file_path, "r") as file:
        return file.read()

def create_document_entries(doc_id, filename, chunks):
    return [
        {
            "id": f"{doc_id}-{chunk_idx}",
            "content": chunk,
            "filename": filename,
        }
        for chunk_idx, chunk in enumerate(chunks)
    ]
