# scripts/chunk_merging.py

import tiktoken
import logging

# Initialize tiktoken for token counting with cl100k_base encoding
encoding = tiktoken.get_encoding("cl100k_base")

def chunk_merging(retrieved_chunks):
    """
    Merge retrieved chunks intelligently based on overlapping content.
    
    Args:
        retrieved_chunks (list of dict): A list of dictionaries where each dictionary contains:
                                         - 'id': The document ID
                                         - 'content': The text content of the chunk
                                         - 'score': The relevance score of the chunk
                                         - 'metadata': Additional metadata such as filename, filepath, chunk_size, overlap_size, etc.
    
    Returns:
        list of dict: A list of merged chunks with their metadata, including score, id, and metadata.
    """
    merged_chunks = []
    
    # Group the chunks by the document they belong to
    grouped_chunks = {}
    for chunk in retrieved_chunks:
        doc_id = chunk['metadata'].get('doc_id', None)
        if doc_id not in grouped_chunks:
            grouped_chunks[doc_id] = []
        grouped_chunks[doc_id].append(chunk)
    
    # Sort the chunks by their chunk ID within each group
    for doc_id, chunks in grouped_chunks.items():
        grouped_chunks[doc_id] = sorted(chunks, key=lambda x: x['id'])

    # Token-based overlap removal and merging with logging
    for doc_id, chunks in grouped_chunks.items():
        merged_chunk_content = ""
        merged_chunk_score = None
        merged_chunk_metadata = {}
        merged_chunk_size = 0

        for i, chunk in enumerate(chunks):
            chunk_content = chunk['content']
            chunk_size = chunk['metadata'].get('chunk_size', 0)
            overlap_size = chunk['metadata'].get('overlap_size', 0)

            # If this is not the first chunk, remove the overlap
            if i > 0 and overlap_size > 0:
                # Encode the chunk content to tokens
                chunk_tokens = encoding.encode(chunk_content)
                original_token_count = len(chunk_tokens)

                # Remove the overlap tokens from the beginning
                chunk_tokens = chunk_tokens[overlap_size:]
                reduced_token_count = len(chunk_tokens)

                # Log potential issues with minimal overlap or over-removal
                if overlap_size >= original_token_count:
                    logging.warning(
                        f"Overlap size {overlap_size} is greater than or equal to the original chunk token count {original_token_count}. "
                        f"This could indicate over-removal of content in chunk ID {chunk['id']}."
                    )

                if original_token_count - reduced_token_count != overlap_size:
                    logging.warning(
                        f"Mismatch in overlap removal: expected to remove {overlap_size} tokens but removed {original_token_count - reduced_token_count} "
                        f"tokens in chunk ID {chunk['id']}. Possible minimal overlap or incorrect calculation."
                    )

                # Decode the tokens back to text
                chunk_content = encoding.decode(chunk_tokens)
                chunk_size = reduced_token_count  # Adjust size to the new token count

            # Merge content and update metadata
            merged_chunk_content += chunk_content + " "
            merged_chunk_size += chunk_size  # Use the adjusted chunk_size
            merged_chunk_score = max(merged_chunk_score, chunk['score']) if merged_chunk_score is not None else chunk['score']
            merged_chunk_metadata = chunk['metadata']

        # Add the final merged chunk to the list
        if merged_chunk_content:
            merged_chunks.append({
                "id": chunks[0]['id'],  # Use the ID of the first chunk in the group
                "content": merged_chunk_content.strip(),
                "score": merged_chunk_score,
                "metadata": {
                    **merged_chunk_metadata,
                    "chunk_size": merged_chunk_size,
                    "overlap_size": 0  # Overlap has been removed in the merged chunk
                }
            })

    return merged_chunks
