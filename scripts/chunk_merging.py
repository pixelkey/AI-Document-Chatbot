# scripts/chunk_merging.py

from nltk.tokenize import sent_tokenize

def intelligent_chunk_merging(retrieved_chunks):
    """
    Merge retrieved chunks intelligently based on overlapping content.
    
    Args:
        retrieved_chunks (list of dict): A list of dictionaries where each dictionary contains:
                                         - 'id': The document ID
                                         - 'content': The text content of the chunk
                                         - 'score': The relevance score of the chunk
                                         - 'metadata': Additional metadata such as filename and filepath
    
    Returns:
        list of dict: A list of merged chunks with their metadata, including score, id, and metadata.
    """
    merged_chunks = []
    current_merged_chunk = ""
    current_merged_score = None
    current_merged_ids = []
    current_merged_metadata = {}

    # Sort chunks by their ID to maintain order
    sorted_chunks = sorted(retrieved_chunks, key=lambda x: x['id'])

    for i in range(len(sorted_chunks)):
        chunk_content = sorted_chunks[i]['content']
        chunk_score = sorted_chunks[i]['score']
        chunk_id = sorted_chunks[i]['id']
        chunk_metadata = sorted_chunks[i].get('metadata', {})

        # If overlap metadata exists, use it to trim the overlap
        overlap_size = chunk_metadata.get('overlap_size', 0)
        if overlap_size > 0:
            chunk_content = chunk_content[overlap_size:]

        if not current_merged_chunk:
            current_merged_chunk = chunk_content
            current_merged_score = chunk_score
            current_merged_ids = [chunk_id]
            current_merged_metadata = chunk_metadata
        else:
            overlap = find_overlap(current_merged_chunk, chunk_content)
            if overlap:
                # If overlap is found, merge the current chunk with the next one
                current_merged_chunk += " " + chunk_content[len(overlap):]
                current_merged_score = max(current_merged_score, chunk_score)
                current_merged_ids.append(chunk_id)
                # Metadata could be merged or replaced; here we keep the metadata of the first chunk
            else:
                # If no overlap is found, finalize the current merged chunk and start a new one
                merged_chunks.append({
                    "id": current_merged_ids[0],  # Use the first ID or combine the IDs
                    "content": current_merged_chunk,
                    "score": current_merged_score,
                    "metadata": current_merged_metadata
                })
                current_merged_chunk = chunk_content
                current_merged_score = chunk_score
                current_merged_ids = [chunk_id]
                current_merged_metadata = chunk_metadata

    # Add the last merged chunk
    if current_merged_chunk:
        merged_chunks.append({
            "id": current_merged_ids[0],  # Use the first ID or combine the IDs
            "content": current_merged_chunk,
            "score": current_merged_score,
            "metadata": current_merged_metadata
        })

    return merged_chunks

def find_overlap(chunk1, chunk2):
    """
    Find the overlapping content between two chunks.
    
    Args:
        chunk1 (str): The first text chunk.
        chunk2 (str): The second text chunk.
    
    Returns:
        str: The overlapping content, or an empty string if no overlap is found.
    """
    chunk1_sentences = sent_tokenize(chunk1)
    chunk2_sentences = sent_tokenize(chunk2)

    overlap_text = ""

    # Try to match the end of chunk1 with the start of chunk2
    for i in range(len(chunk1_sentences)):
        for j in range(len(chunk2_sentences)):
            if chunk1_sentences[i:] == chunk2_sentences[:len(chunk1_sentences) - i]:
                overlap_text = " ".join(chunk2_sentences[:len(chunk1_sentences) - i])
                return overlap_text

    return overlap_text
