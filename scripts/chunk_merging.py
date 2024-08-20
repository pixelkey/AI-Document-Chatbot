# scripts/chunk_merging.py
import tiktoken
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
    current_merged_chunk = ""
    current_merged_score = None
    current_merged_ids = []
    current_merged_metadata = {}


    # What we need to accomplish:
    # We should group the chunks by the document they belong to and sort them by their ID
    # If there is an overlap between two chunks, within the same document, we should merge them
    # If there is no overlap between two chunks, within the same document, we should keep them separate
    #
    # The resulting new merged chunk should have:
    # - the same ID as the first chunk in the group
    # - the combined content of all chunks in the group (with overlapping content removed)
    # - the highest score of all chunks in the group
    # - the metadata of the first chunk in the group for filename and filepath
    # - the chunk_size should be the length of the content by adding the chunk_size of all chunks in the group minus the overlap_sizes
    # - the overlap_size should be 0 for the new merged chunk


    # Group the chunks by the document they belong to.
    # within a chunk the doc_id can be found in the metadata: chunk['metadata'].get('doc_id', None)
    # We should group the chunks by the document they belong to and sort them by their ID

    grouped_chunks = {}
    for chunk in retrieved_chunks:
        doc_id = chunk['metadata'].get('doc_id', None)
        if doc_id not in grouped_chunks:
            grouped_chunks[doc_id] = []
        grouped_chunks[doc_id].append(chunk)

    
    # Sort the chunks by their chunk['id'] within each group
    # For example if a group has chunks with IDs [3, 1, 2], they should be sorted as [1, 2, 3]
    for each_chunk, chunks in grouped_chunks.items():
        grouped_chunks[each_chunk] = sorted(chunks, key=lambda x: x['id'])

    # Log the grouped chunks for debugging
    print("Grouped Chunks:")
    
    for doc_id, chunks in grouped_chunks.items():
        print(f"Document ID: {doc_id}")
        for chunk in chunks:
            print(f"Chunk ID: {chunk['id']} | Score: {chunk['score']} | Content: {chunk['content']}")

    
    # Remove the overlap from the content of each chunk within the same group if there is an overlap
    for doc_id, chunks in grouped_chunks.items():
        
        # Only merge chunks if there is more than one chunk in the group
        if len(chunks) > 1:
            for chunk in chunks:
            # If the chunk has an overlap, we should remove it from the content
                chunk_overlap = chunk['metadata'].get('overlap_size', 0)
                chunk_content = chunk['content']
                # The overlap size is encoded using encoding = tiktoken.get_encoding("cl100k_base")
                # Therefore we need to encode the content and then slice it
                chunk_content = encoding.encode(chunk_content)
                chunk_content = chunk_content[chunk_overlap:]
                chunk_content = encoding.decode(chunk_content)
                chunk['content'] = chunk_content


    # Log the updated chunks after removing overlaps
    print("Merged Chunks:")
    for doc_id, chunks in grouped_chunks.items():
        print(f"Document ID: {doc_id}")
        for chunk in chunks:
            print(f"Chunk ID: {chunk['id']} | Score: {chunk['score']} | Content: {chunk['content']}")




    return retrieved_chunks
