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


    # Merge the chunks within each group
    for doc_id, chunks in grouped_chunks.items():
        current_merged_chunk = ""
        current_merged_score = None
        current_merged_ids = []
        current_merged_metadata = {}

        for chunk in chunks:
            # If the current merged chunk is empty, initialize it with the current chunk's content
            if not current_merged_chunk:
                current_merged_chunk = chunk['content']
                current_merged_score = chunk['score']
                current_merged_ids.append(chunk['id'])
                current_merged_metadata = chunk['metadata']
                current_merged_chunk_size = chunk['metadata']['chunk_size']
                current_merged_overlap_size = chunk['metadata']['overlap_size']
            
            else:
                # Check if there is an overlap between the current merged chunk and the current chunk
                # The overlap of each chunk can be found in the metadata: chunk['metadata'].get('overlap_size', 0)
                overlap_size = chunk['metadata'].get('overlap_size', 0)
                # The chunk size of each chunk can be found in the metadata: chunk['metadata'].get('chunk_size', 0)
                chunk_size = chunk['metadata'].get('chunk_size', 0)
                # The overlap_size and chunk_size are in number of tokens
                # The overlap_size is the number of tokens that are repeated between the chunks
                # The chunk_size is the total number of tokens in the chunk

                # If there is an overlap, merge the chunks intelligently
                if overlap_size > 0:
                    # Find the position of the overlap in the current merged chunk
                    overlap_position = current_merged_chunk.rfind(chunk['content'][:overlap_size])
                    # Merge the chunks based on the overlap position
                    current_merged_chunk += chunk['content'][overlap_size:]
                    current_merged_chunk_size += chunk_size - overlap_size
                    current_merged_overlap_size = 0
                    current_merged_ids.append(chunk['id'])
                    # Update the score if the current chunk has a higher score
                    if chunk['score'] > current_merged_score:
                        current_merged_score = chunk['score']
                
                else:
                    # If there is no overlap, finalize the current merged chunk and start a new one
                    merged_chunks.append({
                        'id': current_merged_ids[0],
                        'content': current_merged_chunk,
                        'score': current_merged_score,
                        'metadata': current_merged_metadata
                    })
                    current_merged_chunk = chunk['content']
                    current_merged_score = chunk['score']
                    current_merged_ids = [chunk['id']]
                    current_merged_metadata = chunk['metadata']
                    current_merged_chunk_size = chunk['metadata']['chunk_size']
                    current_merged_overlap_size = chunk['metadata']['overlap_size']

        


    # Log the merged chunks for debugging
    print("Merged Chunks:")
    for chunk in merged_chunks:
        # Show all metadata fields for each chunk and the content
        print(f"Score: {chunk['score']} | Metadata: {chunk['metadata']}")
        print(f"Content: {chunk['content']}")




    return retrieved_chunks
