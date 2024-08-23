# scripts/chatbot_functions.py

import logging
from faiss_utils import similarity_search_with_score
from document_processing import normalize_text

def chatbot_response(input_text, context, history):
    """
    Handle user input, process it, retrieve relevant documents, and generate a response.
    Args:
        input_text (str): The user's input.
        context (dict): Context containing client, memory, and other settings.
        history (list): Session state storing chat history.
    Returns:
        Tuple: Updated chat history, references, cleared input, and session state.
    """

    # Normalize the user's input text
    normalized_input = normalize_text(input_text)
    logging.info(f"Normalized input: {normalized_input}")

    # Attempt to retrieve relevant documents using similarity search
    try:
        search_results = similarity_search_with_score(
            normalized_input, context["vector_store"], context["embeddings"], context["EMBEDDING_DIM"]
        )
        logging.info(f"Retrieved documents with scores")
    except KeyError as e:
        logging.error(f"Error while retrieving documents: {e}")
        return history, "Error in retrieving documents.", "", history

    # Filter the results based on a similarity score threshold
    filtered_results = [
        result for result in search_results if result['score'] >= context["SIMILARITY_THRESHOLD"]
    ]
    logging.info(
        f"Filtered results by similarity threshold: {[result['score'] for result in filtered_results]}"
    )

    # Remove duplicates based on the content of the documents
    seen_contents = set()
    unique_filtered_results = []
    for result in filtered_results:
        content_hash = hash(result['content'])
        if content_hash not in seen_contents:
            unique_filtered_results.append(result)
            seen_contents.add(content_hash)

    # Sort the filtered results by similarity score in descending order
    unique_filtered_results.sort(key=lambda x: x['score'], reverse=True)
    filtered_docs = [
        result for result in unique_filtered_results[:context["TOP_SIMILARITY_RESULTS"]]
    ]
    
    # Log top similarity results
    logging.info(
        f"Top similarity results: {[(res['id'], res['score']) for res in unique_filtered_results[:context['TOP_SIMILARITY_RESULTS']]]}"
    )

    # Combine content from filtered documents to form the input for the LLM
    combined_input = f"{context['SYSTEM_PROMPT']}\n\n"
    combined_input += "\n\n".join(
        [
            f"{idx+1}. Context Document {doc['metadata'].get('doc_id', '')} - Chunk {doc['id']} | Path: {doc['metadata'].get('filepath', '')}/{doc['metadata'].get('filename', '')}\n{doc['content']}"
            for idx, doc in enumerate(filtered_docs)
        ]
    )
    combined_input += f"\n\nUser Prompt:\n{input_text}"

    # Log the final content sent to the LLM
    logging.info(f"Final content sent to LLM:\n{combined_input}")

    # Include previous chat history in the conversation
    messages = [{"role": "system", "content": context["SYSTEM_PROMPT"]}]
    for h in history:
        messages.append({"role": "user", "content": h})
    
    # Append the current input to the messages
    messages.append({"role": "user", "content": combined_input})

    # Generate the LLM response
    try:
        response = context["client"].chat.completions.create(
            model=context["LLM_MODEL"],
            messages=messages,
            max_tokens=min(
                context["MAX_TOKENS"] - len(context["encoding"].encode(str(messages))), 8000
            ),  # Adjust the max tokens for completion
        )
        logging.info(f"Generated LLM response successfully")
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return history, "Error generating response.", "", history

    # Update the conversation history with the new response
    history.append(f"User: {input_text}\nBot: {response.choices[0].message.content}")

    # Construct reference list
    references = "References:\n" + "\n".join(
        [
            f"[Document {doc['metadata'].get('doc_id', '')} - Chunk {doc['id']}: {doc['metadata'].get('filepath', '')}/{doc['metadata'].get('filename', '')}]"
            for doc in filtered_docs
        ]
    )

    # Return updated history, references, cleared input, and session state
    return "\n".join(history), references, "", history



def clear_history(context, history):
    """
    Clear the chat history and reset the session state.
    Args:
        context (dict): Context containing memory and other settings.
        history (list): Session state to be cleared.
    Returns:
        Tuple: Cleared chat history, references, input field, and session state.
    """
    context["memory"].clear()  # Clear the conversation memory
    history.clear()  # Clear the history in the session state
    return "", "", "", history
