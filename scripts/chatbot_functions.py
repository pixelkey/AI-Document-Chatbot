# scripts/chatbot_functions.py

import logging
from faiss_utils import similarity_search_with_score
from document_processing import normalize_text
import config

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
    filtered_docs = unique_filtered_results[:context["TOP_SIMILARITY_RESULTS"]]

    # Log top similarity results
    logging.info(
        f"Top similarity results: {[(res['id'], res['score']) for res in filtered_docs]}"
    )

    # Combine content from filtered documents to form the context for the LLM
    context_documents = "\n\n".join(
        [
            f"{idx+1}. Context Document {doc['metadata'].get('doc_id', '')} - Chunk {doc['id']} | Path: {doc['metadata'].get('filepath', '')}/{doc['metadata'].get('filename', '')}\n{doc['content']}"
            for idx, doc in enumerate(filtered_docs)
        ]
    )

    # Build the conversation history
    conversation_history = "\n".join(history)

    # Build the final prompt to be sent to the LLM
    if config.MODEL_SOURCE == "openai":
        # For OpenAI client, use messages
        messages = [{"role": "system", "content": context["SYSTEM_PROMPT"]}]
        if conversation_history:
            messages.append({"role": "assistant", "content": conversation_history})
        messages.append({"role": "user", "content": input_text})

        # Log the messages being sent
        logging.info(f"Messages sent to OpenAI API: {messages}")

        # Generate the LLM response
        try:
            response = context["client"].chat.completions.create(
                model=context["LLM_MODEL"],
                messages=messages,
                max_tokens=min(
                    context["LLM_MAX_TOKENS"] - len(context["encoding"].encode(str(messages))), 8000
                ),  # Adjust the max tokens for completion
            )
            # Extract the response content
            response_text = response.choices[0].message.content
            logging.info(f"Generated LLM response successfully")
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return history, "Error generating response.", "", history

    elif config.MODEL_SOURCE == "local":
        # For local model client, build a prompt that includes the system prompt, conversation history, and user input
        prompt = f"{context['SYSTEM_PROMPT']}\n\n"

        if conversation_history:
            prompt += f"Conversation History:\n{conversation_history}\n\n"

        prompt += f"Context Documents:\n{context_documents}\n\nUser Prompt:\n{input_text}"

        # Log the final prompt sent to the LLM
        logging.info(f"Final prompt sent to local LLM:\n{prompt}")

        # Calculate the max tokens for the model
        max_tokens = min(
            context["LLM_MAX_TOKENS"] - len(context["encoding"].encode(prompt)), 8000
        )

        # Generate the LLM response
        try:
            response = context["client"].invoke(
                prompt,
                max_tokens=max_tokens,
            )
            # The response is a string
            response_text = response
            logging.info(f"Generated LLM response successfully")
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return history, "Error generating response.", "", history

    # Update the conversation history with the new exchange
    history.append(f"User: {input_text}\nBot: {response_text}")

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
