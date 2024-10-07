# scripts/chatbot_functions.py

import logging
from faiss_utils import similarity_search_with_score
from document_processing import normalize_text
import config

def retrieve_and_format_references(input_text, context):
    """
    Retrieve relevant documents and format references.
    Args:
        input_text (str): The user's input.
        context (dict): Context containing client, memory, and other settings.
    Returns:
        Tuple: references, filtered_docs, and context_documents.
    """
    # Normalize the user's input text
    normalized_input = normalize_text(input_text)
    logging.info(f"Normalized input: {normalized_input}")

    # Retrieve relevant documents
    filtered_docs = retrieve_relevant_documents(normalized_input, context)
    if not filtered_docs:
        return "No relevant documents found.", None, None

    # Construct the references
    references = build_references(filtered_docs)

    # Build the context documents for LLM prompt
    context_documents = build_context_documents(filtered_docs)

    return references, filtered_docs, context_documents

def chatbot_response(input_text, context_documents, context, history):
    """
    Handle user input, generate a response, and update the conversation history.
    Args:
        input_text (str): The user's input.
        context_documents (str): The context documents for the LLM.
        context (dict): Context containing client, memory, and other settings.
        history (list): Session state storing chat history.
    Returns:
        Tuple: Updated chat history, LLM response, and cleared input.
    """
    # Ensure that history is treated as a list
    if not isinstance(history, list):
        history = []

    # Generate the response based on the model source
    response_text = generate_response(input_text, context_documents, context, history)
    if response_text is None:
        return history, "Error generating response.", ""

    # Update the conversation history with the new exchange
    history.append(f"User: {input_text}\nBot: {response_text}")

    # Return updated history, LLM response, and cleared input
    return "\n".join(history), response_text, ""



def retrieve_relevant_documents(normalized_input, context):
    """
    Retrieve relevant documents using similarity search.
    """
    try:
        search_results = similarity_search_with_score(
            normalized_input, context["vector_store"], context["embeddings"], context["EMBEDDING_DIM"]
        )
        logging.info("Retrieved documents with scores.")
    except KeyError as e:
        logging.error(f"Error while retrieving documents: {e}")
        return None

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

    return filtered_docs

def build_context_documents(filtered_docs):
    """
    Combine content from filtered documents to form the context documents.
    """
    context_documents = "\n\n".join(
        [
            f"{idx+1}. Context Document {doc['metadata'].get('doc_id', '')} - Chunk {doc['id']} | Path: {doc['metadata'].get('filepath', '')}/{doc['metadata'].get('filename', '')}\n{doc['content']}"
            for idx, doc in enumerate(filtered_docs)
        ]
    )
    return context_documents

def build_references(filtered_docs):
    """
    Construct the reference list from filtered documents.
    """
    references = "References:\n" + "\n".join(
        [
            f"[Document {doc['metadata'].get('doc_id', '')} - Chunk {doc['id']}: {doc['metadata'].get('filepath', '')}/{doc['metadata'].get('filename', '')}]\n{doc['content']}\n"
            for doc in filtered_docs
        ]
    )
    return references

def generate_response(input_text, context_documents, context, history):
    """
    Generate the LLM response based on the model source.
    """
    try:
        if config.MODEL_SOURCE == "openai":
            return generate_openai_response(input_text, context_documents, context, history)
        elif config.MODEL_SOURCE == "local":
            return generate_local_response(input_text, context_documents, context, history)
        else:
            logging.error(f"Unsupported MODEL_SOURCE: {config.MODEL_SOURCE}")
            return None
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return None

def generate_openai_response(input_text, context_documents, context, history):
    """
    Generate response using OpenAI API.
    """
    messages = []

    # System prompt
    messages.append({"role": "system", "content": context['SYSTEM_PROMPT']})

    # Add context documents as a system message
    messages.append({"role": "system", "content": f"Context Documents:\n{context_documents}"})

    # Add conversation history
    if history:
        for exchange in history:
            if "\nBot: " in exchange:
                user_part, bot_part = exchange.split("\nBot: ")
                user_text = user_part.replace("User: ", "")
                bot_text = bot_part
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": bot_text})
            else:
                # Handle cases where the exchange doesn't split as expected
                messages.append({"role": "user", "content": exchange})

    # Add the current user input
    messages.append({"role": "user", "content": input_text})

    # Log the messages being sent
    logging.info(f"Messages sent to OpenAI API: {messages}")

    # Calculate max tokens
    try:
        tokens_consumed = len(context["encoding"].encode(str(messages)))
        max_tokens = min(
            context["LLM_MAX_TOKENS"] - tokens_consumed, 8000
        )
    except Exception as e:
        logging.warning(f"Token encoding error: {e}")
        max_tokens = 8000  # Fallback to default max tokens

    # Generate the LLM response
    try:
        response = context["client"].chat.completions.create(
            model=context["LLM_MODEL"],
            messages=messages,
            max_tokens=max_tokens,
        )
        # Extract the response content
        response_text = response.choices[0].message.content
        logging.info("Generated LLM response successfully.")
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return None

    return response_text

def generate_local_response(input_text, context_documents, context, history):
    """
    Generate response using local model.
    """
    # Build the prompt for the local model
    prompt = build_local_prompt(context['SYSTEM_PROMPT'], history, context_documents, input_text)

    # Log the final prompt sent to the local LLM
    logging.info(f"Final prompt sent to local LLM:\n{prompt}")

    # Calculate the max tokens for the model
    try:
        tokens_consumed = len(context["encoding"].encode(prompt))
        max_tokens = min(
            context["LLM_MAX_TOKENS"] - tokens_consumed, 8000
        )
    except Exception as e:
        logging.warning(f"Token encoding error: {e}")
        max_tokens = 8000  # Fallback to default max tokens

    # Generate the LLM response
    try:
        response_text = context["client"].invoke(
            prompt,
            max_tokens=max_tokens,
        )
        logging.info("Generated LLM response successfully.")
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return None

    return response_text

def build_local_prompt(system_prompt, history, context_documents, input_text):
    """
    Build the prompt for the local model, including conversation history and context documents.
    """
    prompt = f"{system_prompt}\n\n"

    if history:
        prompt += "Conversation History:\n"
        for exchange in history:
            prompt += f"{exchange}\n"
        prompt += "\n"

    prompt += f"Context Documents:\n{context_documents}\n\nUser Prompt:\n{input_text}"
    return prompt



def clear_history(context, history):
    """
    Clear the chat history and reset the session state.
    Args:
        context (dict): Context containing memory and other settings.
        history (list): Session state to be cleared.
    Returns:
        Tuple: Cleared chat history, cleared references, cleared input field, and session state.
    """
    # Ensure history is treated as a list before clearing
    if not isinstance(history, list):
        history = []

    context["memory"].clear()  # Clear the conversation memory
    history.clear()  # Clear the history in the session state
    
    return "", "", "", history

