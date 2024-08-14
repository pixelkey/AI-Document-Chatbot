# scripts/chatbot_functions.py

import logging
from faiss_utils import similarity_search_with_score
from document_processing import normalize_text


def chatbot_response(input_text, context):
    # Normalize input text
    normalized_input = normalize_text(input_text)
    logging.info(f"Normalized input: {normalized_input}")

    # Retrieve relevant documents with similarity scores
    try:
        search_results = similarity_search_with_score(
            normalized_input, context["vector_store"], context["embeddings"], context["EMBEDDING_DIM"]
        )
        logging.info(f"Retrieved documents with scores")
    except KeyError as e:
        logging.error(f"Error while retrieving documents: {e}")
        return context["memory"].buffer, "Error in retrieving documents.", ""

    # Filter documents based on similarity score and limit to top results with the token limit consideration
    filtered_results = [
        result for result in search_results if result[1] >= context["SIMILARITY_THRESHOLD"]
    ]
    logging.info(
        f"Filtered results by similarity threshold: {[result[1] for result in filtered_results]}"
    )

    # Remove duplicates by content
    seen_contents = set()
    unique_filtered_results = []
    for result in filtered_results:
        content_hash = hash(result[0].page_content)
        if content_hash not in seen_contents:
            unique_filtered_results.append(result)
            seen_contents.add(content_hash)

    # Sort filtered results by similarity score in descending order
    unique_filtered_results.sort(key=lambda x: x[1], reverse=True)
    filtered_docs = [
        result[0] for result in unique_filtered_results[:context["TOP_SIMILARITY_RESULTS"]]
    ]
    logging.info(
        f"Top similarity results: {[(doc.metadata['id'], score) for doc, score in unique_filtered_results[:context['TOP_SIMILARITY_RESULTS']]]}"
    )

    # Create the final combined input
    combined_input = f"{context['SYSTEM_PROMPT']}\n\n"
    combined_input += "\n\n".join(
        [
            f"Context Document {idx+1} ({doc.metadata.get('filepath', '')}/{doc.metadata['filename']}):\n{doc.page_content}"
            for idx, doc in enumerate(filtered_docs)
        ]
    )
    combined_input += f"\n\nUser Prompt:\n{input_text}"

    logging.info(f"Prepared combined input for LLM")

    # log the combined input for debugging
    logging.info(f"Combined input: {combined_input}")

    # Create the list of messages for the Chat API
    messages = [
        {"role": "system", "content": context["SYSTEM_PROMPT"]},
    ]
    for idx, doc in enumerate(filtered_docs):
        messages.append(
            {
                "role": "user",
                "content": f"Context Document {idx+1} ({doc.metadata.get('filepath', '')}/{doc.metadata['filename']}): {doc.page_content}",
            }
        )
    messages.append({"role": "user", "content": f"User Prompt: {input_text}"})

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
        return context["memory"].buffer, "Error generating response.", ""

    # Update conversation memory
    context["memory"].save_context(
        {"input": input_text}, {"output": response.choices[0].message.content}
    )

    # Construct reference list with clickable links
    references = "References:\n" + "\n".join(
        [
            f"[Chunk {doc.metadata['id']}: {doc.metadata.get('filepath', '')}/{doc.metadata['filename']}]"
            for doc in filtered_docs
        ]
    )

    # Return chat history, references, and clear input
    return context["memory"].buffer, references, ""


def clear_history(context):
    context["memory"].clear()  # Clear the conversation memory
    return "", "", ""
