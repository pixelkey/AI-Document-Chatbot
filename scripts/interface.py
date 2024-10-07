# scripts/interface.py

import gradio as gr
from chatbot_functions import chatbot_response, clear_history, retrieve_and_format_references

def setup_gradio_interface(context):
    """
    Sets up the Gradio interface.
    Args:
        context (dict): Context containing client, memory, and other settings.
    Returns:
        gr.Blocks: Gradio interface object.
    """
    with gr.Blocks(css=".separator { margin: 8px 0; border-bottom: 1px solid #ddd; }") as app:
        # Output fields
        chat_history = gr.Textbox(label="Chat History", lines=10, interactive=False)
        references = gr.Textbox(label="References", lines=2, interactive=False)

        # Input fields
        input_text = gr.Textbox(label="Input Text")
        submit_button = gr.Button("Submit")
        clear_button = gr.Button("Clear")

        # Initialize session state separately for each user
        session_state = gr.State(value=[])

        # Define a function to handle both reference retrieval and LLM response generation
        def handle_user_input(input_text, history):
            references, filtered_docs, context_documents = retrieve_and_format_references(input_text, context)
            # Update the interface with the references immediately
            yield "\n---\n".join([f"{u}\n{b}" for u, b in history]), references, input_text, history

            # Generate the LLM response if references were found
            if filtered_docs:
                new_history, response, _ = chatbot_response(input_text, context_documents, context, history)
                yield new_history, references, "", history
            else:
                # Return the original history and input if no relevant documents were found
                yield "\n---\n".join([f"{u}\n{b}" for u, b in history]), references, "", history

        # Setup event handlers with explicit state management
        submit_button.click(
            handle_user_input,
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state],
        )
        clear_button.click(
            lambda history: clear_history(context, history),
            inputs=[session_state],
            outputs=[chat_history, references, input_text, session_state],
        )

        input_text.submit(
            handle_user_input,
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state],
        )

        # Layout
        gr.Column([chat_history, references, input_text, submit_button, clear_button])

    return app
